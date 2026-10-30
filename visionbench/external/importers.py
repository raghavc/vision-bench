"""Importers for public vision-reasoning benchmarks on the Hugging Face Hub.

Each importer turns rows into TaskInstances (family "ext:<name>") and writes a
regular VisionBench dataset, so external sets go through the same harness,
verifiers and statistics as the procedural families. Revisions are pinned to
the dataset commit SHA at import time (recorded in the manifest).

Chosen for reasoning headroom in 2026, not knowledge recall (see
METHODOLOGY.md):
  visulogic, babyvision, mmsi, visualpuzzles, o3bench, hss (Scale AI, rubric),
  emma (mini), zerobench (gated: needs an HF token), mathvision (calibration anchor).

Multiple-choice items keep their letter answers. Chance level is stored per
item so reports can show chance-normalized accuracy. Free-form answers are
checked programmatically first, then (only for external sets) by the
reference-guided equivalence judge.
"""

from __future__ import annotations

import ast
import io
import json
import random
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from visionbench.config import Settings
from visionbench.core.types import AnswerSpec, RubricItem, TaskInstance, make_uid
from visionbench.tasks.dataset import save_instance

LETTERS = "ABCDEFGHIJ"
CONVERT = "refs/convert/parquet"


# ------------------------------------------------------------------ hub utils
class Hub:
    def __init__(self, repo: str, token: str | None):
        from huggingface_hub import HfApi
        self.repo = repo
        self.token = token
        self.api = HfApi(token=token)
        info = self.api.dataset_info(repo)
        self.sha = info.sha
        self.gated = getattr(info, "gated", False)

    def file(self, filename: str, revision: str | None = None) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(self.repo, filename, repo_type="dataset", revision=revision or self.sha,
                                    token=self.token))

    def parquet_rows(self, config: str, split: str, columns: list[str] | None = None) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq
        rows: list[dict[str, Any]] = []
        k = 0
        while True:
            try:
                p = self.file(f"{config}/{split}/{k:04d}.parquet", revision=CONVERT)
            except Exception:
                if k == 0:
                    raise
                break
            rows += pq.read_table(p, columns=columns).to_pylist()
            k += 1
        return rows


def pil(x: Any) -> Image.Image | None:
    if x is None:
        return None
    if isinstance(x, dict):
        if x.get("bytes"):
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
        return None
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    return None


def stratified(rows: list[dict], key: Callable[[dict], str], limit: int | None, seed: int) -> list[dict]:
    if not limit or limit >= len(rows):
        return rows
    rnd = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for g in groups.values():
        rnd.shuffle(g)
    out: list[dict] = []
    # round-robin across strata, proportional-ish
    while len(out) < limit:
        progressed = False
        for g in groups.values():
            if g and len(out) < limit:
                out.append(g.pop())
                progressed = True
        if not progressed:
            break
    return out


def mc_prompt(question: str, options: list[str] | None) -> str:
    q = question.strip()
    if options:
        q += "\n\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return q


def mc_spec(answer: str, n: int) -> AnswerSpec:
    letters = list(LETTERS[:n])
    return AnswerSpec(kind="choice", value=answer.strip().strip("()").upper()[:1], choices=letters,
                      format_hint=f"the single letter of the correct option ({', '.join(letters)})")


def free_spec(answer: str) -> AnswerSpec:
    a = str(answer).strip()
    try:
        float(a.replace(",", ""))
        return AnswerSpec(kind="number", value=float(a.replace(",", "")), tolerance=1e-6, rel_tolerance=1e-4,
                          format_hint="just the final value")
    except ValueError:
        return AnswerSpec(kind="text", value=a, format_hint="just the final answer, as short as possible")


def make(name: str, rid: Any, prompt: str, spec: AnswerSpec, images: list[Image.Image], category: str = "",
         difficulty: int = 0, facts: dict | None = None, rubric: list[RubricItem] | None = None) -> TaskInstance:
    f = {"source_id": str(rid), "category": category, "n_options": len(spec.choices or []) or None,
         "chance": (1 / len(spec.choices)) if spec.choices else 0.0, "lenient_equivalence": spec.kind in ("text", "number")}
    f.update(facts or {})
    uid = make_uid("ext", name, rid)
    return TaskInstance(family=f"ext:{name}", version="1", seed=0, difficulty=difficulty, prompt=prompt, answer=spec,
                        images=[im for im in images if im is not None], facts=f, rubric=rubric or [],
                        skills=[category] if category else [], uid=uid, cluster=uid)


# ------------------------------------------------------------------ sources
def visulogic(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = [json.loads(l) for l in hub.file("data.jsonl").read_text().splitlines() if l.strip()]
    rows = stratified(rows, lambda r: r["tag"], limit, seed)
    zf = zipfile.ZipFile(hub.file("images.zip"))
    names = {Path(n).name: n for n in zf.namelist()}
    for r in rows:
        img = Image.open(io.BytesIO(zf.read(names[Path(r["image_path"]).name]))).convert("RGB")
        n = len(re.findall(r"^[A-D]:", r["question"], re.M)) or 4
        yield make("visulogic", r["id"], r["question"], mc_spec(r["label"], n), [img], r["tag"])


def babyvision(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = hub.parquet_rows("default", "train")
    rows = stratified(rows, lambda r: r["type"], limit, seed)
    for r in rows:
        if r["ansType"] == "choice" and r.get("options"):
            spec = mc_spec(LETTERS[int(r["choiceAns"])], len(r["options"]))
            prompt = mc_prompt(r["question"], r["options"])
        else:
            spec = free_spec(r["blankAns"])
            prompt = r["question"]
        yield make("babyvision", r["taskId"], prompt, spec, [pil(r["image"])], f"{r['type']}/{r['subtype']}")


def mmsi(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = hub.parquet_rows("default", "test")
    rows = stratified(rows, lambda r: r["question_type"], limit, seed)
    for r in rows:
        imgs = [pil(x) for x in (r["images"] or [])]
        n = len(re.findall(r"\b[A-F]:", r["question"])) or 4
        diff = {0: 1, 1: 3, 2: 5}.get(r.get("difficulty"), 0)
        yield make("mmsi", r["id"], r["question"], mc_spec(r["answer"], n), imgs, r["question_type"], diff)


def visualpuzzles(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = hub.parquet_rows("default", "train")
    rows = stratified(rows, lambda r: f"{r['category']}/{r['difficulty']}", limit, seed)
    for r in rows:
        opts = r.get("options")
        prompt = mc_prompt(r["question"], opts) if opts else r["question"] + "\n\n(The options A-D are shown in the image.)"
        diff = {"easy": 1, "medium": 3, "hard": 5}.get(r.get("difficulty"), 0)
        yield make("visualpuzzles", r["id"], prompt, mc_spec(r["answer"], 4), [pil(r["image"])], r["category"], diff)


def mathvision(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = hub.parquet_rows("default", "testmini")
    rows = stratified(rows, lambda r: r["subject"], limit, seed)
    for r in rows:
        q = re.sub(r"<image\d*>", "", r["question"]).strip()
        if r.get("options"):
            spec, prompt = mc_spec(r["answer"], len(r["options"])), mc_prompt(q, r["options"])
        else:
            spec, prompt = free_spec(r["answer"]), q
        yield make("mathvision", r["id"], prompt, spec, [pil(r["decoded_image"])], r["subject"], int(r.get("level") or 0))


def emma(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = []
    for cfg in ("Math", "Physics", "Chemistry", "Coding"):
        rows += hub.parquet_rows(cfg, "test")
    rows = stratified(rows, lambda r: f"{r['subject']}/{r['category']}", limit, seed)
    for r in rows:
        imgs = [pil(r.get(f"image_{k}")) for k in range(1, 6)]
        imgs = [i for i in imgs if i is not None]
        q = r["question"]
        if r.get("context"):
            q = r["context"] + "\n\n" + q
        q = re.sub(r"<image_(\d)>", lambda m: f"[Image {int(m.group(1)) - 1}]", q)
        if str(r["type"]).lower().startswith("multiple"):
            opts = r.get("options") or []
            opts = [re.sub(r"<image_(\d)>", lambda m: f"[Image {int(m.group(1)) - 1}]", o) for o in opts]
            letters_only = all(re.fullmatch(r"[A-Ea-e]", o or "") for o in opts)
            prompt = q if letters_only else mc_prompt(q, opts)
            spec = mc_spec(r["answer"].upper(), len(opts) or 5)
        else:
            ans = r["answer"]
            if ans.startswith("["):
                try:
                    ans = ast.literal_eval(ans)[-1]
                except (ValueError, SyntaxError):
                    pass
            spec, prompt = free_spec(ans), q
        yield make("emma", r["pid"], prompt, spec, imgs, f"{r['subject']}/{r['category']}")


def zerobench(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = hub.parquet_rows("default", "zerobench")
    rows = stratified(rows, lambda r: "all", limit, seed)
    for r in rows:
        imgs = [pil(x) for x in (r.get("question_images_decoded") or [])]
        yield make("zerobench", r["question_id"], r["question_text"], free_spec(r["question_answer"]), imgs, "zerobench", 5)


def o3bench(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = [json.loads(l) for l in hub.file("test/metadata.jsonl").read_text().splitlines() if l.strip()]
    rows = stratified(rows, lambda r: str(r.get("category") or r.get("type") or Path(r["file_name"]).parent), limit, seed)
    for i, r in enumerate(rows):
        img = Image.open(hub.file(f"test/{r['file_name']}")).convert("RGB")
        opts = r["options"]
        n = len(re.findall(r"^[A-F]\.", opts, re.M)) or 6
        prompt = f"{r['question'].strip()}\n\nOptions:\n{opts.strip()}"
        rid = r.get("id") or f"{r['file_name']}#{make_uid(r['question'])}"  # several questions share one image
        yield make("o3bench", rid, prompt, mc_spec(r["answer"], n), [img], str(Path(r["file_name"]).parent), 5)


def hss(hub: Hub, limit, seed) -> Iterable[TaskInstance]:
    rows = [json.loads(l) for l in hub.file("data/test.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("media_type") == "image"]
    rows = stratified(rows, lambda r: r["subdomain"], limit, seed)
    for r in rows:
        img = Image.open(hub.file(r["media_path"])).convert("RGB")
        crits = r.get("rubric_criteria") or []
        if isinstance(crits, str):
            crits = json.loads(crits)
        rubric = [RubricItem(id=str(c.get("id", i)), criterion=c.get("title", str(c)), weight=1.0, critical=True)
                  for i, c in enumerate(crits)]
        spec = AnswerSpec(kind="rubric", value=r["golden_response"], format_hint="a concise answer with brief justification")
        yield make("hss", r["task_id"], r["prompt"], spec, [img], f"{r['domain']}/{r['subdomain']}",
                   facts={"golden_response": r["golden_response"]}, rubric=rubric)


SOURCES: dict[str, tuple[str, Callable]] = {
    "visulogic": ("VisuLogic/VisuLogic", visulogic),
    "babyvision": ("UnipatAI/BabyVision", babyvision),
    "mmsi": ("RunsenXu/MMSI-Bench", mmsi),
    "visualpuzzles": ("neulab/VisualPuzzles", visualpuzzles),
    "mathvision": ("MathLLMs/MathVision", mathvision),
    "emma": ("luckychao/EMMA-mini", emma),
    "emma-full": ("luckychao/EMMA", emma),
    "zerobench": ("jonathan-roberts1/zerobench", zerobench),
    "o3bench": ("m-Just/O3-Bench", o3bench),
    "hss": ("ScaleAI/HSS", hss),
}


def import_dataset(source: str, settings: Settings, name: str | None = None, limit: int | None = None,
                   seed: int = 0, split: str | None = None) -> Path:
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}; available: {', '.join(SOURCES)}")
    repo, fn = SOURCES[source]
    token = settings.section("huggingface").get("token") or None
    hub = Hub(repo, token)
    name = name or f"ext-{source}" + (f"-{limit}" if limit else "")
    root = settings.path("dataset_dir") / name
    (root / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.time()
    with open(root / "items.jsonl", "w") as f:
        for inst in fn(hub, limit, seed):
            if not inst.images:
                continue
            save_instance(inst, root)
            f.write(json.dumps(inst.to_record(), default=str) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  {n} items", flush=True)
    (root / "manifest.json").write_text(json.dumps({
        "source": source, "repo": repo, "revision": hub.sha, "limit": limit, "sample_seed": seed, "n_items": n,
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "build_seconds": round(time.time() - t0, 1),
        "license_note": "Data stays local and is never committed; respect each dataset's license/terms.",
    }, indent=2))
    return root
