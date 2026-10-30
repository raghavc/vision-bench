"""Human baselines: a self-contained HTML answer form, and import of its answers as a subject.

Humans answer a stratified sample of items in their browser (images at full
resolution, per-item timing, confidence slider). The exported JSON is imported
as attempts from subject `human:<name>`, so humans are verified and reported
exactly like models. This gives human-normalized scores and a per-family
human/model gap.
"""

from __future__ import annotations

import base64
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from visionbench.core.types import Attempt
from visionbench.tasks import dataset as ds

PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>VisionBench — human baseline</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;background:#fcfcfb;color:#0b0b0b}
.wrap{max-width:1100px;margin:0 auto;padding:24px}.item{display:none}.item.on{display:block}
img{max-width:100%;border:1px solid #e4e3df;border-radius:8px;margin:8px 0;cursor:zoom-in}img.full{max-width:none}
.prompt{white-space:pre-wrap;background:#f3f2ef;border-radius:8px;padding:12px 14px}.fmt{color:#52514e;font-size:13px}
textarea{width:100%;min-height:70px;font:14px ui-monospace,Menlo,monospace;padding:8px;border-radius:8px;border:1px solid #b9b8b2}
button{font:inherit;padding:8px 16px;border-radius:8px;border:0;background:#2a78d6;color:#fff;cursor:pointer}
.bar{display:flex;gap:12px;align-items:center;margin:12px 0}.muted{color:#7a7974}
</style></head><body><div class="wrap">
<h1>VisionBench human baseline</h1>
<p class="muted">Answer each item as accurately as you can. Click an image to toggle full size. Your answers stay in this page until you
download them. Progress: <span id="prog"></span></p>
<div id="items"></div>
<div class="bar"><button id="dl">Download answers (JSON)</button><span class="muted" id="saved"></span></div>
</div><script>
const ITEMS = __ITEMS__;
const ANNOTATOR = __NAME__;
const KEY = "vb-human-" + ANNOTATOR + "-" + __TAG__;
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let cur = 0, t0 = Date.now();
const root = document.getElementById("items");
ITEMS.forEach((it, i) => {
  const d = document.createElement("div"); d.className = "item"; d.id = "it" + i;
  d.innerHTML = `<h3>Item ${i + 1} / ${ITEMS.length}</h3>` +
    it.images.map(src => `<img src="${src}" onclick="this.classList.toggle('full')">`).join("") +
    `<div class="prompt"></div><p class="fmt">Answer format: <b></b></p>
     <textarea placeholder="Your answer"></textarea>
     <div class="bar">Confidence <input type="range" min="0" max="100" value="50"> <span>50%</span></div>
     <div class="bar"><button class="prev">← Back</button><button class="next">Save & next →</button></div>`;
  d.querySelector(".prompt").textContent = it.prompt;
  d.querySelector(".fmt b").textContent = it.format;
  const ta = d.querySelector("textarea"), rg = d.querySelector("input"), sp = d.querySelector(".bar span");
  if (state[it.uid]) { ta.value = state[it.uid].answer; rg.value = state[it.uid].confidence; sp.textContent = rg.value + "%"; }
  rg.oninput = () => sp.textContent = rg.value + "%";
  d.querySelector(".next").onclick = () => { save(i, ta.value, rg.value); show(Math.min(i + 1, ITEMS.length - 1)); };
  d.querySelector(".prev").onclick = () => show(Math.max(i - 1, 0));
  root.appendChild(d);
});
function save(i, ans, conf) {
  const it = ITEMS[i], prev = state[it.uid];
  const secs = (Date.now() - t0) / 1000 + (prev ? prev.seconds : 0);
  state[it.uid] = {answer: ans, confidence: Number(conf), seconds: secs};
  localStorage.setItem(KEY, JSON.stringify(state)); prog();
}
function show(i) { document.querySelectorAll(".item").forEach(e => e.classList.remove("on"));
  document.getElementById("it" + i).classList.add("on"); cur = i; t0 = Date.now(); window.scrollTo(0, 0); }
function prog() { document.getElementById("prog").textContent = Object.keys(state).length + " / " + ITEMS.length; }
document.getElementById("dl").onclick = () => {
  const blob = new Blob([JSON.stringify({annotator: ANNOTATOR, dataset: __TAG__, answers: state}, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `human_${ANNOTATOR}.json`; a.click();
  document.getElementById("saved").textContent = "Downloaded.";
};
prog(); show(0);
</script></body></html>"""


def _data_uri(img) -> str:
    buf = io.BytesIO()
    if img.width * img.height > 2_000_000:
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def make_form(dataset_root: Path, out: Path, name: str, n: int, seed: int = 0) -> Path:
    items = ds.load(dataset_root)
    # one rendering per problem, stratified over family x difficulty
    seen, strata = set(), defaultdict(list)
    for it in items:
        if it.cluster in seen or it.answer.kind == "rubric":
            continue
        seen.add(it.cluster)
        strata[(it.family, it.difficulty)].append(it)
    rnd = random.Random(seed)
    for v in strata.values():
        rnd.shuffle(v)
    chosen = []
    while len(chosen) < n and any(strata.values()):
        for k in sorted(strata):
            if strata[k] and len(chosen) < n:
                chosen.append(strata[k].pop())
    rnd.shuffle(chosen)
    payload = []
    for it in chosen:
        it.load_images(dataset_root)
        payload.append({"uid": it.uid, "prompt": it.prompt, "format": it.answer.format_hint,
                        "images": [_data_uri(im) for im in it.images]})
    html = (PAGE.replace("__ITEMS__", json.dumps(payload)).replace("__NAME__", json.dumps(name))
            .replace("__TAG__", json.dumps(dataset_root.name)))
    out.write_text(html)
    return out


def import_answers(answers_path: Path, dataset_root: Path, run_dir: Path) -> int:
    data = json.loads(answers_path.read_text())
    name = data["annotator"]
    items = {it.uid: it for it in ds.load(dataset_root)}
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"dataset": str(dataset_root), "models": [f"human:{name}"], "mode": "direct",
                                        "conditions": ["vision"], "samples": 1, "families": None}, indent=2))
    n = 0
    with open(run_dir / "attempts.jsonl", "a") as f:
        for uid, a in data["answers"].items():
            it = items.get(uid)
            if it is None or not str(a.get("answer", "")).strip():
                continue
            att = Attempt(uid=uid, family=it.family, cluster=it.cluster, difficulty=it.difficulty,
                          model=f"human:{name}", mode="direct", condition="vision", sample=0,
                          raw_answer=a["answer"], confidence=float(a.get("confidence", 50)) / 100,
                          final_text=a["answer"], latency_s=float(a.get("seconds", 0)), stop_reason="human")
            f.write(json.dumps(att.to_record(), default=str) + "\n")
            n += 1
    return n
