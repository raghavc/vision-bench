import asyncio
import json
import random

from visionbench.analysis.aggregate import aggregate
from visionbench.analysis.report import render
from visionbench.config import load_settings
from visionbench.harness.runner import Run
from visionbench.tasks import dataset as ds
from visionbench.tasks.key_facts import crop_grounding, key_facts
from visionbench.verifiers.judge import ERROR_TYPES, STRATEGIES
from visionbench.verifiers.reasoning import derive


def test_derive_metrics():
    j = {"observations": [{"claim": "a", "verdict": "correct"}, {"claim": "b", "verdict": "incorrect"}],
         "key_facts": [{"index": 0, "status": "stated_correctly"}, {"index": 1, "status": "stated_incorrectly"}],
         "reasoning_valid": True, "answer_supported": False, "error_type": "perception_misread",
         "strategy": "decomposition", "self_verification": False, "efficiency": "adequate"}
    m = derive(j, 3, "correct")
    assert m["perception_precision"] == 0.5 and abs(m["key_fact_recall"] - 1 / 3) < 1e-9
    assert m["unsupported_success"] and not m["sound_success"]
    assert derive(j, 3, "incorrect")["failure_locus"] == "perception"


def test_crop_grounding():
    from visionbench.tasks.base import get_family
    inst = get_family("visual_search").generate(0, 2)
    boxes = inst.facts["evidence_boxes"]
    traj = [{"event": "crop", "box": [b[0] - 5, b[1] - 5, b[2] + 5, b[3] + 5]} for b in boxes[:1]]
    g = crop_grounding(traj, inst)
    assert g["n_crops"] == 1 and abs(g["evidence_coverage"] - 1 / len(boxes)) < 1e-9


def test_reasoning_profiles_in_report(tmp_path):
    settings = load_settings().with_overrides({"run": {"dataset_dir": str(tmp_path / "d"), "output_dir": str(tmp_path / "r")}})
    root = ds.build(ds.DatasetSpec(name="t", families=["gear_train", "clock_reasoning"], difficulties=[2, 4],
                                   per_cell=3), tmp_path / "d")
    run_dir = tmp_path / "r" / "x"
    run = Run(settings, root, run_dir, models=["mock:noisy-0.7", "mock:noisy-0.5"], mode="direct",
              conditions=["vision"], samples=1)
    asyncio.run(run.collect(progress=False))
    asyncio.run(run.verify(progress=False))
    items = {it.uid: it for it in ds.load(root)}
    rnd = random.Random(0)
    with open(run_dir / "critiques.jsonl", "w") as f:
        for v in map(json.loads, (run_dir / "verdicts.jsonl").read_text().splitlines()):
            model = v["attempt_key"].split("|")[0]
            kf = key_facts(items[v["uid"]])
            j = {"observations": [{"claim": "x", "verdict": rnd.choice(["correct", "incorrect"])}],
                 "key_facts": [{"index": i, "status": rnd.choice(["stated_correctly", "stated_incorrectly",
                                                                    "not_mentioned"])} for i in range(len(kf))],
                 "reasoning_valid": rnd.random() < 0.7, "answer_supported": rnd.random() < 0.8,
                 "first_error_step": "", "error_type": rnd.choice(ERROR_TYPES), "strategy": rnd.choice(STRATEGIES),
                 "self_verification": rnd.random() < 0.4, "efficiency": "adequate", "strengths": "s",
                 "weaknesses": "w", "critique": "c"}
            f.write(json.dumps({"attempt_key": v["attempt_key"], "uid": v["uid"], "model": model, "mode": "direct",
                                "family": items[v["uid"]].family, "difficulty": items[v["uid"]].difficulty,
                                "sample": 0, "outcome": v["outcome"], "key_facts": kf, "judge": j,
                                "grounding": None, "metrics": derive(j, len(kf), v["outcome"])}) + "\n")
    res = aggregate(run_dir, settings)
    assert set(res["reasoning"]["subjects"]) == {"mock:noisy-0.7 [direct]", "mock:noisy-0.5 [direct]"}
    assert res["reasoning"]["head_to_head"][0]["shared"] == len(items)
    html = render(res, tmp_path / "r.html").read_text()
    assert "Reasoning analysis" in html and "Head-to-head" in html
