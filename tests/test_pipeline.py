import asyncio
import json

from visionbench.analysis.aggregate import aggregate
from visionbench.analysis.report import render
from visionbench.config import load_settings
from visionbench.harness.runner import Run
from visionbench.tasks import dataset as ds


def test_mock_end_to_end(tmp_path):
    settings = load_settings().with_overrides({"run": {"dataset_dir": str(tmp_path / "d"), "output_dir": str(tmp_path / "r")}})
    spec = ds.DatasetSpec(name="t", families=["counting", "maze_path", "gear_train"], difficulties=[1, 3],
                          per_cell=3, rerenders=1)
    root = ds.build(spec, tmp_path / "d")
    run = Run(settings, root, tmp_path / "r" / "x", models=["mock:oracle", "mock:noisy-0.6"], mode="direct",
              conditions=["vision", "blind"], samples=2)
    asyncio.run(run.collect(progress=False))
    asyncio.run(run.verify(progress=False))
    # resumability: a second collect adds nothing
    n = len((tmp_path / "r" / "x" / "attempts.jsonl").read_text().splitlines())
    asyncio.run(run.collect(progress=False))
    assert len((tmp_path / "r" / "x" / "attempts.jsonl").read_text().splitlines()) == n
    res = aggregate(tmp_path / "r" / "x", settings)
    by = {s["model"]: s for s in res["subjects"]}
    assert by["mock:oracle"]["accuracy"]["mean"] == 1.0
    assert by["mock:noisy-0.6"]["accuracy"]["mean"] < 1.0
    assert by["mock:oracle"]["render_consistency"]["solved_on_all_renders"] == 1.0
    assert "blind" in by["mock:oracle"]
    out = render(res, tmp_path / "report.html")
    assert "<svg" in out.read_text()
    json.dumps(res, default=str)
