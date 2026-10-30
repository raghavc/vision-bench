"""VisionBench command line.

  visionbench families                         list task families
  visionbench preview gear_train -d 4          render one item (PNG + prompt + answer)
  visionbench build core-dev --per-cell 10     build a dataset (dev split by default)
  visionbench import visulogic --limit 300     import a public benchmark as a dataset
  visionbench estimate core-dev --models ...   rough token/cost estimate before spending
  visionbench run core-dev --models claude-opus-5-5 --samples 3 --conditions vision,blind
  visionbench verify <run>  /  critique <run>  /  report <run>  /  judge-audit <run>
  visionbench selftest                         oracle round-trip on every family (offline)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from visionbench.config import ROOT, load_settings


def _range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_families(args, settings):
    from visionbench.tasks.base import load_all
    for name, f in sorted(load_all().items()):
        print(f"{name:20s} v{f.version}  [{', '.join(f.skills)}]\n    {f.description}")


def cmd_preview(args, settings):
    from visionbench.tasks.base import get_family
    inst = get_family(args.family).generate(args.seed, args.difficulty, render_seed=args.render)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(inst.images):
        p = out / f"{args.family}_d{args.difficulty}_s{args.seed}_{k}.png"
        im.save(p)
        print(f"image: {p}  ({im.width}x{im.height})")
    print(f"\nPROMPT:\n{inst.prompt}\n\nFORMAT: {inst.answer.format_hint}\nANSWER: {json.dumps(inst.answer.value, default=str)}")
    if args.facts:
        print("\nFACTS:", json.dumps(inst.facts, default=str, indent=1)[:4000])


def cmd_build(args, settings):
    from visionbench.tasks import dataset as ds
    from visionbench.tasks.base import load_all
    fams = sorted(load_all()) if args.families == "all" else args.families.split(",")
    spec = ds.DatasetSpec(name=args.name, families=fams, difficulties=_range(args.difficulties),
                          per_cell=args.per_cell, split=args.split, rerenders=args.rerenders, start=args.start)
    t0 = time.time()
    root = ds.build(spec, settings.path("dataset_dir"), secret=settings.run.get("private_seed_secret"),
                    progress=lambda n: print(f"\r  {n} items", end="", file=sys.stderr) if n % 25 == 0 else None)
    print(f"\nbuilt {root} in {time.time() - t0:.0f}s")


def cmd_import(args, settings):
    from visionbench.external import importers
    root = importers.import_dataset(args.source, settings, name=args.name, limit=args.limit, seed=args.seed,
                                    split=args.split)
    print(f"imported -> {root}")


def _dataset_root(settings, name: str) -> Path:
    p = Path(name)
    return p if p.exists() else settings.path("dataset_dir") / name


def _settings_with_cli(args, settings):
    return settings.with_overrides({
        "subject": {"effort": args.effort, "max_tokens": args.max_tokens},
        "harness": {"mode": args.mode, "max_turns": args.max_turns, "code_execution": args.code_execution or None},
        "run": {"concurrency": args.concurrency, "process_judge": args.process_judge or None},
        "openrouter": {"budget_usd": args.budget},
    })


def cmd_run(args, settings):
    from visionbench.harness.runner import Run
    settings = _settings_with_cli(args, settings)
    models = args.models.split(",") if args.models else settings.subject.get("models", ["claude-opus-5-5"])
    conds = args.conditions.split(",") if args.conditions else (
        ["vision", "blind"] if settings.run.get("blind_control", True) else ["vision"])
    mode = args.mode or settings.harness.get("mode", "direct")
    samples = args.samples or int(settings.run.get("samples_per_item", 3))
    name = args.run_name or f"{Path(args.dataset).name}-{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = settings.path("output_dir") / name
    run = Run(settings, _dataset_root(settings, args.dataset), run_dir, models=models, mode=mode, conditions=conds,
              samples=samples, families=args.families.split(",") if args.families else None, limit=args.limit,
              per_family=args.per_family)
    print(f"[run] -> {run_dir}", file=sys.stderr)
    async def stages():  # one event loop for every stage: HTTP clients are bound to the loop that created them
        await run.collect()
        await run.verify()
        if args.critique:
            await run.critique()
    asyncio.run(stages())
    _report(run_dir, settings)


def cmd_critique(args, settings):
    from visionbench.harness.runner import Run
    run_dir = Path(args.run) if Path(args.run).exists() else settings.path("output_dir") / args.run
    m = json.loads((run_dir / "manifest.json").read_text())
    run = Run(settings, Path(m["dataset"]), run_dir, models=m["models"], mode=m["mode"], conditions=m["conditions"],
              samples=m["samples"], families=m.get("families"))
    asyncio.run(run.critique(all_samples=args.all_samples, limit=args.limit,
                             models=args.models.split(",") if args.models else None))
    _report(run_dir, settings)


def cmd_verify(args, settings):
    from visionbench.harness.runner import Run
    run_dir = Path(args.run) if Path(args.run).exists() else settings.path("output_dir") / args.run
    m = json.loads((run_dir / "manifest.json").read_text())
    run = Run(settings, Path(m["dataset"]), run_dir, models=m["models"], mode=m["mode"], conditions=m["conditions"],
              samples=m["samples"], families=m.get("families"))
    if args.reset:
        (run_dir / "verdicts.jsonl").unlink(missing_ok=True)
    asyncio.run(run.verify(process_judge=args.process_judge or None))
    _report(run_dir, settings)


def _report(run_dir: Path, settings):
    from visionbench.analysis.aggregate import aggregate
    from visionbench.analysis.report import render
    res = aggregate(run_dir, settings)
    (run_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
    out = render(res, run_dir / "report.html", title=f"VisionBench · {run_dir.name}")
    print(f"[report] {out}")
    for s in res["subjects"]:
        a = s["accuracy"]
        extra = ""
        if "blind" in s:
            extra += f"  blind {100 * s['blind']['mean']:.1f}%"
        if "caption" in s:
            extra += f"  caption {100 * s['caption']['mean']:.1f}%"
        print(f"  {s['label']:40s} {100 * a['mean']:5.1f}%  [{100 * a['ci95'][0]:.1f}, {100 * a['ci95'][1]:.1f}]"
              f"  n={a['n_items']}{extra}")


def cmd_report(args, settings):
    run_dir = Path(args.run) if Path(args.run).exists() else settings.path("output_dir") / args.run
    _report(run_dir, settings)


def cmd_judge_audit(args, settings):
    from visionbench.verifiers.audit import judge_audit
    run_dir = Path(args.run) if Path(args.run).exists() else settings.path("output_dir") / args.run
    res = asyncio.run(judge_audit(run_dir, settings, n=args.n))
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=1))
    (run_dir / "judge_audit.json").write_text(json.dumps(res, indent=1, default=str))


def cmd_critique_audit(args, settings):
    from visionbench.verifiers.audit import critique_consistency
    run_dir = Path(args.run) if Path(args.run).exists() else settings.path("output_dir") / args.run
    res = asyncio.run(critique_consistency(run_dir, settings, n=args.n))
    print(json.dumps(res, indent=1))
    (run_dir / "critique_consistency.json").write_text(json.dumps(res, indent=1))


def cmd_estimate(args, settings):
    from visionbench.harness.estimate import estimate
    models = args.models.split(",") if args.models else settings.subject.get("models", ["claude-opus-5-5"])
    print(estimate(_dataset_root(settings, args.dataset), settings, models, args.samples,
                   args.conditions.split(","), args.mode))


def cmd_human_form(args, settings):
    from visionbench.human import make_form
    root = _dataset_root(settings, args.dataset)
    out = Path(args.out or f"human_form_{args.name}.html")
    make_form(root, out, args.name, args.n, args.seed)
    print(f"form -> {out}  (send it to the annotator; they click 'Download answers' when done)")


def cmd_human_import(args, settings):
    from visionbench.human import import_answers
    run_dir = Path(args.run) if Path(args.run).exists() else settings.path("output_dir") / args.run
    if (run_dir / "manifest.json").exists():
        root = Path(json.loads((run_dir / "manifest.json").read_text())["dataset"])
    else:
        root = _dataset_root(settings, args.dataset)
    n = import_answers(Path(args.answers), root, run_dir)
    print(f"imported {n} human answers into {run_dir}; now run: visionbench verify {run_dir}")


def cmd_selftest(args, settings):
    from visionbench.harness.mock import format_truth
    from visionbench.tasks.base import load_all
    from visionbench.verifiers import answers
    bad = 0
    for name, fam in sorted(load_all().items()):
        n = 0
        for d in fam.difficulties:
            for s in range(args.n):
                inst = fam.generate(s, d, render_seed=s % 2)
                if inst.answer.kind == "rubric":
                    n += 1
                    continue
                out = answers.check(inst.answer, format_truth(inst), fam, inst)
                if out[0] != "correct":
                    bad += 1
                    print(f"FAIL {name} d{d} s{s}: {out[0]} {out[4]}")
                n += 1
        print(f"{name:20s} {n} items: oracle round-trip ok")
    print("selftest:", "PASS" if bad == 0 else f"{bad} FAILURES")
    sys.exit(1 if bad else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="visionbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", default=None, help="path to settings.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("families").set_defaults(fn=cmd_families)

    p = sub.add_parser("preview")
    p.add_argument("family")
    p.add_argument("-d", "--difficulty", type=int, default=3)
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("-r", "--render", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "previews"))
    p.add_argument("--facts", action="store_true")
    p.set_defaults(fn=cmd_preview)

    p = sub.add_parser("build")
    p.add_argument("name")
    p.add_argument("--families", default="all")
    p.add_argument("--difficulties", default="1-5")
    p.add_argument("--per-cell", type=int, default=10)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--rerenders", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("import")
    p.add_argument("source")
    p.add_argument("--name", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="subsampling seed")
    p.add_argument("--split", default=None)
    p.set_defaults(fn=cmd_import)

    for name, fn in (("run", cmd_run), ("estimate", cmd_estimate)):
        p = sub.add_parser(name)
        p.add_argument("dataset")
        p.add_argument("--models", default=None, help="comma-separated; 'mock:oracle', 'mock:noisy-0.7' work offline")
        p.add_argument("--mode", choices=["direct", "agentic"], default=None)
        p.add_argument("--conditions", default=None if name == "run" else "vision,blind",
                       help="vision,blind,caption")
        p.add_argument("--samples", type=int, default=None if name == "run" else 3)
        if name == "run":
            p.add_argument("--families", default=None)
            p.add_argument("--limit", type=int, default=None)
            p.add_argument("--per-family", type=int, default=None, help="stratified subset: N items per family")
            p.add_argument("--run-name", default=None)
            p.add_argument("--effort", default=None)
            p.add_argument("--max-tokens", type=int, default=None)
            p.add_argument("--max-turns", type=int, default=None)
            p.add_argument("--concurrency", type=int, default=None)
            p.add_argument("--code-execution", action="store_true")
            p.add_argument("--process-judge", action="store_true")
            p.add_argument("--critique", action="store_true", help="also run the reasoning critique (sample 0 per item)")
            p.add_argument("--budget", type=float, default=None, help="OpenRouter USD cap for this invocation")
        p.set_defaults(fn=fn)

    p = sub.add_parser("verify")
    p.add_argument("run")
    p.add_argument("--reset", action="store_true", help="re-verify everything")
    p.add_argument("--process-judge", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("critique", help="grade how each agent reasoned (key facts, validity, strategy)")
    p.add_argument("run")
    p.add_argument("--all-samples", action="store_true", help="critique every sample, not just sample 0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--models", default=None)
    p.set_defaults(fn=cmd_critique)

    p = sub.add_parser("report")
    p.add_argument("run")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("judge-audit")
    p.add_argument("run")
    p.add_argument("--n", type=int, default=150)
    p.set_defaults(fn=cmd_judge_audit)

    p = sub.add_parser("human-form", help="HTML answer form for a human baseline")
    p.add_argument("dataset")
    p.add_argument("--name", required=True, help="annotator id")
    p.add_argument("--n", type=int, default=75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_human_form)

    p = sub.add_parser("human-import", help="add a human's answers to a run as subject human:<name>")
    p.add_argument("answers")
    p.add_argument("run")
    p.add_argument("--dataset", default=None, help="needed if the run directory is new")
    p.set_defaults(fn=cmd_human_import)

    p = sub.add_parser("critique-audit", help="test-retest agreement of the reasoning critic")
    p.add_argument("run")
    p.add_argument("--n", type=int, default=40)
    p.set_defaults(fn=cmd_critique_audit)

    p = sub.add_parser("selftest")
    p.add_argument("--n", type=int, default=6)
    p.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    settings = load_settings(args.settings)
    args.fn(args, settings)


if __name__ == "__main__":
    main()
