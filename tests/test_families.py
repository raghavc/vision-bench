import numpy as np
import pytest

from visionbench.harness.mock import format_truth
from visionbench.tasks.base import load_all
from visionbench.verifiers import answers

FAMILIES = sorted(load_all())


@pytest.mark.parametrize("name", FAMILIES)
def test_deterministic(name):
    fam = load_all()[name]
    a = fam.generate(3, fam.difficulties[-1])
    b = fam.generate(3, fam.difficulties[-1])
    assert a.uid == b.uid and a.prompt == b.prompt and a.answer.value == b.answer.value
    assert np.array_equal(np.asarray(a.images[0]), np.asarray(b.images[0]))


@pytest.mark.parametrize("name", FAMILIES)
def test_rerender_preserves_answer(name):
    fam = load_all()[name]
    for d in fam.difficulties:
        base = fam.generate(5, d, render_seed=0)
        alt = fam.generate(5, d, render_seed=2)
        assert base.answer.value == alt.answer.value
        assert base.cluster == alt.cluster and base.uid != alt.uid


@pytest.mark.parametrize("name", FAMILIES)
def test_oracle_round_trip(name):
    fam = load_all()[name]
    for d in fam.difficulties:
        for seed in range(3):
            inst = fam.generate(seed, d)
            if inst.answer.kind == "rubric":
                assert inst.rubric and any(r.critical for r in inst.rubric)
                continue
            outcome, *_ = answers.check(inst.answer, format_truth(inst), fam, inst)
            assert outcome == "correct", (name, d, seed)


def test_seeds_give_distinct_problems():
    fam = load_all()["gear_train"]
    vals = {str(fam.generate(s, 4).answer.value) for s in range(15)}
    assert len(vals) > 5
