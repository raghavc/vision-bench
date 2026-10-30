from visionbench.core.types import AnswerSpec as A
from visionbench.verifiers.answers import check, extract_answer


def test_extract_last_tag_and_confidence():
    ans, conf, how = extract_answer("x <answer>1</answer> then <answer> 42 </answer><confidence>75%</confidence>")
    assert (ans, conf, how) == ("42", 0.75, "tag")
    assert extract_answer("no tags at all")[2] == "none"


def test_numbers():
    assert check(A("number", 37.5, "", tolerance=1e-6), "75/2")[0] == "correct"
    assert check(A("number", 41.37, "", tolerance=0.005), "$41.37")[0] == "correct"
    assert check(A("integer", 12, ""), "12.5")[0] == "incorrect"
    assert check(A("integer", 12, ""), "either 12 or 13")[0] == "unparseable"


def test_sets_and_mappings_partial_credit():
    out = check(A("set", ["A", "C"], ""), "C, A, D")
    assert out[0] == "incorrect" and abs(out[2] - 0.8) < 1e-9
    out = check(A("mapping", {"A": "3", "B": "1"}, ""), "A=3, B=2")
    assert out[0] == "incorrect" and out[2] == 0.5


def test_choice_requires_single_letter():
    assert check(A("choice", "B", "", choices=list("ABCD")), "(B)")[0] == "correct"
    assert check(A("choice", "B", "", choices=list("ABCD")), "A or B")[0] == "unparseable"


def test_grid_partial():
    out = check(A("grid", [[1, 0], [0, 2]], ""), "1 0\n0 3")
    assert out[0] == "incorrect" and out[2] == 0.75
