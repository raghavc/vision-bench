import math

from visionbench.analysis import stats


def row(uid, cluster, score, sample=0, model="m"):
    return stats.Row(model=model, mode="direct", condition="vision", uid=uid, cluster=cluster, family="f",
                     difficulty=1, sample=sample, score=score, partial=None, outcome="correct" if score else "incorrect",
                     confidence=0.5, cost=0, output_tokens=0, latency=0, tool_calls=0, error_category=None)


def test_pass_k_estimators():
    assert stats.pass_at_k(5, 0, 1) == 0
    assert math.isclose(stats.pass_at_k(5, 2, 1), 0.4)
    assert math.isclose(stats.pass_hat_k(5, 2, 2), 1 / 10)
    assert stats.pass_hat_k(3, 3, 3) == 1


def test_clustered_se_larger_than_naive_when_clusters_correlate():
    rows = []
    for c in range(20):
        v = float(c % 2)
        for k in range(4):  # 4 identical re-renders per cluster
            rows.append(row(f"{c}-{k}", f"c{c}", v))
    im = stats.item_means(rows)
    mean, se, n, C = stats.clustered_mean_se(im)
    naive = math.sqrt(mean * (1 - mean) / n)
    assert C == 20 and n == 80 and se > 1.5 * naive


def test_paired_compare_detects_difference():
    a = [row(f"i{i}", f"i{i}", 1.0, model="a") for i in range(60)]
    b = [row(f"i{i}", f"i{i}", float(i % 3 == 0), model="b") for i in range(60)]
    c = stats.paired_compare(a, b)
    assert c["diff"] > 0.5 and c["p_value"] < 0.01 and c["ci95"][0] > 0


def test_holm_monotone():
    adj = stats.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == 0.03 and adj["z"] == 0.06 and adj["y"] == 0.06


def test_rasch_orders_subjects():
    resp = {}
    for i in range(30):
        resp[("strong", f"i{i}")] = [1.0, 1.0, float(i % 5 != 0)]
        resp[("weak", f"i{i}")] = [float(i % 2 == 0), 0.0, 0.0]
    theta, b = stats.rasch(resp)
    assert theta["strong"] > theta["weak"]
