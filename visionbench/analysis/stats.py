"""Statistics following "Adding Error Bars to Evals" (Miller, 2024) and newer practice.

  * Average the k samples within each item first; the item is the unit.
  * Standard errors are clustered on the latent problem (re-renders of one
    problem share a cluster). Bootstrap CIs resample clusters.
  * Model comparisons are paired on shared items: mean difference, clustered
    SE, bootstrap CI, a sign-flip permutation p-value, and Holm correction
    across pairs.
  * pass@k and pass^k use the unbiased combinatorial estimators (Chen 2021;
    tau-bench) instead of naive powers.
  * Calibration (ECE, Brier) uses the model's stated confidence.
  * A Rasch (1PL) IRT fit gives item difficulties and flags suspect items.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

RNG = np.random.default_rng(20241101)
N_BOOT = 2000


@dataclass
class Row:
    model: str
    mode: str
    condition: str
    uid: str
    cluster: str
    family: str
    difficulty: int
    sample: int
    score: float
    partial: float | None
    outcome: str
    confidence: float | None
    cost: float
    output_tokens: int
    latency: float
    tool_calls: int
    error_category: str | None


# ---------------------------------------------------------------- basics
def item_means(rows: Iterable[Row], value=lambda r: r.score) -> dict[str, tuple[float, str, int]]:
    """uid -> (mean over samples, cluster, n_samples)."""
    acc: dict[str, list[float]] = defaultdict(list)
    cl: dict[str, str] = {}
    for r in rows:
        acc[r.uid].append(value(r))
        cl[r.uid] = r.cluster
    return {u: (float(np.mean(v)), cl[u], len(v)) for u, v in acc.items()}


def clustered_mean_se(items: dict[str, tuple[float, str, int]]) -> tuple[float, float, int, int]:
    """Mean of item scores and cluster-robust SE. Returns (mean, se, n_items, n_clusters)."""
    if not items:
        return float("nan"), float("nan"), 0, 0
    x = np.array([v[0] for v in items.values()])
    clusters = [v[1] for v in items.values()]
    n = len(x)
    mean = float(x.mean())
    groups: dict[str, list[float]] = defaultdict(list)
    for xi, c in zip(x, clusters):
        groups[c].append(xi - mean)
    C = len(groups)
    if n < 2:
        return mean, float("nan"), n, C
    s = sum(sum(g) ** 2 for g in groups.values())
    # small-sample correction C/(C-1)
    var = s / n ** 2 * (C / (C - 1) if C > 1 else 1.0)
    return mean, math.sqrt(var), n, C


def cluster_bootstrap_ci(items: dict[str, tuple[float, str, int]], alpha: float = 0.05, n_boot: int = N_BOOT):
    if len(items) < 2:
        return (float("nan"), float("nan"))
    by_c: dict[str, list[float]] = defaultdict(list)
    for v in items.values():
        by_c[v[1]].append(v[0])
    sums = np.array([sum(v) for v in by_c.values()])
    cnts = np.array([len(v) for v in by_c.values()])
    C = len(sums)
    idx = RNG.integers(0, C, size=(n_boot, C))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def summarize(rows: list[Row], value=lambda r: r.score) -> dict[str, Any]:
    im = item_means(rows, value)
    mean, se, n, C = clustered_mean_se(im)
    lo, hi = cluster_bootstrap_ci(im)
    return {"mean": mean, "se": se, "ci95": [lo, hi], "n_items": n, "n_clusters": C,
            "n_attempts": len(rows)}


# ---------------------------------------------------------------- pass@k
def _comb(n: int, k: int) -> float:
    return math.comb(n, k) if 0 <= k <= n else 0


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k draws without replacement is correct)."""
    if n < k:
        return float("nan")
    return 1.0 - _comb(n - c, k) / _comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(all k draws without replacement are correct)  (tau-bench pass^k)."""
    if n < k:
        return float("nan")
    return _comb(c, k) / _comb(n, k)


def pass_k_table(rows: list[Row], ks: Iterable[int] = (1, 2, 3, 5)) -> dict[str, Any]:
    per_item: dict[str, list[float]] = defaultdict(list)
    cl = {}
    for r in rows:
        per_item[r.uid].append(r.score)
        cl[r.uid] = r.cluster
    out = {}
    for k in ks:
        at = {u: (pass_at_k(len(v), int(sum(v)), k), cl[u], 1) for u, v in per_item.items() if len(v) >= k}
        hat = {u: (pass_hat_k(len(v), int(sum(v)), k), cl[u], 1) for u, v in per_item.items() if len(v) >= k}
        if at:
            out[f"pass@{k}"] = clustered_mean_se(at)[0]
            out[f"pass^{k}"] = clustered_mean_se(hat)[0]
    return out


# ---------------------------------------------------------- comparisons
def paired_compare(a: list[Row], b: list[Row], n_perm: int = 5000) -> dict[str, Any]:
    ia, ib = item_means(a), item_means(b)
    shared = sorted(set(ia) & set(ib))
    if len(shared) < 2:
        return {"n_items": len(shared)}
    d = {u: (ia[u][0] - ib[u][0], ia[u][1], 1) for u in shared}
    mean, se, n, C = clustered_mean_se(d)
    lo, hi = cluster_bootstrap_ci(d)
    # sign-flip permutation test at the cluster level
    by_c: dict[str, float] = defaultdict(float)
    for v in d.values():
        by_c[v[1]] += v[0]
    sums = np.array(list(by_c.values()))
    obs = abs(sums.sum())
    flips = RNG.choice([-1.0, 1.0], size=(n_perm, len(sums)))
    p = float((np.abs((flips * sums).sum(1)) >= obs - 1e-12).mean())
    xa = np.array([ia[u][0] for u in shared])
    xb = np.array([ib[u][0] for u in shared])
    corr = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 0 and xb.std() > 0 else float("nan")
    return {"diff": mean, "se": se, "ci95": [lo, hi], "p_value": p, "n_items": n, "n_clusters": C,
            "item_correlation": corr, "mde_80pct_power": 2.8 * se if se == se else float("nan")}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def rank_spread(model_rows: dict[str, list[Row]], n_boot: int = 1000) -> dict[str, tuple[int, int]]:
    """Bootstrap rank intervals (clusters resampled jointly across models)."""
    ims_all = {m: item_means(r) for m, r in model_rows.items()}
    # subjects with partial coverage would shrink the shared item set to almost nothing; rank the well-covered ones
    most = max((len(v) for v in ims_all.values()), default=0)
    ims = {m: v for m, v in ims_all.items() if len(v) >= 0.8 * most}
    models = list(ims)
    shared = sorted(set.intersection(*[set(v) for v in ims.values()])) if ims else []
    if not shared or len(models) < 2:
        return {m: (1, 1) for m in models}
    clusters = sorted({ims[models[0]][u][1] for u in shared})
    cidx = {c: i for i, c in enumerate(clusters)}
    mat = np.zeros((len(models), len(clusters)))
    cnt = np.zeros(len(clusters))
    for u in shared:
        j = cidx[ims[models[0]][u][1]]
        cnt[j] += 1
        for i, m in enumerate(models):
            mat[i, j] += ims[m][u][0]
    ranks = []
    for _ in range(n_boot):
        s = RNG.integers(0, len(clusters), len(clusters))
        sc = mat[:, s].sum(1) / cnt[s].sum()
        ranks.append((-sc).argsort().argsort() + 1)
    ranks = np.array(ranks)
    return {m: (int(np.quantile(ranks[:, i], 0.025)), int(np.quantile(ranks[:, i], 0.975)))
            for i, m in enumerate(models)}


# ------------------------------------------------------------ calibration
def calibration(rows: list[Row], bins: int = 10) -> dict[str, Any]:
    pts = [(r.confidence, r.score) for r in rows if r.confidence is not None and r.outcome != "error"]
    if len(pts) < 10:
        return {"n": len(pts)}
    c = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for i in range(bins):
        m = (c >= edges[i]) & ((c < edges[i + 1]) if i < bins - 1 else (c <= edges[i + 1]))
        if m.sum():
            gap = abs(c[m].mean() - y[m].mean())
            ece += m.sum() / len(c) * gap
            table.append({"bin": [float(edges[i]), float(edges[i + 1])], "n": int(m.sum()),
                          "conf": float(c[m].mean()), "acc": float(y[m].mean())})
    brier = float(((c - y) ** 2).mean())
    # overconfidence: mean confidence minus accuracy
    return {"n": len(pts), "ece": float(ece), "brier": brier, "mean_conf": float(c.mean()),
            "accuracy": float(y.mean()), "overconfidence": float(c.mean() - y.mean()), "bins": table}


# --------------------------------------------------------------- IRT (1PL)
def rasch(responses: dict[tuple[str, str], list[float]], iters: int = 500, lr: float = 0.05, l2: float = 0.01):
    """Joint MLE Rasch model on (subject, item) -> list of 0/1 outcomes.

    Returns (theta per subject, b per item). Subjects are (model, mode) combos.
    L2 keeps estimates finite for perfect/zero items.
    """
    subjects = sorted({s for s, _ in responses})
    items = sorted({i for _, i in responses})
    si = {s: k for k, s in enumerate(subjects)}
    ii = {i: k for k, i in enumerate(items)}
    S, I = [], []
    Y, N = [], []
    for (s, i), ys in responses.items():
        S.append(si[s]); I.append(ii[i]); Y.append(sum(ys)); N.append(len(ys))
    S, I, Y, N = map(np.array, (S, I, Y, N))
    theta = np.zeros(len(subjects))
    b = np.zeros(len(items))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(theta[S] - b[I])))
        g = Y - N * p
        gt = np.bincount(S, g, len(subjects)) - l2 * theta
        gb = -np.bincount(I, g, len(items)) - l2 * b
        theta += lr * gt / max(1, len(items)) * 10
        b += lr * gb / max(1, len(subjects))
        theta -= theta.mean()  # identifiability
    return dict(zip(subjects, theta.tolist())), dict(zip(items, b.tolist()))


def suspect_items(responses, theta, b, top_k: int = 20) -> list[dict[str, Any]]:
    """Items failed by every sample of the strongest subject while weaker subjects pass: possible bad keys."""
    if not theta:
        return []
    best = max(theta, key=theta.get)
    out = []
    for (s, i), ys in responses.items():
        if s != best or sum(ys) > 0:
            continue
        others = [sum(v) / len(v) for (s2, i2), v in responses.items() if i2 == i and s2 != best]
        if others and max(others) >= 0.5:
            out.append({"uid": i, "best_subject": best, "best_acc": 0.0, "max_other_acc": max(others),
                        "difficulty_b": b.get(i)})
    return sorted(out, key=lambda d: -d["max_other_acc"])[:top_k]
