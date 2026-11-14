"""Evaluation at deployment operating points, not AUC.

Primary metric: recall at fixed false-positive rate (0.1%, 1%), threshold chosen on val, reported on test.
Every metric is also sliced by tag (jailbreak, paraphrase, multi_turn, lang:*) so shift shows up as a row,
and by source so you can see which datasets a probe silently fails on.
Bootstrap CIs so a 2-point difference between layers isn't mistaken for signal.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

FPR_POINTS = (0.001, 0.01, 0.05)


def threshold_at_fpr(neg_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold such that P(neg >= t) <= fpr. With few negatives this is conservative."""
    if len(neg_scores) == 0:
        return float("inf")
    k = int(np.floor(fpr * len(neg_scores)))
    s = np.sort(neg_scores)[::-1]
    return float(s[k]) + 1e-6 if k < len(s) else float(s[-1]) - 1e-6


def recall_at(scores, y, t) -> float:
    pos = scores[y == 1]
    return float((pos >= t).mean()) if len(pos) else float("nan")


def fpr_at(scores, y, t) -> float:
    neg = scores[y == 0]
    return float((neg >= t).mean()) if len(neg) else float("nan")


def precision_at(scores, y, t) -> float:
    flagged = scores >= t
    return float(y[flagged].mean()) if flagged.any() else float("nan")


def auroc(scores, y) -> float:
    from sklearn.metrics import roc_auc_score
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def bootstrap(fn, scores, y, n=200, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(fn(scores[idx], y[idx]))
    vals = np.array(vals)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


@dataclass
class Report:
    n_pos: int
    n_neg: int
    auroc: float
    thresholds: dict[str, float]            # "fpr=0.001" -> t (chosen on val)
    recall: dict[str, float]                # on test, at each threshold
    recall_ci: dict[str, tuple[float, float]]
    realized_fpr: dict[str, float]          # what the val threshold actually gives on test
    precision: dict[str, float]
    slices: dict[str, dict[str, float]] = field(default_factory=dict)  # tag -> {"n":, "recall@fpr=..":}

    def row(self, key="fpr=0.01") -> str:
        lo, hi = self.recall_ci[key]
        return f"auroc={self.auroc:.3f} R@{key}={self.recall[key]:.3f} [{lo:.3f},{hi:.3f}] realizedFPR={self.realized_fpr[key]:.4f} P={self.precision[key]:.3f}"


def evaluate(val_scores, val_y, test_scores, test_y, test_tags: list[list[str]] | None = None, fprs=FPR_POINTS) -> Report:
    thresholds = {f"fpr={f}": threshold_at_fpr(val_scores[val_y == 0], f) for f in fprs}
    recall, ci, rf, prec = {}, {}, {}, {}
    for k, t in thresholds.items():
        recall[k] = recall_at(test_scores, test_y, t)
        ci[k] = bootstrap(lambda s, y: recall_at(s, y, t), test_scores, test_y)
        rf[k] = fpr_at(test_scores, test_y, t)
        prec[k] = precision_at(test_scores, test_y, t)
    slices = {}
    if test_tags is not None:
        all_tags = sorted({t for ts in test_tags for t in ts})
        for tag in all_tags:
            m = np.array([tag in ts for ts in test_tags])
            if m.sum() == 0:
                continue
            row = {"n": int(m.sum())}
            for k, t in thresholds.items():
                row[f"recall@{k}"] = recall_at(test_scores[m], test_y[m], t)
                row[f"fpr@{k}"] = fpr_at(test_scores[m], test_y[m], t)
            slices[tag] = row
    return Report(int((test_y == 1).sum()), int((test_y == 0).sum()), auroc(test_scores, test_y), thresholds, recall, ci, rf, prec, slices)
