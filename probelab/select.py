"""Active learning: pick what to label next so the probe improves fastest.

Strategies over an unlabeled pool (scored by the current probe):
  uncertain  : closest to threshold          -> sharpens the boundary
  positive   : highest scores                -> precision audit; catches FP families
  disagree   : tier0 vs tier1 disagreement   -> finds where the linear probe is wrong
  diverse    : k-means on activations of the uncertain band -> covers new modes, not one cluster
Output is a JSONL of examples to label; labeled output re-enters the train split. Loop.
"""
from __future__ import annotations
import numpy as np


def uncertain(scores: np.ndarray, threshold: float, k: int) -> np.ndarray:
    return np.argsort(np.abs(scores - threshold))[:k]


def top_positive(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores)[:k]


def disagree(s0: np.ndarray, s1: np.ndarray, t0: float, t1: float, k: int) -> np.ndarray:
    d = (s0 >= t0) != (s1 >= t1)
    idx = np.where(d)[0]
    return idx[np.argsort(-np.abs(s0[idx] - t0))[:k]]


def diverse(X: np.ndarray, cand: np.ndarray, k: int, seed=0) -> np.ndarray:
    from sklearn.cluster import KMeans
    if len(cand) <= k:
        return cand
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X[cand])
    picks = []
    for c in range(k):
        members = cand[km.labels_ == c]
        if len(members):
            d = np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1)
            picks.append(members[np.argmin(d)])
    return np.array(picks)
