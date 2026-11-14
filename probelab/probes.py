"""Probe zoo. Every probe: fit(X, y) -> self ; score(X) -> float32 [N] (higher = more positive).

Tier 0 (always-on, microseconds): mass-mean direction, logistic regression.
Tier 1 (flagged samples only): small MLP.  SAE-feature probes plug in here later.
Probes are tiny; the cache is the expensive thing. Retraining every probe on new labels is seconds.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class Probe:
    kind = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Probe":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_state(cls, st: dict) -> "Probe":
        raise NotImplementedError


class MassMean(Probe):
    """Direction = mean(pos) - mean(neg), whitened by pooled covariance (LDA-lite). Very robust with few labels."""
    kind = "massmean"

    def __init__(self, shrink: float = 0.1):
        self.shrink = shrink
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        mu_p, mu_n = X[y == 1].mean(0), X[y == 0].mean(0)
        Xc = np.concatenate([X[y == 1] - mu_p, X[y == 0] - mu_n])
        cov = Xc.T @ Xc / max(len(Xc) - 1, 1)
        cov = (1 - self.shrink) * cov + self.shrink * np.eye(X.shape[1]) * np.trace(cov) / X.shape[1]
        self.w = np.linalg.solve(cov, mu_p - mu_n).astype(np.float32)
        self.b = -float(((mu_p + mu_n) / 2) @ self.w)
        return self

    def score(self, X):
        return (X @ self.w + self.b).astype(np.float32)

    def state(self):
        return {"kind": self.kind, "w": self.w.tolist(), "b": self.b}

    @classmethod
    def from_state(cls, st):
        p = cls(); p.w = np.array(st["w"], dtype=np.float32); p.b = st["b"]; return p


class Logistic(Probe):
    kind = "logistic"

    def __init__(self, C: float = 0.5):
        self.C = C
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")

    def fit(self, X, y):
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def score(self, X):
        return self.clf.decision_function(self.scaler.transform(X)).astype(np.float32)

    def state(self):
        return {
            "kind": self.kind, "C": self.C,
            "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
            "coef": self.clf.coef_[0].tolist(), "intercept": float(self.clf.intercept_[0]),
        }

    @classmethod
    def from_state(cls, st):
        p = cls(C=st["C"])
        p.scaler.mean_ = np.array(st["mean"]); p.scaler.scale_ = np.array(st["scale"])
        p.scaler.var_ = p.scaler.scale_ ** 2; p.scaler.n_features_in_ = len(st["mean"])
        p.clf.coef_ = np.array([st["coef"]]); p.clf.intercept_ = np.array([st["intercept"]]); p.clf.classes_ = np.array([0, 1])
        return p


class MLP(Probe):
    kind = "mlp"

    def __init__(self, hidden: int = 128, alpha: float = 1e-3):
        self.hidden, self.alpha = hidden, alpha
        self.scaler = StandardScaler()
        self.clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=1000, early_stopping=False, random_state=0)

    def fit(self, X, y):
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def score(self, X):
        p = self.clf.predict_proba(self.scaler.transform(X))[:, 1]
        return np.log(p + 1e-9) - np.log(1 - p + 1e-9)

    def state(self):  # sklearn MLP is not json-trivial; pickle via joblib in save()
        return {"kind": self.kind, "hidden": self.hidden, "alpha": self.alpha, "_pickle": True}


REGISTRY = {"massmean": MassMean, "logistic": Logistic, "mlp": MLP}


@dataclass
class ProbeSpec:
    concept: str
    layer: int
    pooling: str
    kind: str
    model: str

    def name(self) -> str:
        return f"{self.concept}__L{self.layer}_{self.pooling}_{self.kind}"


def save_probe(spec: ProbeSpec, probe: Probe, threshold: float, out_dir: str | Path) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    st = probe.state()
    path = out / f"{spec.name()}.json"
    if st.get("_pickle"):
        import joblib
        joblib.dump(probe, out / f"{spec.name()}.joblib")
    path.write_text(json.dumps({"spec": asdict(spec), "threshold": threshold, "state": st}))
    return path


def load_probe(path: str | Path) -> tuple[ProbeSpec, Probe, float]:
    d = json.loads(Path(path).read_text())
    spec = ProbeSpec(**d["spec"])
    if d["state"].get("_pickle"):
        import joblib
        probe = joblib.load(Path(path).with_suffix(".joblib"))
    else:
        probe = REGISTRY[d["state"]["kind"]].from_state(d["state"])
    return spec, probe, d["threshold"]
