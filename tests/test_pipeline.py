import numpy as np
from probelab import probes as P, evals


def _data(n=400, d=32, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 0] += 3 * y
    return X, y


def test_probes_separate():
    X, y = _data()
    for kind in ["massmean", "logistic", "mlp"]:
        pr = P.REGISTRY[kind]().fit(X[:300], y[:300])
        assert evals.auroc(pr.score(X[300:]), y[300:]) > 0.95, kind


def test_threshold_at_fpr_is_conservative():
    neg = np.random.default_rng(1).normal(size=10000)
    t = evals.threshold_at_fpr(neg, 0.01)
    assert (neg >= t).mean() <= 0.01


def test_roundtrip():
    X, y = _data()
    for kind in ["massmean", "logistic"]:
        pr = P.REGISTRY[kind]().fit(X, y)
        pr2 = P.REGISTRY[kind].from_state(pr.state())
        assert np.allclose(pr.score(X), pr2.score(X), atol=1e-4)


def test_evaluate_slices():
    X, y = _data()
    pr = P.Logistic().fit(X[:200], y[:200])
    tags = [["a"] if i % 2 else ["b"] for i in range(200)]
    rep = evals.evaluate(pr.score(X[200:300]), y[200:300], pr.score(X[300:]), y[300:], tags[:100])
    assert set(rep.slices) == {"a", "b"}
