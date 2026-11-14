from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import typer
from rich import print as rprint
from rich.table import Table

from . import evals, probes as P, select as S
from .extract import Cache, extract as _extract, model_tag
from .schema import read_jsonl, write_jsonl

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def extract(data: Path, model: str, layers: str = "-1", out: Path = Path("cache"), batch_size: int = 8, max_len: int = 512, tag: str = ""):
    """Cache pooled residual activations for every example in DATA. Layers: comma list, negatives ok, 'all'."""
    exs = read_jsonl(data)
    tag = tag or data.stem
    if layers == "all":
        from transformers import AutoConfig
        n = AutoConfig.from_pretrained(model).num_hidden_layers
        ls = list(range(1, n + 1))
    else:
        ls = [int(x) for x in layers.split(",")]
    p = _extract(exs, model, ls, out, batch_size=batch_size, max_len=max_len, dataset_tag=tag)
    rprint(f"[green]cached[/] {len(exs)} examples x {len(ls)} layers -> {p}")


def _split(exs, concept):
    by = {"train": [], "val": [], "test": []}
    for ex in exs:
        if concept in ex.labels:
            by[ex.split].append(ex)
    return by


def _xy(cache, exs, layer, pooling, concept):
    X = cache.feats(layer, pooling, [e.id for e in exs])
    y = np.array([e.labels[concept] for e in exs])
    return X, y


@app.command()
def sweep(data: Path, model: str, concept: str, cache_dir: Path = Path("cache"), kinds: str = "massmean,logistic", poolings: str = "mean,last", fpr_key: str = "fpr=0.01", tag: str = "", out: Path = None):
    """Train every (layer, pooling, kind) probe from the cache; rank by recall at fixed FPR. The plot that tells you if tier-0 is viable."""
    exs = read_jsonl(data)
    cache = Cache(cache_dir / model_tag(model) / (tag or data.stem))
    sp = _split(exs, concept)
    if not sp["val"]:
        rprint("[yellow]no val split; using train for thresholds (optimistic)[/]"); sp["val"] = sp["train"]
    rows = []
    for layer in cache.layers:
        for pooling in poolings.split(","):
            Xtr, ytr = _xy(cache, sp["train"], layer, pooling, concept)
            Xva, yva = _xy(cache, sp["val"], layer, pooling, concept)
            Xte, yte = _xy(cache, sp["test"], layer, pooling, concept)
            for kind in kinds.split(","):
                pr = P.REGISTRY[kind]().fit(Xtr, ytr)
                rep = evals.evaluate(pr.score(Xva), yva, pr.score(Xte), yte, [e.tags for e in sp["test"]])
                rows.append((layer, pooling, kind, rep))
    rows.sort(key=lambda r: (-r[3].recall[fpr_key], r[3].realized_fpr[fpr_key]))
    t = Table(title=f"{concept} @ {model}  (test n+={rows[0][3].n_pos} n-={rows[0][3].n_neg})")
    for c in ["layer", "pool", "kind", "auroc", f"R@{fpr_key}", "95% CI", "realized FPR", "prec"]:
        t.add_column(c)
    for layer, pooling, kind, rep in rows:
        lo, hi = rep.recall_ci[fpr_key]
        t.add_row(str(layer), pooling, kind, f"{rep.auroc:.3f}", f"{rep.recall[fpr_key]:.3f}", f"[{lo:.2f},{hi:.2f}]", f"{rep.realized_fpr[fpr_key]:.4f}", f"{rep.precision[fpr_key]:.3f}")
    rprint(t)
    best = rows[0]
    if best[3].slices:
        st = Table(title=f"best probe L{best[0]}_{best[1]}_{best[2]} by slice")
        st.add_column("slice"); st.add_column("n"); st.add_column(f"recall@{fpr_key}"); st.add_column(f"fpr@{fpr_key}")
        for tag_, r in sorted(best[3].slices.items()):
            st.add_row(tag_, str(r["n"]), f"{r[f'recall@{fpr_key}']:.3f}", f"{r[f'fpr@{fpr_key}']:.4f}")
        rprint(st)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([{"layer": l, "pooling": p, "kind": k, **rep.__dict__} for l, p, k, rep in rows], default=list))
        rprint(f"wrote {out}")


@app.command()
def train(data: Path, model: str, concept: str, layer: int, pooling: str = "mean", kind: str = "logistic", fpr: float = 0.01, cache_dir: Path = Path("cache"), out: Path = Path("probes"), tag: str = ""):
    """Train one probe on train split, pick threshold at FPR on val, save it."""
    exs = read_jsonl(data)
    cache = Cache(cache_dir / model_tag(model) / (tag or data.stem))
    sp = _split(exs, concept)
    val = sp["val"] or sp["train"]
    Xtr, ytr = _xy(cache, sp["train"], layer, pooling, concept)
    Xva, yva = _xy(cache, val, layer, pooling, concept)
    pr = P.REGISTRY[kind]().fit(Xtr, ytr)
    sv = pr.score(Xva)
    t = evals.threshold_at_fpr(sv[yva == 0], fpr)
    spec = P.ProbeSpec(concept, layer, pooling, kind, model)
    path = P.save_probe(spec, pr, t, out)
    rprint(f"[green]saved[/] {path}  threshold@fpr={fpr}: {t:.3f}  val recall={evals.recall_at(sv, yva, t):.3f}")


@app.command()
def score(probe: Path, data: Path, cache_dir: Path = Path("cache"), tag: str = "", out: Path = None):
    """Score every example in DATA with a saved probe (reads the cache; no model forward)."""
    spec, pr, t = P.load_probe(probe)
    exs = read_jsonl(data)
    cache = Cache(cache_dir / model_tag(spec.model) / (tag or data.stem))
    s = pr.score(cache.feats(spec.layer, spec.pooling, [e.id for e in exs]))
    flagged = int((s >= t).sum())
    rprint(f"{spec.name()}: flagged {flagged}/{len(exs)} at t={t:.3f}")
    if out:
        with open(out, "w") as f:
            for e, sc in zip(exs, s):
                f.write(json.dumps({"id": e.id, "score": float(sc), "flag": bool(sc >= t)}) + "\n")


@app.command()
def select(probe: Path, pool: Path, k: int = 50, strategy: str = "uncertain", cache_dir: Path = Path("cache"), tag: str = "", out: Path = Path("data/to_label.jsonl")):
    """Pick K unlabeled examples from POOL to label next. Label them, append to train, re-run sweep/train."""
    spec, pr, t = P.load_probe(probe)
    exs = read_jsonl(pool)
    cache = Cache(cache_dir / model_tag(spec.model) / (tag or pool.stem))
    X = cache.feats(spec.layer, spec.pooling, [e.id for e in exs])
    s = pr.score(X)
    if strategy == "uncertain":
        idx = S.uncertain(s, t, k)
    elif strategy == "positive":
        idx = S.top_positive(s, k)
    elif strategy == "diverse":
        idx = S.diverse(X, S.uncertain(s, t, k * 5), k)
    else:
        raise typer.BadParameter(strategy)
    picked = [exs[i] for i in idx]
    for e, i in zip(picked, idx):
        e.tags = list(set(e.tags + [f"selected:{strategy}", f"score:{s[i]:.2f}"]))
    write_jsonl(out, picked)
    rprint(f"[green]wrote[/] {len(picked)} examples to {out} ({strategy})")


if __name__ == "__main__":
    app()
