"""Extract residual-stream activations once; everything downstream reads the cache.

Cache layout (one dir per model+dataset):
  cache/<model_tag>/<dataset_tag>/
    manifest.json      ids, layers, poolings, hidden size, model, dataset hash
    L<layer>_<pool>.npy  float16 [N, d]   pool in {mean, last}
Pooling over tokens is the cheap tier. Per-token caches are opt-in (big).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import Example

POOLINGS = ("mean", "last")


def model_tag(name: str) -> str:
    return name.replace("/", "__")


def dataset_hash(examples: list[Example]) -> str:
    h = hashlib.sha1()
    for ex in examples:
        h.update(ex.id.encode())
    return h.hexdigest()[:10]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str | None = None, dtype=None):
    device = device or pick_device()
    dtype = dtype or (torch.float16 if device != "cpu" else torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device).eval()
    return tok, model, device


@torch.no_grad()
def extract(
    examples: list[Example],
    model_name: str,
    layers: list[int],
    out_dir: str | Path,
    batch_size: int = 8,
    max_len: int = 512,
    dataset_tag: str = "ds",
) -> Path:
    tok, model, device = load_model(model_name)
    n_layers = model.config.num_hidden_layers
    layers = [l if l >= 0 else n_layers + l for l in layers]
    d = model.config.hidden_size
    N = len(examples)
    out = Path(out_dir) / model_tag(model_name) / dataset_tag
    out.mkdir(parents=True, exist_ok=True)

    bufs = {(l, p): np.zeros((N, d), dtype=np.float16) for l in layers for p in POOLINGS}
    texts = [ex.render(tok) for ex in examples]

    for s in range(0, N, batch_size):
        batch = texts[s : s + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        res = model(**enc, output_hidden_states=True)
        hs = res.hidden_states  # tuple len n_layers+1 ; hs[i] = output of block i (hs[0]=embeddings)
        mask = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        lengths = enc["attention_mask"].sum(1) - 1
        for l in layers:
            h = hs[l].float()  # pool in fp32: fp16 sums over hundreds of tokens overflow
            mean = (h * mask.float()).sum(1) / mask.float().sum(1).clamp(min=1)
            last = h[torch.arange(h.shape[0], device=device), lengths]
            bufs[(l, "mean")][s : s + len(batch)] = mean.float().cpu().numpy().astype(np.float16)
            bufs[(l, "last")][s : s + len(batch)] = last.float().cpu().numpy().astype(np.float16)
        if (s // batch_size) % 20 == 0:
            print(f"  extract {s+len(batch)}/{N}", flush=True)

    for (l, p), arr in bufs.items():
        np.save(out / f"L{l}_{p}.npy", arr)
    manifest = {
        "model": model_name,
        "dataset_tag": dataset_tag,
        "dataset_hash": dataset_hash(examples),
        "ids": [ex.id for ex in examples],
        "layers": layers,
        "poolings": list(POOLINGS),
        "hidden_size": d,
        "n_layers": n_layers,
        "max_len": max_len,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


class Cache:
    """Read side of the activation cache."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.index = {i: k for k, i in enumerate(self.manifest["ids"])}
        self._mem: dict[tuple[int, str], np.ndarray] = {}

    @property
    def layers(self) -> list[int]:
        return self.manifest["layers"]

    def feats(self, layer: int, pooling: str, ids: list[str] | None = None) -> np.ndarray:
        key = (layer, pooling)
        if key not in self._mem:
            X = np.load(self.path / f"L{layer}_{pooling}.npy").astype(np.float32)
            bad = ~np.isfinite(X).all(1)
            if bad.any():
                print(f"  warn: {bad.sum()} rows non-finite in L{layer}_{pooling}; zero-filled")
                X[bad] = 0
            self._mem[key] = X
        X = self._mem[key]
        if ids is None:
            return X
        return X[[self.index[i] for i in ids]]
