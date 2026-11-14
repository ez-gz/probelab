# probelab

Cheap activation monitors for LLMs. Extract residual-stream activations **once**, train **many** tiny probes in
seconds, evaluate at **deployment operating points** (recall at fixed FPR, sliced by distribution shift), and
close the loop with **active-learning selection** of what to label next.

Primary use: safety/alignment monitors (harmful intent, deception, eval-awareness, sandbagging).
Same pipeline for custom user-aligned concepts: a concept is just a label key in the JSONL.

## Why this shape

- **Cache is the cost center, probes are free.** One forward pass per example, pooled activations stored per layer.
  Every probe variant, threshold, and re-train reads the cache. Iteration loop is seconds, not GPU-hours.
- **Eval at the operating point.** AUC hides everything. We report recall at FPR ∈ {0.1%, 1%, 5%} with the
  threshold chosen on val and *realized* FPR reported on test, plus bootstrap CIs. Shift slices (`jailbreak`,
  `paraphrase`, `lang:*`, `source`) are rows in the table, so a probe that learned the wrong feature shows up.
- **Gets better with labels.** `select` scores an unlabeled pool with the current probe and picks the
  uncertain / high-scoring / diverse examples. Label, append to train, re-sweep.

## Quickstart (runs on a laptop in ~30s)

```bash
uv venv && uv pip install -e . pytest joblib
uv run python scripts/make_toy.py                                       # toy set w/ shift slices
uv run probelab extract data/toy_harm.jsonl Qwen/Qwen3-1.7B --layers 4,8,12,16,20,24,28
uv run probelab sweep   data/toy_harm.jsonl Qwen/Qwen3-1.7B harmful_intent --out runs/toy_sweep.json
uv run probelab train   data/toy_harm.jsonl Qwen/Qwen3-1.7B harmful_intent --layer 16 --pooling last --fpr 0.01
uv run probelab select  probes/harmful_intent__L16_last_logistic.json data/toy_harm.jsonl --strategy diverse --k 20
```

Real data: `uv run python scripts/build_harmful.py` (toxic-chat + JailbreakBench + alpaca), then same commands with
`data/harmful_intent.jsonl` and `--layers all`.

## What the toy run already shows

Best tier-0 probe on Qwen3-1.7B hits recall 1.0 in-distribution, but the `jailbreak` slice has **FPR 1.0**:
benign requests wrapped in DAN framing get flagged. The probe learned "jailbreak-shaped text", not "harmful
intent". A threshold set on shift-free val realizes 29% FPR on the shifted test set. That is the whole
argument for this harness: the metric that matters is only visible when shift is a slice.

## Results: harmful intent on Qwen3-1.7B

`results/qwen3-1.7b_harmful_intent.json` is the full sweep: 112 probes (28 layers x mean/last pooling x
logistic/mass-mean) trained on 14,365 prompts from toxic-chat, JailbreakBench and Alpaca, and scored on 6,634
held-out test prompts, 526 of them harmful. Thresholds are set on val; everything below is measured on test.

Best probe by recall at 1% FPR: layer 15, mean pooling, logistic. AUROC 0.956.

| target FPR | recall | realized FPR on test | precision |
|---|---|---|---|
| 0.1% | 0.17 | 0.05% | 0.97 |
| 1% | 0.70 (95% CI 0.66-0.74) | 1.3% | 0.82 |
| 5% | 0.87 | 7.8% | 0.49 |

The same probe at the 1% threshold, by slice:

| slice | n | recall | FPR |
|---|---|---|---|
| toxic-chat (in distribution) | 5,864 | 0.72 | 1.1% |
| toxic-chat prompts tagged as jailbreaks (all harmful) | 112 | 0.93 | n/a |
| JailbreakBench harmful/benign pairs (test only, never trained on) | 200 | 0.58 | 18% |
| Alpaca benign instructions | 570 | n/a | 0.0% |

What this says:

- **The headline number hides the JailbreakBench row.** On benign requests written as counterparts to harmful
  ones, the false-positive rate is 18%, not 1%, and recall drops to 0.58. It is the toy-run failure again at
  real scale: part of what the probe reads is "sounds like a harmful request", not intent.
- **Mean pooling beats last-token.** Best logistic probe: 0.70 recall at 1% FPR with mean pooling, 0.59 with the
  last token.
- **Middle layers carry it.** Best recall at 1% FPR per layer rises from about 0.5 in layers 1-5 to 0.70 at
  layers 15-16, then sits at 0.62-0.69 through layer 28.
- **Mass-mean is close to logistic** (0.67 vs 0.70) with no training loop.

## Layout

```
probelab/       the package
  schema.py     JSONL example schema (text|messages, labels{concept:0/1}, split, source, tags)
  extract.py    HF forward w/ output_hidden_states -> cache/<model>/<dataset>/L<l>_<pool>.npy + manifest
  probes.py     MassMean (LDA-lite), Logistic, MLP ; save/load ; ProbeSpec
  evals.py      threshold@FPR, recall/precision/realized-FPR, bootstrap CI, tag slices
  select.py     uncertain / top-positive / disagree / diverse (k-means) selection
  cli.py        extract | sweep | train | score | select
scripts/        make_toy.py (toy set with shift slices), build_harmful.py (the real dataset)
data/           the toy set; built datasets land here, git-ignored
results/        sweep outputs quoted in this README
tests/          pipeline tests on synthetic activations
```

## Roadmap

1. The same harmful-intent run on Llama-3-8B with all layers; publish the recall@FPR-by-layer plot.
2. Per-token cache (opt-in) + max-pooling over tokens; needed for long multi-turn traces.
3. Concept 2: deception / eval-awareness. Labels from model-generated pairs (honest vs instructed-to-lie),
   the contested case for linear readability.
4. SAE-feature probes as tier 1 (Gemma Scope / Llama Scope dictionaries) for interpretable "why".
5. Tier-0 → tier-1 cascade with `disagree` selection; cascade cost model.
6. Serving shim: hook into vLLM/HF generate and emit probe scores per request.
