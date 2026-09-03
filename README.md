# Coordinate Frames Determine the Visibility of Refusal Geometry

Code and derived results for a study of the *shape*, not just the location, of refusal and
compliance representations in instruction-tuned language models.

## Abstract

Refusal is largely mediated by a single direction in a transformer's residual stream. However,
this locates the refusal state without describing the underlying geometric shape. We investigate
if refusal and compliance representations differ in dispersion and demonstrate that it depends on
the coordinate frame. Tracking the dispersion across three open-weight models, we observe a sign
change at a model-specific fractional depth in two safety-aligned models; beyond which, refusal
representations become the widest class. In contrast, the model that refuses only 9.4% of harmful
prompts never establishes a sustained inversion. Crucially, in these models, the inversion layer
coincides with the exact layer-depth at which detection performance is plateauing. Contrary to our
initial hypothesis (that this heteroscedasticity would degrade linear probes with depth),
degradation proves to be frame-dependent rather than depth-dependent: linear probes are strong
into the deepest layers within PCA subspace, whereas quadratic boundaries dominate only in
Principal Nested Sphere (PNS) coordinates. Decomposing separability into mean and dispersion
reveals that dispersion carries 61–72% of the signal in PNS coordinates and accounts for variance
in two cases: separating jailbreaks from refusals than jailbreaks from benign prompts. Ultimately,
no probe geometry is universal across all evaluation axes: boundaries that discriminate best are
often the worst calibrated and degrade most severely under leave-one-corpus-out transfer. On the
other hand, scalar trace weights in power diagrams are rank-inert. We present these findings as a
proof of concept for measuring refusal geometry and nothing else.

## Scope

This is a measurement study, not a proposed detector. We make no causal claim that the geometry
constitutes refusal, no claim to beat linear probes at detection, and no universality claim beyond
the three models tested. Findings are reported as a proof of concept for measuring refusal
geometry.

## Models and data

Three open-weight instruction-tuned models, each collected under identical sampling settings
(500 prompts per class, 1500 total, balanced):

| Model | Hidden dim | Role |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | 2048 | primary collection, layers 2–36 (stride 2) |
| `meta-llama/Llama-3.1-8B-Instruct` | 4096 | cross-model replication |
| `mistralai/Mistral-7B-Instruct-v0.3` | 4096 | cross-model replication |

Prompts are drawn in equal quotas from four corpora — SORRY-Bench, WildJailbreak, ToxicChat, and
Aegis 2.0 — loaded via `dataset/load_datasets.py`. Several are gated on the Hugging Face Hub;
accept their terms and set an HF token before collecting.

Classes are assigned by **what the model actually did**, not by the prompt's dataset label:

- **Refusal** — harmful prompt, model refused.
- **Jailbreak** — harmful prompt, model complied.
- **Benign** — harmless prompt, model complied.

Harmless prompts that the model refuses (over-refusal) fall outside these three classes and are
skipped.

## Repository layout

```
dataset/        corpus loading (four sources, stratified quotas)
experiment1/    activation collection, PNS unwrapping, per-class sites and covariance
experiment2/    covariance-aware boundaries, margin calibration, dispersion decomposition
experiment3/    baselines and benchmarks: AUC shootout, cross-corpus transfer, fractal diagnostic
paper/          methodology write-up (LaTeX + HTML)
```

Each experiment directory carries its own README with the detail for that stage. Every script is
run **from the repository root** and resolves its own paths, regardless of which folder it lives
in.

Raw activation arrays (~4.3 GB) are gitignored. Everything derived from them — PNS scores, class
sites, result CSVs, and figures — is committed, so the analysis is reproducible without re-running
the GPU collection.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r experiment1/requirements.txt
```

Python 3.10+. Install a CUDA build of `torch` matching your GPU (cu124 or newer for RTX 50-series);
collection additionally needs a CUDA GPU and `bitsandbytes` for 4-bit quantization. Experiments 2
and 3 run on CPU from the committed data.

## Reproducing

### Experiment 1 — collection and geometry (requires a GPU)

```bash
python experiment1/pipeline/step1_collect_activations.py --model Qwen/Qwen2.5-3B-Instruct
python experiment1/pipeline/step2_pns_unwrap.py
python experiment1/pipeline/step3_sites_covariance.py
```

Step 1 writes to fixed filenames under `experiment1/data/`. To collect a second model without
overwriting the first, use the runner, which redirects output into `experiment1/data/runs/<model>/`:

```bash
python experiment1/pipeline/run_llama.py --model meta-llama/Llama-3.1-8B-Instruct --dry-run
python experiment1/pipeline/run_llama.py --model meta-llama/Llama-3.1-8B-Instruct
```

`--dry-run` is a preflight that loads no weights: it checks CUDA, VRAM, `bitsandbytes`, model
access, disk, and datasets, then prints the layer list it would collect. Gated-repo failures
surface here, before any multi-gigabyte download.

### Experiments 2 and 3 — boundaries and benchmarks (CPU, uses committed data)

```bash
python experiment2/pipeline/step1_boundary.py            --space rawpca
python experiment2/pipeline/step2_margin_calibration.py  --space rawpca
python experiment2/pipeline/step3_boundary_error.py      --space rawpca --gmm-k 2
python experiment2/pipeline/step4_dispersion_ratio.py
python experiment2/pipeline/step5_inversion_permutation.py
python experiment2/pipeline/step6_refusal_ablation.py

python experiment3/pipeline/step1_auc_shootout.py  --space rawpca
python experiment3/pipeline/step2_cross_family.py  --space rawpca
python experiment3/pipeline/step3_fractal.py       --space rawpca
```

Pass `--space pns` to repeat any of these in Principal Nested Spheres coordinates — the frame
contrast is the point of the study, so most results exist in both. Restrict work with
`--layers 24 30 34` and `--models power_multi qda`.

`python experiment2/verify.py` re-derives every number the Experiment 2 figures quote from its
source file, and checks the mathematical identities the geometry relies on. Run it after changing
anything in `experiment2/lib`.

## Where each result lives

| Claim in the abstract | File |
|---|---|
| Dispersion sign change / inversion depth, with permutation null | `experiment2/data/inversion_permutation.csv` |
| Mean-vs-dispersion decomposition of separability | `experiment2/data/dispersion_ratio.csv` |
| Per-layer detection AUC by detector and frame | `experiment3/data/auc_by_layer_{rawpca,pns}_*.csv` |
| Leave-one-corpus-out transfer | `experiment3/data/cross_family_*.csv` |
| Margin calibration and ECE | `experiment2/data/calibration_{rawpca,pns}.csv` |
| Per-class covariance spread | `experiment1/data/class_spread.csv`, `experiment1/data/runs/*/class_spread.csv` |
| Covariance participation ratio | `experiment3/data/fractal_*.csv` |
| Refusal-direction ablation | `experiment2/data/` (step 6 outputs) |

## Known limitations

- Steps 2 and 3 of Experiment 1 write model-untagged CSVs to a flat `experiment1/data/`; the
  committed layer CSVs therefore describe the primary Qwen collection only. Per-model downstream
  plumbing is not built.
- The cross-layer trajectory ("fractal z-state") is not operationalized — every step works within
  a single layer's space.
- Over-refusal (harmless prompts the model refuses) is not collected, so no false-positive rate on
  benign-but-alarming prompts is reported.
