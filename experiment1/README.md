# Experiment 1 — Unwrapping the Geometry & Building the Map

## Goal

This experiment tests a simple but consequential idea: that the internal
activations of a language model do not live in a flat, ordinary space, but on
the surface of a curved high-dimensional sphere. It also checks whether the
three behaviour classes we care about occupy that surface very differently —
with refusals clustered tightly together and jailbreaks spread out widely. If
both claims hold, they are the geometric groundwork on which the later
power-diagram decision boundary and its safety margin are built. In short, we
are unwrapping the geometry and drawing the first version of the map.

## The three classes (assigned by behaviour, not by label)

We do not trust a prompt's dataset label alone. Instead we run the model, read
what it actually did, and classify the activation by the model's own behaviour:

- **Refusal** — a harmful prompt that the model *refused*.
- **Jailbreak** — a harmful prompt that the model *complied* with.
- **Benign** — a harmless prompt that the model *complied* with.

A harmless prompt that the model refuses is "over-refusal." It is not one of
our three classes, so it is skipped.

Prompts are drawn from the project's four datasets — SORRY-Bench,
WildJailbreak, ToxicChat, and Aegis 2.0 — loaded through
`../dataset/load_datasets.py`.

## The three steps

The pipeline is three sequential, self-contained scripts. Each reads the
previous step's output and writes its own.

### Step 1 — Collect activations

**What:** We run an instruction-tuned LLM (default
`meta-llama/Llama-3.1-8B-Instruct`) on a balanced sample of prompts, let it
generate a response, detect whether it refused, and assign one of the three
classes above. For each prompt we save the last-token residual-stream
activation — the model's internal state — at layers **1, 8, 16, 24, and 32**.

**Why:** Sampling several layers lets us watch the geometry form. Early layers
carry surface features; middle and late layers carry the model's "intent."
Recording several layers means we can later choose where the classes separate
most cleanly.

### Step 2 — Unwrap the sphere with Principal Nested Spheres (PNS)

**What:** We first reduce dimensionality with PCA, project the points onto the
unit sphere `S^{k-1}`, then fit full **Principal Nested Spheres** (Jung,
Dryden & Marron, 2012) to unwrap the sphere into nested polar coordinates.

**Why:** Ordinary PCA flattens the space by pressing it onto straight lines.
On a curved surface that flattening destroys angles — and angles are exactly
what encode directional intent. PNS instead peels the sphere one nested layer
at a time, like removing the skins of an onion. It keeps the **radius** as a
faithful record of magnitude (`||x||`) and the **angles** as a faithful record
of direction. The result is a coordinate system that respects the curvature
rather than fighting it.

A sanity plot (`raw_vs_polar_layer{L}.png`) shows the raw sphere beside the
unwrapped polar view. If the geometry is real, Refusal points sit tightly
grouped while Jailbreak points spread out.

### Step 3 — Build sites and covariance

**What:** In the unwrapped PNS space we compute, for each class `c`, a mean
vector `mu_c` and a full covariance matrix `Sigma_c`. We write these out and
summarise each class's spread with the trace and the log-generalized-variance
of its covariance.

**Why:** The mean `mu_c` is the class's "home" — its centre of mass. The
covariance `Sigma_c` describes how widely and in which directions the class
spreads around that home. Comparing the spreads tells us whether the classes
are equally diffuse (they are not) — which is precisely why a naive
equal-distance boundary will fail and a power diagram is needed later.

## How to run

Figures and data are produced only when you run the scripts, in order:

1. `python step1_collect_activations.py`
2. `python step2_pns_unwrap.py`
3. `python step3_sites_covariance.py`

Useful flags:

| Flag | Purpose |
|------|---------|
| `--model` | Choose the LLM (default `meta-llama/Llama-3.1-8B-Instruct`). |
| `--sample-per-class` | How many prompts to collect per class. |
| `--no-4bit` | Disable 4-bit quantization (use full/bf16 weights). |
| `--layers` | Which layers to record (default `1 8 16 24 32`). |
| `--pca-dims` | PCA dimensionality before projecting to the sphere. |

## Requirements and hardware notes

- A **CUDA GPU** and **Python 3.10+** are required.
- Llama-3.1-8B is **gated**: accept the model terms on Hugging Face and provide
  an HF token (`huggingface-cli login`).
- On a **12 GB GPU** (for example an RTX 5070) the 8B model needs **4-bit
  quantization** (on by default, via `bitsandbytes`). Alternatively use a
  smaller model such as `meta-llama/Llama-3.2-3B-Instruct` in bf16.
- **RTX 50-series (Blackwell)** GPUs need a recent CUDA build of PyTorch
  (cu124 or newer).
- On **Windows**, WSL2 is recommended for CUDA plus `bitsandbytes`.

See `requirements.txt` for the Python dependencies.

## Outputs and folder layout

Written to `data/` (one file per layer `L`):

- `activations_layer{L}.npy` — last-token residual-stream activations.
- `labels.csv` — behavioural class per prompt.
- `collection_meta.json` — run configuration and metadata.
- `pns_layer{L}.npy` — points in unwrapped PNS (polar) coordinates.
- `pns_model_layer{L}.npz` — the fitted PNS model parameters.
- `sites_layer{L}.npz` — per-class `mu_c` and `Sigma_c`.
- `class_spread.csv` — trace and log-generalized-variance per class.

Written to `figures/`:

- `raw_vs_polar_layer{L}.png` — raw sphere vs unwrapped polar view.
- `class_spread_layer{L}.png` — bar chart of per-class spread.

## What success looks like

The raw-vs-polar plot visibly separates the classes, with Refusal clustered
tightly and Jailbreak diffuse. And `class_spread.csv` shows both trace(`Sigma_c`)
and log-generalized-variance ordered **Refusal < Benign < Jailbreak**. That
ordering is the evidence we are after: the classes have genuinely mismatched
covariances. That mismatch is what motivates the power-diagram treatment in the
later experiments, where unequal spreads are handled properly rather than
assumed away.
