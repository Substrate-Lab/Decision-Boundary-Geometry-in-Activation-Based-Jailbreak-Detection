# Guide for Yip Yip

You already ran the Qwen collection, so this is the short version — what changes, what to watch,
and the one recommendation I'd push back on you about. Every VRAM and timing number below is
read straight out of your own run report. Full detail lives in
[`RUNBOOK_LLAMA.md`](RUNBOOK_LLAMA.md); you shouldn't need it.

---

## Read this first: use the 3B, not the 8B

Your own run report says the Qwen-3B collection peaked at **8.23 GB reserved on a 7.96 GB card.**

```
NVIDIA GeForce RTX 5070 Laptop GPU · 7.96 GB · capability 12.0
peak allocated   4.25 GB
peak reserved    8.23 GB      ← already over the card
headroom        -0.27 GB
```

You were at the ceiling with a 3B model in 4-bit. Llama-3.1-8B is roughly **2.8× the weights**
(~5.6 GB vs ~2.0 GB in nf4) before any KV cache. At `--batch-size 4` with 1024-token prompts it
will OOM.

**There's also a scientific reason, and I think it's the stronger one.** Qwen2.5-**3B** vs
Llama-3.1-**8B** confounds two variables — model family *and* scale. If the geometry differs you
won't know which caused it. Llama-3.2-**3B**-Instruct is the same scale as your Qwen run, so the
comparison isolates family cleanly.

```powershell
python experiment1\pipeline\run_llama.py --model meta-llama/Llama-3.2-3B-Instruct --dry-run
```

If you want the 8B anyway, it's your call — use `--batch-size 1` and expect ~2 hours.

---

## The one rule

**Do not run `step1_collect_activations.py` directly.** It writes to fixed filenames with no model
in the path and will overwrite your committed Qwen `labels.csv`, `labels_full.csv` and
`collection_meta.json`. `run_llama.py` redirects output into
`experiment1\data\runs\<model>\` so your Qwen data is never touched.

---

## Run it

```powershell
# 1. preflight — loads nothing, ~10 seconds
python experiment1\pipeline\run_llama.py --model meta-llama/Llama-3.2-3B-Instruct --dry-run

# 2. collect
python experiment1\pipeline\run_llama.py --model meta-llama/Llama-3.2-3B-Instruct
```

Your environment already satisfies every preflight check — python 3.11.9, torch 2.11.0+cu128
(Blackwell needs cu124+, you have cu128), transformers 5.15.1, bitsandbytes working. The only one
that might fail is **model access**: Llama is gated, so if you haven't accepted the terms on the
HF page, do that and `huggingface-cli login`. Preflight catches it on `config.json`, before any
multi-GB download.

If you already have Llama weights locally, point at them and skip the download entirely:

```powershell
python experiment1\pipeline\run_llama.py --model C:\Users\ethan\models\Llama-3.2-3B-Instruct
```

Settings default to your Qwen values — 500/class, pools 1400/700, batch 4, 24 new tokens, 4-bit,
seed 0 — so you don't have to remember what you used.

---

## Layers — the back half only

You found layer **30 of 36** for Qwen, which is **0.83 of depth**. So there's no reason to spend
GPU time on early layers again. The default is the back half at stride 2:

| model | depth | layers collected | count |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 28 | `14,16,18,20,22,24,26,28` | 8 |
| Llama-3.1-8B-Instruct | 32 | `16,18,20,22,24,26,28,30,32` | 9 |

Override with `--layers 20,24,28` to go cheaper, or `--stride 1` for every block.

---

## What to expect

| | your Qwen run | Llama-3.2-3B estimate |
|---|---|---|
| prompts | 33 s | ~33 s (same pools) |
| model load | 14 s | ~15 s |
| collection | **12.4 min** | ~12–15 min |
| peak reserved | 8.23 GB | similar — watch it |
| output size | — | **~330 MB** (8 layers, hidden 3072) |

If it OOMs, drop `--batch-size` to 2 and it'll roughly double the collection time. Don't change
anything else — the other settings are what make it comparable to Qwen.

---

## Before you walk away, check it landed

```powershell
python - <<'PY'
import json, glob, numpy as np, pandas as pd
d = "experiment1/data/runs/Llama-3.2-3B-Instruct/"
m = json.load(open(d + "collection_meta.json"))
print("model      ", m["model_id"])
print("layers     ", m["layers"])
print("hidden dim ", m["hidden_dim"])
print("balanced   ", m["balanced_class_counts"])
lab = pd.read_csv(d + "labels.csv")
print("label rows ", len(lab), dict(lab.class_label.value_counts()))
for f in sorted(glob.glob(d + "activations_layer*.npy"))[:2]:
    print(f.split('/')[-1], np.load(f).shape)
PY
```

**Expect:** `hidden_dim` **3072** (Qwen was 2048), 500 per class, 1500 label rows.

**The interesting failure mode:** if the class counts come out badly skewed, that's not a bug —
it means Llama's refusal behaviour differs from Qwen's. Your Qwen run had a 49.4% refusal rate on
harmful prompts and 50.6% jailbreak compliance. If Llama refuses far more, you'll run short of
Jailbreak samples. **Report the actual numbers rather than re-running with different settings** —
that skew is a real result about the model, and changing settings to hide it would cost us the
finding.

---

## Sending it back

`experiment1\data\` is gitignored — **don't commit any of it.**

- **Zip the whole `runs\Llama-3.2-3B-Instruct\` folder** (~330 MB) if you can. The `.npy` arrays
  are what the Experiment 2 geometry actually needs.
- If that's awkward, `collection_meta.json` + `analytics\run_*.json` + `labels.csv` is ~2 MB and
  gets us norm statistics and class balance — but **not** the power diagrams.

---

## Honest status

- The preflight, layer selection, path handling and output redirection are tested. **The
  collection path itself has not been run on a GPU by us** — you're the first. If it throws, send
  the traceback verbatim rather than working around it.
- Steps 2 and 3 still write model-untagged CSVs to the flat `experiment1\data\`. **Tonight is
  collection only** — don't run them yet or they'll overwrite Qwen's layer CSVs the same way
  step 1 would have.
- `experiment1\README.md` documents Llama throughout because that's `step1`'s default; the
  committed data is Qwen. Not a mistake in your run, just a stale doc.
