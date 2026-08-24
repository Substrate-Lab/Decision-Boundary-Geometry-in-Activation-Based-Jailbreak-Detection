# Experiment 2 — the covariance-aware boundary and margin calibration

Experiment 2 builds the class boundary on top of the activations collected in Experiment 1 and
turns the distance-to-boundary into a calibrated confidence. It reuses `experiment1/data`
directly and adds nothing to the GPU collection step.

## The open decision this scaffold keeps live

The research plan flags an unresolved tension (§6.1): the full-covariance discriminant
`P_i(x) = (x - μ_i)ᵀ Σ_i⁻¹ (x - μ_i) + log det Σ_i` is exactly **QDA**, not a power diagram. If
adopted as the method, the "exact hyperplane / convex cell / clean power margin" story is lost,
the Alexandrov optimal-transport guarantee no longer applies, and the method collapses into its
own Experiment 3 baseline.

So the boundary is **pluggable**, not hardcoded, in `discriminants.py`:

| model          | boundary        | is it a power diagram?          | plan resolution |
|----------------|-----------------|---------------------------------|-----------------|
| `lda`          | linear          | no (shared-covariance baseline) | baseline        |
| `power_single` | linear          | yes, single-site Laguerre       | resolution 1    |
| `power_multi`  | piecewise-linear| yes, multi-site Laguerre        | resolution 3 (recommended) |
| `qda`          | quadric         | **no — this is QDA**            | resolution 2    |

The default across the steps is `power_multi`. `qda` is included so the comparison can be made
empirically, but it is labelled QDA everywhere and never presented as "our power diagram." The
PI decision on which to headline is still pending — see `CLAUDE.local.md` §6.

## What is implemented vs deferred

- Covariances use **Ledoit-Wolf shrinkage** so `Σ⁻¹` and `log det Σ` stay defined at small n (§6.2).
- Both working spaces are selectable: `rawpca` (raw activations, PCA-reduced) and `pns`
  (Experiment 1 step-2 PNS scores). PNS is kept optional because it *raises* covariance
  similarity and so shrinks the mismatch the boundary exploits (§6.3 / §7 caveat 3).
- The **fractal z-state** (cross-layer trajectory, §6.3) is **not** operationalized here; each
  step works in a single layer's space. Treat it as out of scope for this scaffold.
- Per-layer AUC vs baselines (the "does the method actually win" test, §7 caveat 2) belongs to
  Experiment 3 and is **not** built here.

## Pipeline

```
step1_boundary.py            fit each model on a train split, assign cells on a held-out test
                             split, log per-point geometric margins + boundary_summary.csv
step2_margin_calibration.py  rank-normalize the margin to a confidence, bin it against accuracy,
                             report ECE and pooled reliability curves
step3_boundary_error.py      fit a per-class GMM reference, measure RMS boundary gap and cell
                             disagreement per model (qda vs k=1 GMM is flagged degenerate)
```

## Code layout

```
experiment2/
  lib/          shared code: space.py (working representation, labels, layer resolution,
                anchored data/output dirs) and discriminants.py (the four models)
  pipeline/     the ordered steps: step1_boundary -> step2_margin_calibration -> step3_boundary_error
  data/ figures/  generated outputs (gitignored)
```

Scripts run from the repo root and read the Experiment 1 activations via `lib/space.py`.

## Running

```
python experiment2/pipeline/step1_boundary.py --space rawpca
python experiment2/pipeline/step2_margin_calibration.py --space rawpca
python experiment2/pipeline/step3_boundary_error.py --space rawpca --gmm-k 2
```

Restrict with `--layers 24 30 34` and `--models power_multi qda`. Outputs (CSVs, figures) land in
`experiment2/data` and `experiment2/figures`, both gitignored like Experiment 1.

Dependencies: numpy, pandas, scikit-learn, matplotlib (already in `experiment1/requirements.txt`).
