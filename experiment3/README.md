# Experiment 3 — baselines, benchmarks, and the fractal diagnostic

Experiment 3 asks the question Experiments 1 and 2 set up but never answered: **does the
covariance-aware boundary actually win**, and where? It reuses the Experiment 1 activations and
the Experiment 2 discriminants, adds a linear probe and a plain Voronoi as baselines, and scores
everything with a single ROC-AUC so the comparison is like-for-like.

## Runnable now (no new data)

These three steps run on the existing Qwen-3B collection (`experiment1/data`) on CPU.

```
step1_auc_shootout.py   per-layer k-fold ROC-AUC of every detector on a binary task
                        (default Jailbreak vs Benign). This is the plan §7 caveat-2 test:
                        does the power diagram's edge over the linear probe widen at L24-34?
step2_cross_family.py   leave-one-source-dataset-out generalization; in-dist vs held-out AUC
                        and the drop per detector (the deployment condition)
step3_fractal.py        covariance participation ratio D = trace(Sigma)^2/||Sigma||_F^2 per
                        class per layer; expects D(Jailbreak) >> D(Refusal)
```

Detectors (`detectors.py`): `linear_probe`, `plain_voronoi`, `lda`, `qda`, `power_single`,
`power_multi`. The geometry ones reuse Experiment 2's discriminants. Because the power methods use
scalar weights they are genuinely distinct from `qda`, so "power vs qda" here is a real
comparison, not the §6.1 formula-against-itself trap.

The four source datasets (SORRY-Bench, WildJailbreak, ToxicChat, Aegis 2.0) each contribute
Jailbreak and Benign rows, which is what makes the leave-one-out cross-family split possible.

## Needs new data collection (scaffold documents, does not fake)

These parts of the plan §5 Experiment 3 are intentionally **not** implemented, because the data
they need was not collected:

- **Cross-model consistency** (Llama, Mistral vs Qwen; report AUC variance). Requires re-running
  `experiment1/step1_collect_activations.py` on each model (GPU). Once those runs exist, a thin
  aggregator over each model's `auc_by_layer_*.csv` gives the variance — deferred until the runs
  are done, so we do not pretend single-model numbers are cross-model.
- **Over-refusal (FPR / precision on benign-but-scary prompts).** The current labeling skipped the
  over-refusal quadrant, so there are no activations for it. Needs an OR-Bench-style collection
  pass with the over-refusal class kept.
- **Layer-stability / early-exit horizon** is read directly off the step-1 AUC-vs-layer curve
  (where AUC plateaus), so it is reported qualitatively from step 1 rather than as its own script.

## Open decision this experiment informs

The first Experiment 2 run found QDA beating the power diagram on accuracy in both raw and PNS
space, and PNS *widening* rather than closing the gap (see `CLAUDE.local.md` §6 and the geometry
decision memory). Step 1 here is the sharper test: AUC, per layer, probe vs geometry. If the power
diagram never overtakes the linear probe at the horizon, that is strong evidence for owning the
method as QDA-in-space (resolution 2) rather than as a power diagram. Report the curve to the PI
before committing the framing.

## Code layout

```
experiment3/
  lib/          detectors.py (linear probe, plain Voronoi, and adapters over the Experiment 2
                discriminants, imported as experiment2.lib.discriminants)
  pipeline/     the ordered steps: step1_auc_shootout -> step2_cross_family -> step3_fractal
  data/ figures/  generated outputs (gitignored)
```

Scripts run from the repo root; they reuse `experiment2.lib` for the working space and
discriminants, so Experiment 2 must sit alongside this folder.

## Running

```
python experiment3/pipeline/step1_auc_shootout.py --space rawpca
python experiment3/pipeline/step2_cross_family.py --space rawpca
python experiment3/pipeline/step3_fractal.py --space rawpca
```

Add `--space pns` to repeat in PNS space, `--positive Refusal --negative Benign` for the
refusal-detection task, or `--layers 24 28 30 34` to restrict. Outputs (CSVs, figures) land in
`experiment3/data` and `experiment3/figures`, both gitignored.

Dependencies: numpy, pandas, scikit-learn, matplotlib (already in `experiment1/requirements.txt`).
