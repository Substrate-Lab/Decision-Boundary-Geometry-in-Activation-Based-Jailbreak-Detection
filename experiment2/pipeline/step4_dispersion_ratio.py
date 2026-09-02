#!/usr/bin/env python3
"""Tier 1: rho(layer) = D_disp / D_B per model, per layer, per class pair, with intervals.

	Measures directly what share of each pair's separability is carried by dispersion rather
	than by a mean shift, so the result does not depend on which detector was fitted.

	Frames. The PNS frame runs from the committed pns_layer*.npy scores. The PCA frame cannot
	be computed from what is in the repo: step2_pns_unwrap.py L2-normalises the PCA scores onto
	the unit sphere before fitting the nested spheres (line 169) and saves only the residual
	angles, so the radial magnitude is discarded and never written. Reconstructing the nested
	spheres therefore recovers a direction on S^9 but not PCA coordinates. That frame needs
	activations_layer*.npy, which is gitignored and absent. The script reports the frame as
	unavailable rather than quietly emitting one frame and calling it the answer.

	Usage:
		.venv/bin/python experiment2/pipeline/step4_dispersion_ratio.py
		.venv/bin/python experiment2/pipeline/step4_dispersion_ratio.py --draws 2000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
	LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from experiment2.lib.dispersion import bootstrap_pair, pair_statistics

EXP1_DATA = REPO_ROOT / "experiment1" / "data"
PAIRS = [("Jailbreak", "Benign"), ("Jailbreak", "Refusal")]


# Every collected model: the original flat Qwen layout plus the per-model run directories.
def discover_models() -> list:
	found = []
	if (EXP1_DATA / "labels.csv").exists() and any(EXP1_DATA.glob("pns_layer*.npy")):
		meta = json.loads((EXP1_DATA / "collection_meta.json").read_text())
		found.append({"model": re.split(r"[\\/]", meta["model_id"])[-1], "dir": EXP1_DATA})
	runs = EXP1_DATA / "runs"
	if runs.exists():
		for directory in sorted(runs.iterdir()):
			if (directory / "labels.csv").exists() and any(directory.glob("pns_layer*.npy")):
				found.append({"model": directory.name, "dir": directory})
	return found


# Layer indices that actually have a PNS score file, in order.
def available_layers(directory: Path) -> list:
	layers = []
	for path in directory.glob("pns_layer*.npy"):
		match = re.search(r"pns_layer(\d+)\.npy$", path.name)
		if match:
			layers.append(int(match.group(1)))
	return sorted(layers)


# Whether the raw arrays needed for the PCA frame are present for this model.
def rawpca_available(directory: Path) -> bool:
	return any(directory.glob("activations_layer*.npy"))


# Lowest cross-validated error any of three fitted detectors achieves on the pair. Classes are
# equal-sized here, so this is the equal-prior error the Bhattacharyya bound applies to.
def best_detector_error(points, labels, class_a, class_b, seed=0) -> dict:
	keep = np.isin(labels, [class_a, class_b])
	X, y = points[keep], (labels[keep] == class_a).astype(int)
	splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
	errors = {}
	for name, estimator in [
		("logistic", LogisticRegression(max_iter=2000)),
		("lda", LinearDiscriminantAnalysis()),
		("qda", QuadraticDiscriminantAnalysis(reg_param=1e-3)),
	]:
		score = cross_val_score(estimator, X, y, cv=splitter, scoring="accuracy")
		errors[f"err_{name}"] = float(1.0 - score.mean())
	errors["err_best"] = min(errors.values())
	errors["best_detector"] = min(errors, key=lambda k: errors[k] if k.startswith("err_") and k != "err_best" else 2)
	return errors


# Scale-free spread contrast that replaces a raw Delta: comparable across models and frames.
def log_trace_ratio(points, labels, numerator: str, denominator: str) -> float:
	num = np.trace(np.cov(points[labels == numerator], rowvar=False))
	den = np.trace(np.cov(points[labels == denominator], rowvar=False))
	return float(np.log(num / den))


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--draws", type=int, default=1000, help="Bootstrap resamples per cell.")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--out", type=str, default="experiment2/data/dispersion_ratio.csv")
	args = parser.parse_args()

	models = discover_models()
	if not models:
		print("no collected models found under experiment1/data", file=sys.stderr)
		return 1

	print(f"models: {', '.join(entry['model'] for entry in models)}")
	missing_rawpca = [entry["model"] for entry in models if not rawpca_available(entry["dir"])]
	if missing_rawpca:
		print(f"\nPCA frame UNAVAILABLE for: {', '.join(missing_rawpca)}")
		print("  activations_layer*.npy absent; step2 discards the radial magnitude when it")
		print("  normalises onto the sphere, so PCA coordinates cannot be recovered from the")
		print("  committed PNS scores. Reporting the PNS frame only.\n")

	rows = []
	for entry in models:
		directory, model = entry["dir"], entry["model"]
		labels = pd.read_csv(directory / "labels.csv")["class_label"].to_numpy()
		layers = available_layers(directory)
		print(f"{model}  ({len(layers)} layers, n={len(labels)})")
		for layer in layers:
			points = np.load(directory / f"pns_layer{layer}.npy")
			for class_a, class_b in PAIRS:
				stats = pair_statistics(points, labels, class_a, class_b)
				stats.update(bootstrap_pair(
					points, labels, class_a, class_b,
					draws=args.draws, seed=args.seed,
					keys=("rho", "d_bhattacharyya"),
				))
				stats.update(best_detector_error(points, labels, class_a, class_b, args.seed))
				# Surrogate fails if a fitted detector beats the Gaussian floor.
				stats["below_floor"] = bool(stats["err_best"] < stats["error_lower"])
				stats["above_ceiling"] = bool(stats["err_best"] > stats["error_upper"])
				stats["log_trace_R_over_B"] = log_trace_ratio(points, labels, "Refusal", "Benign")
				rows.append({
					"model": model, "frame": "pns", "layer": layer,
					"pair": f"{class_a}_vs_{class_b}", "dims": points.shape[1], **stats,
				})

	frame = pd.DataFrame(rows)
	destination = REPO_ROOT / args.out
	destination.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(destination, index=False)

	for (model, pair), block in frame.groupby(["model", "pair"], sort=False):
		peak = block.loc[block["rho"].idxmax()]
		print(f"\n{model} — {pair.replace('_', ' ')}   (PNS frame)")
		print(f"  {'layer':>6}{'rho':>8}{'95% CI':>18}{'D_B':>9}{'log tr ratio':>14}")
		for _, row in block.iterrows():
			marker = "  <- peak" if row["layer"] == peak["layer"] else ""
			print(f"  {int(row['layer']):>6}{row['rho']:>8.3f}"
			      f"{f'[{row.rho_lo:.3f}, {row.rho_hi:.3f}]':>18}"
			      f"{row['d_bhattacharyya']:>9.3f}{row['log_trace_ratio']:>14.3f}{marker}")

	below = frame[frame["below_floor"]]
	print("\n" + "=" * 66)
	print("Gaussian surrogate check — equal-prior Bhattacharyya bracket")
	print(f"  floor   P_e >= 0.5*(1-sqrt(1-exp(-2 D_B)))   breached by {len(below)}/{len(frame)}")
	print(f"  ceiling P_e <= 0.5*exp(-D_B)                 exceeded by "
	      f"{int(frame['above_ceiling'].sum())}/{len(frame)} (suboptimal fit, not surrogate failure)")
	if below.empty:
		margin = frame["err_best"] - frame["error_lower"]
		print("\n  No detector beats the Gaussian floor. Surrogate is self-consistent.")
		print(f"  Tightest cell sits {margin.min():.4f} above it "
		      f"({frame.loc[margin.idxmin(), 'model']} L"
		      f"{int(frame.loc[margin.idxmin(), 'layer'])} "
		      f"{frame.loc[margin.idxmin(), 'pair']}).")
	else:
		print("\n  Detectors beat the Gaussian floor here — report the breach, not the bound:")
		for _, row in below.iterrows():
			print(f"    {row['model']} L{int(row['layer'])} {row['pair']}: "
			      f"err {row['err_best']:.4f} < floor {row['error_lower']:.4f}")

	print("\n" + "=" * 66)
	print("Tier 1 #3 — log(tr S_Refusal / tr S_Benign), PNS frame")
	pivot = (frame[frame["pair"] == "Jailbreak_vs_Benign"]
	         .pivot(index="layer", columns="model", values="log_trace_R_over_B"))
	print(pivot.round(3).to_string())

	print(f"\nwrote {destination.relative_to(REPO_ROOT)}  ({len(frame)} rows)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
