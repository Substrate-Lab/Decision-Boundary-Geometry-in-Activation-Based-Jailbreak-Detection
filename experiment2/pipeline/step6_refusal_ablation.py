#!/usr/bin/env python3
"""Does the dispersion signal survive removing the refusal direction?

	The worry the Jailbreak-vs-Refusal numbers have to answer is that rho is not measuring
	dispersion at all, but re-expressing a single mean-difference direction -- the refusal
	direction of Arditi et al. If projecting that direction out collapses rho, the dispersion
	story is a restatement of a known result. If rho survives, it is carrying something the
	direction does not.

	The direction is the standard difference of means, computed in the working frame:

		r = mean(Refusal) - mean(Jailbreak union Benign),   rhat = r / ||r||

	and ablation is the rank-one projection X -> X - (X rhat) rhat^T, applied to every point
	before any class statistic is recomputed.

	What this is NOT. This ablates inside the 9-dimensional PNS frame, not the 4096-dimensional
	residual stream, so it is not a reproduction of Arditi's intervention and should not be
	described as one. The raw-activation version needs activations_layer*.npy, which is absent.
	Two consequences run in opposite directions and both belong in any write-up: removing one of
	nine dimensions is a far harsher cut than removing one of 4096, so survival here is strong
	evidence; but the PNS frame has already discarded the radial coordinate, so a refusal
	direction living mostly in that radial component would never appear here to be removed.

	Usage:
		.venv/bin/python experiment2/pipeline/step6_refusal_ablation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from experiment2.lib.dispersion import log_trace_ratio, pair_statistics, total_variance
from experiment2.pipeline.step4_dispersion_ratio import available_layers, discover_models

PAIRS = [("Jailbreak", "Benign"), ("Jailbreak", "Refusal")]


# Unit difference-of-means refusal direction in whatever frame the points are given in.
def refusal_direction(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
	direction = points[labels == "Refusal"].mean(axis=0) - points[labels != "Refusal"].mean(axis=0)
	norm = np.linalg.norm(direction)
	if norm < 1e-12:
		raise ValueError("refusal direction is degenerate")
	return direction / norm


# Re-express the points in the orthogonal complement of one direction.
#
# Zeroing the component instead (X - (X rhat) rhat^T) leaves the cloud inside an 8-dimensional
# subspace while still writing it in 9 coordinates, so every class covariance comes out
# rank-deficient, det = 0, and the Bhattacharyya log-determinants diverge. Projecting onto an
# orthonormal basis of the complement is the same geometric operation with a non-degenerate
# representation, and every statistic downstream stays defined.
#
# The consequence to keep in mind when reading the output: D_B after is computed in 8 dimensions
# and D_B before in 9, so their ratio is a retention figure, not a like-for-like distance. rho
# is a within-frame share and remains the comparable quantity.
def ablate(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
	basis, _ = np.linalg.qr(np.column_stack([direction, np.eye(len(direction))]))
	return points @ basis[:, 1:]


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--out", type=str, default="experiment2/data/refusal_ablation.csv")
	args = parser.parse_args()

	rows = []
	for entry in discover_models():
		directory, model = entry["dir"], entry["model"]
		labels = pd.read_csv(directory / "labels.csv")["class_label"].to_numpy()
		for layer in available_layers(directory):
			points = np.load(directory / f"pns_layer{layer}.npy")
			direction = refusal_direction(points, labels)
			ablated = ablate(points, direction)

			# Share of the frame's total spread that the removed direction carried.
			variance_removed = 1.0 - total_variance(ablated) / total_variance(points)

			for class_a, class_b in PAIRS:
				before = pair_statistics(points, labels, class_a, class_b)
				after = pair_statistics(ablated, labels, class_a, class_b)
				rows.append({
					"model": model, "layer": layer, "frame": "pns",
					"pair": f"{class_a}_vs_{class_b}",
					"rho_before": before["rho"], "rho_after": after["rho"],
					"rho_delta": after["rho"] - before["rho"],
					"d_b_before": before["d_bhattacharyya"],
					"d_b_after": after["d_bhattacharyya"],
					"d_b_retained": after["d_bhattacharyya"] / before["d_bhattacharyya"],
					"d_mean_before": before["d_mean"], "d_mean_after": after["d_mean"],
					"d_disp_before": before["d_disp"], "d_disp_after": after["d_disp"],
					"delta_before": log_trace_ratio(points, labels, "Refusal", "Benign"),
					"delta_after": log_trace_ratio(ablated, labels, "Refusal", "Benign"),
					"frame_variance_removed": variance_removed,
				})

	frame = pd.DataFrame(rows)
	destination = REPO_ROOT / args.out
	frame.to_csv(destination, index=False)

	for (model, pair), block in frame.groupby(["model", "pair"], sort=False):
		block = block.sort_values("layer")
		print(f"\n{'=' * 80}\n{model} — {pair.replace('_', ' ')}   (PNS frame, refusal direction removed)")
		print(f"  {'layer':>6}{'rho before':>12}{'rho after':>11}{'change':>9}"
		      f"{'D_B kept':>10}{'Delta before':>14}{'Delta after':>13}")
		for _, row in block.iterrows():
			print(f"  {int(row['layer']):>6}{row['rho_before']:>12.3f}{row['rho_after']:>11.3f}"
			      f"{row['rho_delta']:>+9.3f}{row['d_b_retained']:>10.2f}"
			      f"{row['delta_before']:>+14.3f}{row['delta_after']:>+13.3f}")
		print(f"  mean rho {block['rho_before'].mean():.3f} -> {block['rho_after'].mean():.3f}"
		      f"   |  mean D_B retained {block['d_b_retained'].mean():.2f}"
		      f"   |  frame variance removed {block['frame_variance_removed'].mean():.1%}")

	print(f"\nwrote {destination.relative_to(REPO_ROOT)}  ({len(frame)} rows)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
