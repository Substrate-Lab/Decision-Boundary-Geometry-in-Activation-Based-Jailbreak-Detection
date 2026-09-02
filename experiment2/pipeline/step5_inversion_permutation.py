#!/usr/bin/env python3
"""Significance for the Refusal/Benign spread inversion, by label permutation.

	The headline claim is a sign change in Delta = log(tr S_Refusal / tr S_Benign) at a single
	layer: early layers have Refusal tighter than Benign, deep layers reverse it. That claim has
	been carried by the point estimate alone. This supplies the missing null.

	The null is that Refusal and Benign are interchangeable labels on the same activations. It
	is built by pooling only those two classes, reshuffling the assignment, and recomputing
	Delta -- Jailbreak rows are excluded rather than mixed in, and both class sizes are held
	fixed, so a draw differs from the observed statistic in label assignment alone.

	Reported per layer: Delta with a bootstrap interval, the permutation null's spread, the
	z-score against it, and a two-sided p. The p uses (1 + count) / (1 + draws), which floors at
	1/(1+draws) rather than returning a zero the resampling cannot support.

	Multiplicity is real -- roughly seventeen layers per model -- so a Holm-adjusted p is
	reported alongside the raw one, and the headline quotes the adjusted value.

	Usage:
		.venv/bin/python experiment2/pipeline/step5_inversion_permutation.py --draws 1000
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

from experiment2.lib.dispersion import bootstrap_delta, permutation_delta
from experiment2.pipeline.step4_dispersion_ratio import available_layers, discover_models

NUMERATOR, DENOMINATOR = "Refusal", "Benign"


# Holm step-down adjustment, which controls the family-wise error rate without assuming the
# layer tests are independent -- they are not, being nested layers of one network.
def holm(pvalues: np.ndarray) -> np.ndarray:
	order = np.argsort(pvalues)
	adjusted = np.empty(len(pvalues))
	running = 0.0
	for rank, index in enumerate(order):
		running = max(running, (len(pvalues) - rank) * pvalues[index])
		adjusted[index] = min(1.0, running)
	return adjusted


# Every layer where Delta changes sign from the previous layer.
#
# Counting these is what separates a single-layer inversion from an oscillation. A tail-only
# test ("does the sign settle opposite to the first layer?") returns a layer number either way,
# which would let a curve that crosses zero three times be reported as one clean sign change.
def sign_crossings(layers: list, deltas: np.ndarray) -> list:
	signs = np.sign(deltas)
	return [layers[i] for i in range(1, len(signs))
	        if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]]


# The layer at or after which Delta's sign settles and stays opposite to the shallowest layer.
def sustained_inversion(layers: list, deltas: np.ndarray):
	signs = np.sign(deltas)
	if signs[0] == 0:
		return None
	for position in range(1, len(signs)):
		tail = signs[position:]
		if np.all(tail == tail[0]) and tail[0] == -signs[0]:
			return layers[position]
	return None


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--draws", type=int, default=1000)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--out", type=str, default="experiment2/data/inversion_permutation.csv")
	args = parser.parse_args()

	rows = []
	for entry in discover_models():
		directory, model = entry["dir"], entry["model"]
		labels = pd.read_csv(directory / "labels.csv")["class_label"].to_numpy()
		layers = available_layers(directory)
		print(f"{model}: {len(layers)} layers x {args.draws} permutations")
		for layer in layers:
			points = np.load(directory / f"pns_layer{layer}.npy")
			record = {"model": model, "layer": layer, "frame": "pns"}
			record.update(permutation_delta(points, labels, NUMERATOR, DENOMINATOR,
			                                draws=args.draws, seed=args.seed))
			record.update(bootstrap_delta(points, labels, NUMERATOR, DENOMINATOR,
			                              draws=args.draws, seed=args.seed))
			rows.append(record)

	frame = pd.DataFrame(rows)
	frame["p_holm"] = np.concatenate([
		holm(block["p_two_sided"].to_numpy())
		for _, block in frame.groupby("model", sort=False)
	])
	destination = REPO_ROOT / args.out
	frame.to_csv(destination, index=False)

	for model, block in frame.groupby("model", sort=False):
		block = block.sort_values("layer")
		layers = block["layer"].tolist()
		deltas = block["delta"].to_numpy()
		inversion = sustained_inversion(layers, deltas)
		crossings = sign_crossings(layers, deltas)

		print(f"\n{'=' * 78}\n{model} — Delta = log(tr S_Refusal / tr S_Benign), PNS frame")
		print(f"  {'layer':>6}{'Delta':>9}{'95% CI':>18}{'null sd':>9}{'z':>8}"
		      f"{'p':>8}{'p_holm':>9}{'sign':>7}")
		for _, row in block.iterrows():
			mark = " <-" if inversion is not None and row["layer"] == inversion else ""
			print(f"  {int(row['layer']):>6}{row['delta']:>9.3f}"
			      f"{f'[{row.delta_lo:+.2f}, {row.delta_hi:+.2f}]':>18}"
			      f"{row['null_sd']:>9.3f}{row['z_vs_null']:>8.1f}"
			      f"{row['p_two_sided']:>8.4f}{row['p_holm']:>9.4f}"
			      f"{row['sign_stability']:>7.2f}{mark}")

		floor = float(block["p_floor"].iloc[0])
		print(f"\n  sign crossings: {len(crossings)}" +
		      (f" at layer(s) {', '.join(str(c) for c in crossings)}" if crossings else ""))

		if inversion is None or len(crossings) != 1:
			print(f"  NOT a single-layer inversion. Delta crosses zero {len(crossings)} times, "
			      f"so the headline claim")
			print("  does not hold for this model and should be stated as holding for the "
			      "others only.")
			weak = block[block["p_holm"] > 0.05]
			if len(weak):
				layer_list = ", ".join(str(int(v)) for v in weak["layer"])
				print(f"  Layers indistinguishable from the null (p_holm > 0.05): {layer_list}")
			continue

		after = block[block["layer"] >= inversion]
		worst = after["p_holm"].max()
		quoted = f"p < {max(worst, floor):.4f}" if worst <= floor else f"p <= {worst:.4f}"
		print(f"  Single sustained inversion at layer {inversion}. Across all {len(after)} "
		      f"post-inversion layers")
		print(f"  the observed Delta exceeds the label-permutation null at {quoted} "
		      f"(Holm-adjusted,")
		print(f"  {args.draws} permutations, attainable floor {floor:.5f}).")

	print(f"\nwrote {destination.relative_to(REPO_ROOT)}  ({len(frame)} rows)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
