"""Measure how far each method boundary sits from a GMM-fitted reference boundary.

	Step 3 of Experiment 2. For each layer a per-class Gaussian mixture is fitted as the reference
	("true") density; the reference boundary between two classes is where their log-likelihoods
	cross. For each method the same crossing is found along the line joining the two class means,
	and the RMS gap between the method crossing and the reference crossing is the boundary error.
	A global disagreement rate (method cell vs reference cell over the data) is also reported.

	Caveat: with model=qda and gmm-k=1 the method IS the reference (one Gaussian
	per class), so the error is trivially near zero and measures the formula against itself. Those
	rows are flagged degenerate rather than silently reported as a win.

	python step3_boundary_error.py
	python step3_boundary_error.py --models power_multi qda --gmm-k 2 --layers 24 30 34
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.discriminants import build_discriminant
from lib.space import DEFAULT_DATA_DIR, OUTPUT_DIR, load_labels, resolve_layers, working_space

DEFAULT_MODELS = ["power_multi", "qda", "lda", "power_single"]
SEED = 0
RESOLUTION = 400


# Per-class Gaussian mixture used as the reference density; scores are negative log-likelihoods.
class GmmReference:
	# Fit one mixture per class with k components each.
	def __init__(self, k: int):
		self.k = k

	# Fit the per-class mixtures and record the class order.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "GmmReference":
		self.classes_ = np.array([name for name in ["Refusal", "Jailbreak", "Benign"] if np.any(labels == name)])
		self.models_ = {}
		for name in self.classes_:
			rows = points[labels == name]
			components = min(self.k, max(1, len(rows) // 2))
			self.models_[name] = GaussianMixture(n_components=components, covariance_type="full", reg_covar=1e-4, random_state=SEED).fit(rows)
		return self

	# Score each point by the negative log-likelihood under each class mixture.
	def scores(self, points: np.ndarray) -> np.ndarray:
		return np.column_stack([-self.models_[name].score_samples(points) for name in self.classes_])


# Find the parameter t in [0, 1] where the preference between two score columns first flips.
def crossing(score_a: np.ndarray, score_b: np.ndarray):
	difference = score_a - score_b
	sign_change = np.where(np.diff(np.sign(difference)) != 0)[0]
	if len(sign_change) == 0:
		return None
	i = sign_change[0]
	da, db = difference[i], difference[i + 1]
	if db == da:
		return i / (len(difference) - 1)
	local = da / (da - db)
	return (i + local) / (len(difference) - 1)


# RMS boundary gap between a method and the reference along every class-pair mean line.
def boundary_error(points, labels, method, reference, classes):
	means = {name: points[labels == name].mean(axis=0) for name in classes}
	class_column = {name: i for i, name in enumerate(classes)}
	gaps = []
	for a, b in combinations(classes, 2):
		line = np.linspace(0.0, 1.0, RESOLUTION)[:, None] * (means[b] - means[a])[None, :] + means[a][None, :]
		method_scores = method.scores(line)
		reference_scores = reference.scores(line)
		t_method = crossing(method_scores[:, class_column[a]], method_scores[:, class_column[b]])
		t_reference = crossing(reference_scores[:, class_column[a]], reference_scores[:, class_column[b]])
		if t_method is None or t_reference is None:
			continue
		distance = float(np.linalg.norm(means[b] - means[a]))
		gaps.append((t_method - t_reference) * distance)
	if not gaps:
		return float("nan")
	return float(np.sqrt(np.mean(np.square(gaps))))


# Fraction of data points where the method cell differs from the reference cell.
def disagreement_rate(points, method, reference):
	method_cell = method.classes_[method.scores(points).argmin(axis=1)]
	reference_cell = reference.classes_[reference.scores(points).argmin(axis=1)]
	return float(np.mean(method_cell != reference_cell))


# Parse arguments, compare every method to the GMM reference per layer, and write the summary.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
	parser.add_argument("--space", choices=["rawpca", "pns"], default="rawpca")
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--sites-per-class", type=int, default=12)
	parser.add_argument("--gmm-k", type=int, default=2)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	rows = []
	for layer in resolve_layers(args.data_dir, args.layers):
		points = working_space(layer, args.data_dir, args.space, args.pca_dims)
		print(f"layer {layer}: points {points.shape}", file=sys.stderr)
		reference = GmmReference(args.gmm_k).fit(points, labels)
		for model_name in args.models:
			method = build_discriminant(model_name, sites_per_class=args.sites_per_class).fit(points, labels)
			degenerate = model_name == "qda" and args.gmm_k == 1
			rms = boundary_error(points, labels, method, reference, reference.classes_)
			disagreement = disagreement_rate(points, method, reference)
			rows.append({
				"space": args.space,
				"layer": layer,
				"model": model_name,
				"gmm_k": args.gmm_k,
				"rms_boundary_error": rms,
				"disagreement_rate": disagreement,
				"degenerate": int(degenerate),
			})
			flag = " (degenerate: method == reference)" if degenerate else ""
			print(f"  {model_name}: rms {rms:.4g} disagreement {disagreement:.4f}{flag}", file=sys.stderr)

	summary = pd.DataFrame(rows)
	out_path = args.output_dir / f"boundary_error_{args.space}.csv"
	summary.to_csv(out_path, index=False)
	print(f"wrote {out_path} ({len(summary)} rows)", file=sys.stderr)
	print(summary.to_string(index=False))


if __name__ == "__main__":
	main()
