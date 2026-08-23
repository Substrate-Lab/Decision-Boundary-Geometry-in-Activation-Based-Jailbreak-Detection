#!/usr/bin/env python3
"""Unwrap per-layer activation spheres into nested polar coordinates via PNS.

	Step 2 of Experiment 1. Reads step-1 activations, reduces each layer with PCA,
	projects onto the unit sphere, fits Principal Nested Spheres, and emits a
	Euclidean PNS-score representation plus a raw-vs-polar 3D sanity figure.

	python step2_pns_unwrap.py
	python step2_pns_unwrap.py --layers 1 8 16 --pca-dims 8

	Principal Nested Spheres follows Jung, Dryden & Marron (2012), "Analysis of
	principal nested spheres", Biometrika 99(3):551-568: fit a sequence of nested
	subspheres, each an axis v and radius angle r, record signed residual angles,
	then project and rescale down one dimension until the base circle S^1.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sklearn.decomposition import PCA

LAYERS = [1, 8, 16, 24, 32]
PCA_DIMS = 10
SEED = 0

CLASS_COLORS = {
	"Refusal": "#2f6b4f",
	"Jailbreak": "#8c2f2f",
	"Benign": "#8a8a80",
}


# Fitted PNS transform: PCA basis plus the ordered list of nested-subsphere levels.
@dataclass
class PNSModel:
	pca_mean: np.ndarray
	pca_components: np.ndarray
	levels: list = field(default_factory=list)
	base_mean_angle: float = 0.0


# Full-dimensional PNS on ~4096 dims with a few thousand points is ill-posed, so reduce to a modest sphere with PCA first.
def reduce_pca(activations, n_components, seed):
	pca = PCA(n_components=n_components, random_state=seed)
	scores = pca.fit_transform(activations)
	return scores, pca.mean_.astype(float), pca.components_.astype(float)


# L2-normalize each row onto the unit sphere, replacing zero-norm rows with a fixed pole.
def to_unit_sphere(points):
	norms = np.linalg.norm(points, axis=1)
	zero = norms <= 1e-12
	if np.any(zero):
		print(f"  guarded {int(zero.sum())} zero-norm rows", file=sys.stderr)
	safe = np.where(zero, 1.0, norms)
	unit = points / safe[:, None]
	if np.any(zero):
		unit[zero] = 0.0
		unit[zero, 0] = 1.0
	return unit


# Geodesic angle on the unit sphere between each row of X and the axis v.
def geodesic_angle(points, axis):
	dots = np.clip(points @ axis, -1.0, 1.0)
	return np.arccos(dots)


# Build an m x m rotation R with R v = north pole (last basis vector).
def rotation_to_north_pole(axis):
	m = axis.shape[0]
	e = np.zeros(m)
	e[-1] = 1.0
	v = axis / max(np.linalg.norm(axis), 1e-12)
	c = float(v[-1])
	if c > 1.0 - 1e-12:
		return np.eye(m)
	if c < -1.0 + 1e-12:
		R = np.eye(m)
		R[0, 0] = -1.0
		R[-1, -1] = -1.0
		return R
	u2 = e - c * v
	u2 = u2 / np.linalg.norm(u2)
	s = float(np.sqrt(max(0.0, 1.0 - c * c)))
	R = (
		np.eye(m)
		+ s * (np.outer(u2, v) - np.outer(v, u2))
		+ (c - 1.0) * (np.outer(v, v) + np.outer(u2, u2))
	)
	return R


# Fit one subsphere (axis v, radius angle r) minimizing summed squared residual angles.
def fit_subsphere(points):
	m = points.shape[1]
	mean_vec = points.mean(axis=0)
	if np.linalg.norm(mean_vec) > 1e-12:
		v0 = mean_vec / np.linalg.norm(mean_vec)
	else:
		v0 = PCA(n_components=1).fit(points).components_[0]
		v0 = v0 / np.linalg.norm(v0)
	r0 = float(np.clip(geodesic_angle(points, v0).mean(), 1e-3, np.pi / 2.0))

	def objective(params):
		a = params[:m]
		r = params[m]
		norm = np.linalg.norm(a)
		if norm <= 1e-12:
			return 1e12
		v = a / norm
		ang = geodesic_angle(points, v)
		resid = ang - r
		return float(np.dot(resid, resid))

	x0 = np.concatenate([v0, [r0]])
	bounds = [(None, None)] * m + [(1e-4, np.pi / 2.0)]
	result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
	a = result.x[:m]
	r = float(result.x[m])
	v = a / max(np.linalg.norm(a), 1e-12)
	return v, r


# Rotate the axis to the pole, drop the pole coordinate, and renormalize onto the lower unit sphere.
def project_to_subsphere(points, rotation):
	rotated = points @ rotation.T
	reduced = rotated[:, :-1]
	return to_unit_sphere(reduced)


# Circular Fréchet mean angle of a set of angles on S^1.
def circular_mean(angles):
	return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


# Fit nested subspheres from S^{k-1} down to S^1, returning the model and stacked PNS scores.
def fit_pns(sphere_points):
	current = sphere_points
	levels = []
	residuals = []
	while current.shape[1] > 2:
		v, r = fit_subsphere(current)
		residuals.append(geodesic_angle(current, v) - r)
		rotation = rotation_to_north_pole(v)
		levels.append({"axis": v, "radius": r, "rotation": rotation})
		print(f"  fitted subsphere at dim {current.shape[1]} (r={r:.4f})", file=sys.stderr)
		current = project_to_subsphere(current, rotation)
	theta = np.arctan2(current[:, 1], current[:, 0])
	base_mean = circular_mean(theta)
	coord = np.mod(theta - base_mean + np.pi, 2.0 * np.pi) - np.pi
	residuals.append(coord)
	scores = np.column_stack(residuals)
	model = PNSModel(pca_mean=None, pca_components=None, levels=levels, base_mean_angle=base_mean)
	return model, scores


# Push raw activations (or sphere points) through the fitted nested spheres to reproduce PNS scores.
def transform_pns(model, points):
	if model.pca_mean is not None and model.pca_components is not None:
		scores = (points - model.pca_mean) @ model.pca_components.T
		current = to_unit_sphere(scores)
	else:
		current = to_unit_sphere(points)
	residuals = []
	for level in model.levels:
		residuals.append(geodesic_angle(current, level["axis"]) - level["radius"])
		current = project_to_subsphere(current, level["rotation"])
	theta = np.arctan2(current[:, 1], current[:, 0])
	coord = np.mod(theta - model.base_mean_angle + np.pi, 2.0 * np.pi) - np.pi
	residuals.append(coord)
	return np.column_stack(residuals)


# Save the fitted PNSModel arrays into a single npz.
def save_model(path, model):
	arrays = {
		"pca_mean": model.pca_mean,
		"pca_components": model.pca_components,
		"radii": np.array([level["radius"] for level in model.levels], dtype=float),
		"n_levels": np.array(len(model.levels)),
		"base_mean_angle": np.array(model.base_mean_angle),
	}
	for index, level in enumerate(model.levels):
		arrays[f"axis_{index}"] = level["axis"]
		arrays[f"rotation_{index}"] = level["rotation"]
	np.savez(path, **arrays)


# Pad a coordinate array with zero columns so it has at least three dimensions for 3D scatter.
def pad_to_three(values):
	if values.shape[1] >= 3:
		return values[:, :3]
	padded = np.zeros((values.shape[0], 3))
	padded[:, : values.shape[1]] = values
	return padded


# Draw side-by-side 3D scatter of raw PCA geometry versus PNS polar geometry, colored by class.
def plot_raw_vs_polar(pca_scores, pns_scores, labels, layer, out_path):
	raw = pad_to_three(pca_scores)
	polar = pad_to_three(pns_scores)
	fig = plt.figure(figsize=(12.0, 5.5))
	ax_raw = fig.add_subplot(1, 2, 1, projection="3d")
	ax_polar = fig.add_subplot(1, 2, 2, projection="3d")
	classes = list(dict.fromkeys(labels))
	for cls in classes:
		mask = np.array([lab == cls for lab in labels])
		color = CLASS_COLORS.get(cls, "#555555")
		ax_raw.scatter(raw[mask, 0], raw[mask, 1], raw[mask, 2], s=6, alpha=0.55, color=color, label=str(cls))
		ax_polar.scatter(polar[mask, 0], polar[mask, 1], polar[mask, 2], s=6, alpha=0.55, color=color, label=str(cls))
	ax_raw.set_title("Raw geometry (PCA dims 1-3)")
	ax_polar.set_title("PNS polar geometry (score dims 1-3)")
	ax_raw.legend(loc="upper right", fontsize=8)
	fig.suptitle(f"Layer {layer}: raw vs PNS-unwrapped activations")
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"  wrote {out_path}", file=sys.stderr)


# Load labels.csv and return the class labels in row_index order.
def load_labels(data_dir):
	frame = pd.read_csv(data_dir / "labels.csv")
	if "row_index" in frame.columns:
		frame = frame.sort_values("row_index").reset_index(drop=True)
	return frame["class_label"].tolist()


# Run the full PNS pipeline for one layer and write scores, model, and figure.
def process_layer(layer, labels, pca_dims, data_dir, figures_dir):
	activations = np.load(data_dir / f"activations_layer{layer}.npy")
	print(f"layer {layer}: activations {activations.shape}", file=sys.stderr)
	pca_scores, pca_mean, pca_components = reduce_pca(activations, pca_dims, SEED)
	sphere = to_unit_sphere(pca_scores)
	model, scores = fit_pns(sphere)
	model.pca_mean = pca_mean
	model.pca_components = pca_components
	np.save(data_dir / f"pns_layer{layer}.npy", scores)
	save_model(data_dir / f"pns_model_layer{layer}.npz", model)
	print(f"layer {layer}: PNS scores {scores.shape}", file=sys.stderr)
	figures_dir.mkdir(parents=True, exist_ok=True)
	plot_raw_vs_polar(pca_scores, scores, labels, layer, figures_dir / f"raw_vs_polar_layer{layer}.png")


# Parse arguments and process each requested layer.
def main():
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
	parser.add_argument("--pca-dims", type=int, default=PCA_DIMS)
	parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
	parser.add_argument("--figures-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	for layer in args.layers:
		process_layer(layer, labels, args.pca_dims, args.data_dir, args.figures_dir)
	print("done", file=sys.stderr)


if __name__ == "__main__":
	main()
