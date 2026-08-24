#!/usr/bin/env python3
"""3D covariance-mismatch view of layer activations.

	Projects a layer's activations to 3 PCA dimensions and draws, per class, the point cloud
	plus a 1-sigma covariance ellipsoid. Classes with different-shaped or different-sized
	ellipsoids are the covariance mismatch the power diagram exploits; the tight class
	(Refusal) should show the smallest ellipsoid, the diffuse ones (Jailbreak/Benign) larger.

	python covariance_3d.py
	python covariance_3d.py --layers 24 30 34 --n-std 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import CLASS_COLORS, CLASS_NAMES, DATA_DIR, FIGURES_DIR, load_labels

DEFAULT_LAYERS = [8, 16, 24, 30, 36]


# Project a layer's raw activations to 3 PCA dimensions.
def pca_three(layer: int, data_dir: Path) -> np.ndarray:
	activations = np.load(data_dir / f"activations_layer{layer}.npy")
	return PCA(n_components=3, random_state=0).fit_transform(activations)


# Build a mesh of a covariance ellipsoid at n_std standard deviations around a mean.
def ellipsoid_surface(mean: np.ndarray, covariance: np.ndarray, n_std: float):
	phi = np.linspace(0, 2 * np.pi, 30)
	theta = np.linspace(0, np.pi, 15)
	unit = np.array([
		np.outer(np.cos(phi), np.sin(theta)),
		np.outer(np.sin(phi), np.sin(theta)),
		np.outer(np.ones_like(phi), np.cos(theta)),
	])
	values, vectors = np.linalg.eigh(covariance)
	radii = n_std * np.sqrt(np.clip(values, 0, None))
	transformed = vectors @ (radii[:, None] * unit.reshape(3, -1))
	return (transformed + mean[:, None]).reshape(3, *unit.shape[1:])


# Draw the 3D scatter and per-class covariance ellipsoids for one layer.
def plot_layer(layer: int, points: np.ndarray, labels: np.ndarray, n_std: float, out_path: Path) -> None:
	fig = plt.figure(figsize=(8.0, 7.0))
	ax = fig.add_subplot(111, projection="3d")
	for name in CLASS_NAMES:
		mask = labels == name
		if mask.sum() < 2:
			continue
		cloud = points[mask]
		ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=5, alpha=0.25, color=CLASS_COLORS[name], label=name)
		mean = cloud.mean(axis=0)
		covariance = np.cov(cloud, rowvar=False)
		x, y, z = ellipsoid_surface(mean, covariance, n_std)
		ax.plot_surface(x, y, z, color=CLASS_COLORS[name], alpha=0.18, linewidth=0)
	ax.set_xlabel("PCA-1")
	ax.set_ylabel("PCA-2")
	ax.set_zlabel("PCA-3")
	ax.set_title(f"Layer {layer}: class covariance ellipsoids ({n_std}-sigma)")
	ax.legend(loc="upper right", fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments and render a 3D covariance view for each requested layer.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
	parser.add_argument("--n-std", type=float, default=2.0)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	for layer in args.layers:
		print(f"rendering layer {layer}", file=sys.stderr)
		plot_layer(layer, pca_three(layer, args.data_dir), labels, args.n_std, args.figures_dir / f"covariance_3d_layer{layer}.png")


if __name__ == "__main__":
	main()
