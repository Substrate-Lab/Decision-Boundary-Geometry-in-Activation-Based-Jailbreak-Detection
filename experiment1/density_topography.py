#!/usr/bin/env python3
"""3D topographic density map of layer activations.

	Projects a layer's activations to 2D (LDA, class-aware axes), estimates the point
	density with a Gaussian KDE, and renders it as a 3D surface: elevation = density, shaded
	with a topographic colormap (blue lows -> green -> orange/red peaks). Class cores appear
	as mountains; boundaries are valleys. Projected contour lines give the topographic look.

	python density_topography.py
	python density_topography.py --layers 24 30 36 --grid 140
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DEFAULT_LAYERS = [16, 24, 30, 36]
PCA_PREDIMS = 20
GRID = 130

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# Load labels.csv in row_index order.
def load_labels(data_dir: Path) -> np.ndarray:
	frame = pd.read_csv(data_dir / "labels.csv")
	if "row_index" in frame.columns:
		frame = frame.sort_values("row_index").reset_index(drop=True)
	return frame["class_label"].to_numpy()


# Project a layer's activations to 2D class-discriminant axes (PCA then LDA).
def project_2d(layer: int, data_dir: Path, labels: np.ndarray) -> np.ndarray:
	activations = np.load(data_dir / f"activations_layer{layer}.npy")
	dims = min(PCA_PREDIMS, activations.shape[1], activations.shape[0])
	reduced = PCA(n_components=dims, random_state=0).fit_transform(activations)
	return LinearDiscriminantAnalysis(n_components=2).fit_transform(reduced, labels)


# Estimate the KDE density of the 2D points on a grid.
def density_grid(points2d: np.ndarray, grid: int):
	pad = 0.15 * (points2d.max(axis=0) - points2d.min(axis=0) + 1e-6)
	lo = points2d.min(axis=0) - pad
	hi = points2d.max(axis=0) + pad
	grid_x, grid_y = np.meshgrid(np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid))
	kde = gaussian_kde(points2d.T)
	density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
	return grid_x, grid_y, density


# Render the density as a 3D topographic surface with projected contours.
def plot_topography(layer: int, points2d: np.ndarray, grid: int, out_path: Path) -> None:
	grid_x, grid_y, density = density_grid(points2d, grid)
	fig = plt.figure(figsize=(8.5, 7.0))
	ax = fig.add_subplot(111, projection="3d")
	ax.plot_surface(grid_x, grid_y, density, cmap="turbo", linewidth=0, antialiased=True, alpha=0.95)
	ax.contour(grid_x, grid_y, density, zdir="z", offset=0.0, levels=14, cmap="turbo", linewidths=0.6)
	ax.set_xlabel("LDA-1")
	ax.set_ylabel("LDA-2")
	ax.set_zlabel("density")
	ax.set_zlim(0.0, density.max() * 1.05)
	ax.set_title(f"Layer {layer}: activation density topography")
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments and render a topographic density map for each requested layer.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
	parser.add_argument("--grid", type=int, default=GRID)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	for layer in args.layers:
		print(f"rendering layer {layer}", file=sys.stderr)
		plot_topography(layer, project_2d(layer, args.data_dir, labels), args.grid, args.figures_dir / f"density_topography_layer{layer}.png")


if __name__ == "__main__":
	main()
