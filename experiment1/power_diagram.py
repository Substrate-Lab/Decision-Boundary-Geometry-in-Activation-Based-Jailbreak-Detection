#!/usr/bin/env python3
"""Build a multi-site power diagram per layer and render a 2D power map.

	Each class is split into many weighted sub-sites (k-means sub-centroids), each with a
	scalar weight w = trace(Sigma) of its sub-cluster. A point is assigned to the class of the
	sub-site minimising the power distance pi(x) = ||x - mu||^2 - w. The result is a
	Laguerre-Voronoi tessellation: many small convex cells that group into per-class regions,
	so the boundary between classes can curve (piecewise-linear) instead of being one flat
	hyperplane. The diagram is fit in raw activation space (PCA-reduced), where the class
	covariance mismatch actually lives. Each layer gets an accuracy, a mean margin, and a power
	map (2D LDA projection with the sub-cells drawn).

	python power_diagram.py
	python power_diagram.py --sites-per-class 12 --layers 24 30 34
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from common import CLASS_COLORS, CLASS_NAMES, load_labels, resolve_layers

SITES_PER_CLASS = 12
GRID_RESOLUTION = 320

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# Reduce a layer's raw activations to a fixed number of PCA dimensions.
def raw_pca_space(layer: int, data_dir: Path, pca_dims: int) -> np.ndarray:
	activations = np.load(data_dir / f"activations_layer{layer}.npy")
	dims = min(pca_dims, activations.shape[1], activations.shape[0])
	return PCA(n_components=dims, random_state=0).fit_transform(activations)


# Split each class into k-means sub-sites, each with a scalar weight = trace of its sub-covariance.
def fit_subsites(points: np.ndarray, labels: np.ndarray, sites_per_class: int):
	means: list[np.ndarray] = []
	weights: list[float] = []
	site_classes: list[str] = []
	for name in CLASS_NAMES:
		mask = labels == name
		class_points = points[mask]
		if len(class_points) < 2:
			continue
		clusters = min(sites_per_class, len(class_points))
		assignment = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit_predict(class_points)
		for cluster in range(clusters):
			members = class_points[assignment == cluster]
			if len(members) == 0:
				continue
			mean = members.mean(axis=0)
			weight = float(np.trace(np.cov(members, rowvar=False))) if len(members) >= 2 else 0.0
			means.append(mean)
			weights.append(weight)
			site_classes.append(name)
	return np.array(means), np.array(weights), np.array(site_classes)


# Power distances of every point to every sub-site: ||x - mu||^2 - w.
def power_distances(points: np.ndarray, means: np.ndarray, weights: np.ndarray) -> np.ndarray:
	squared = np.sum((points[:, None, :] - means[None, :, :]) ** 2, axis=2)
	return squared - weights[None, :]


# Assign each point to the class of its nearest sub-site and compute the geometric margin.
def assign_and_margin(distances: np.ndarray, site_classes: np.ndarray):
	order = np.argsort(distances, axis=1)
	predicted = site_classes[order[:, 0]]
	smallest = np.take_along_axis(distances, order[:, :1], axis=1)[:, 0]
	second = np.take_along_axis(distances, order[:, 1:2], axis=1)[:, 0]
	return predicted, second - smallest


# Convert a hex color to an RGB triple in 0..1.
def hex_to_rgb(value: str) -> tuple[float, float, float]:
	value = value.lstrip("#")
	return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# Render the 2D power map: sub-cells colored by class, tessellation edges, sub-sites, and points.
def plot_power_map(points2d: np.ndarray, labels: np.ndarray, layer: int, accuracy: float, sites_per_class: int, out_path: Path) -> None:
	means, weights, site_classes = fit_subsites(points2d, labels, sites_per_class)
	pad = 0.1 * (points2d.max(axis=0) - points2d.min(axis=0) + 1e-6)
	lo = points2d.min(axis=0) - pad
	hi = points2d.max(axis=0) + pad
	grid_x, grid_y = np.meshgrid(
		np.linspace(lo[0], hi[0], GRID_RESOLUTION),
		np.linspace(lo[1], hi[1], GRID_RESOLUTION),
	)
	grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
	grid_distances = power_distances(grid, means, weights)
	nearest_site = grid_distances.argmin(axis=1).reshape(grid_x.shape)

	class_index = {name: i for i, name in enumerate(CLASS_NAMES)}
	grid_class = np.vectorize(lambda s: class_index[site_classes[s]])(nearest_site)
	cell_colors = np.array([hex_to_rgb(CLASS_COLORS[name]) for name in CLASS_NAMES])

	edges = np.zeros_like(nearest_site, dtype=bool)
	edges[:, :-1] |= nearest_site[:, :-1] != nearest_site[:, 1:]
	edges[:-1, :] |= nearest_site[:-1, :] != nearest_site[1:, :]
	edge_layer = np.zeros((*edges.shape, 4))
	edge_layer[edges] = (0.15, 0.15, 0.15, 0.5)

	fig, ax = plt.subplots(figsize=(7.5, 6.5))
	extent = (lo[0], hi[0], lo[1], hi[1])
	ax.imshow(cell_colors[grid_class], origin="lower", extent=extent, aspect="auto", alpha=0.3)
	ax.imshow(edge_layer, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
	for name in CLASS_NAMES:
		mask = labels == name
		if mask.any():
			ax.scatter(points2d[mask, 0], points2d[mask, 1], s=7, alpha=0.65, color=CLASS_COLORS[name], label=name)
	for index, name in enumerate(site_classes):
		ax.scatter([means[index, 0]], [means[index, 1]], marker="X", s=55, color=CLASS_COLORS[name], edgecolor="black", linewidth=0.8, zorder=5)
	ax.set_xlabel("LDA-1")
	ax.set_ylabel("LDA-2")
	ax.set_title(f"Multi-site power map, layer {layer} ({sites_per_class} sites/class, accuracy {accuracy:.3f})")
	ax.legend(loc="upper right", fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Class-level geometric margin at each point: nearest other-class minus nearest own-class power distance.
def class_margin_field(distances: np.ndarray, site_classes: np.ndarray):
	present = [name for name in CLASS_NAMES if (site_classes == name).any()]
	class_min = np.stack([distances[:, site_classes == name].min(axis=1) for name in present], axis=1)
	order = np.argsort(class_min, axis=1)
	assigned = np.array(present)[order[:, 0]]
	smallest = np.take_along_axis(class_min, order[:, :1], axis=1)[:, 0]
	second = np.take_along_axis(class_min, order[:, 1:2], axis=1)[:, 0]
	return assigned, second - smallest


# Render the confidence heatmap: class color with opacity = geometric margin (dense near centroids, fading to the boundary).
def plot_confidence_map(points2d: np.ndarray, labels: np.ndarray, layer: int, sites_per_class: int, out_path: Path) -> None:
	means, weights, site_classes = fit_subsites(points2d, labels, sites_per_class)
	pad = 0.1 * (points2d.max(axis=0) - points2d.min(axis=0) + 1e-6)
	lo = points2d.min(axis=0) - pad
	hi = points2d.max(axis=0) + pad
	grid_x, grid_y = np.meshgrid(
		np.linspace(lo[0], hi[0], GRID_RESOLUTION),
		np.linspace(lo[1], hi[1], GRID_RESOLUTION),
	)
	grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
	assigned, margin = class_margin_field(power_distances(grid, means, weights), site_classes)

	ceiling = np.percentile(margin, 95)
	normalized = np.clip(margin / ceiling, 0.0, 1.0) if ceiling > 0 else np.zeros_like(margin)
	class_index = {name: i for i, name in enumerate(CLASS_NAMES)}
	rgb = np.array([hex_to_rgb(CLASS_COLORS[name]) for name in CLASS_NAMES])
	image = np.zeros((grid.shape[0], 4))
	image[:, :3] = rgb[np.array([class_index[name] for name in assigned])]
	image[:, 3] = 0.12 + 0.85 * normalized
	image = image.reshape(*grid_x.shape, 4)

	fig, ax = plt.subplots(figsize=(7.5, 6.5))
	ax.imshow(image, origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), aspect="auto", interpolation="bilinear")
	for name in CLASS_NAMES:
		mask = labels == name
		if mask.any():
			ax.scatter(points2d[mask, 0], points2d[mask, 1], s=6, alpha=0.55, color=CLASS_COLORS[name], edgecolor="none", label=name)
	ax.set_xlabel("LDA-1")
	ax.set_ylabel("LDA-2")
	ax.set_title(f"Confidence heatmap, layer {layer} (opacity = geometric margin)")
	ax.legend(loc="upper right", fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Fit and evaluate the multi-site power diagram for one layer, and save its power map and confidence maps.
def process_layer(layer: int, labels: np.ndarray, data_dir: Path, figures_dir: Path, pca_dims: int, sites_per_class: int) -> dict:
	print(f"processing layer {layer}", file=sys.stderr)
	points = raw_pca_space(layer, data_dir, pca_dims)
	means, weights, site_classes = fit_subsites(points, labels, sites_per_class)
	distances = power_distances(points, means, weights)
	predicted, margin = assign_and_margin(distances, site_classes)
	accuracy = float((predicted == labels).mean())

	points2d = LinearDiscriminantAnalysis(n_components=2).fit_transform(points, labels)
	plot_power_map(points2d, labels, layer, accuracy, sites_per_class, figures_dir / f"power_map_layer{layer}.png")
	plot_confidence_map(points2d, labels, layer, sites_per_class, figures_dir / f"power_confidence_layer{layer}.png")

	return {"layer": layer, "power_accuracy": accuracy, "mean_margin": float(margin.mean()), "n_sites": len(means)}


# Parse arguments, build the multi-site power diagram per layer, save the CSV and power maps.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--sites-per-class", type=int, default=SITES_PER_CLASS)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	layers = resolve_layers(args.data_dir, args.layers)
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	rows = [process_layer(layer, labels, args.data_dir, args.figures_dir, args.pca_dims, args.sites_per_class) for layer in layers]

	result = pd.DataFrame(rows)
	csv_path = args.data_dir / "power_diagram.csv"
	result.to_csv(csv_path, index=False)
	print(f"wrote {csv_path}", file=sys.stderr)
	print(result.to_string(index=False))


if __name__ == "__main__":
	main()
