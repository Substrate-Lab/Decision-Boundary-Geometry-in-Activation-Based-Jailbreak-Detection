#!/usr/bin/env python3
"""Build the scalar-weight power diagram per layer and render a 2D power map.

	One site per class: centroid mu_c and weight w_c = trace(Sigma_c). A point is assigned
	to the class minimising the power distance pi_c(x) = ||x - mu_c||^2 - w_c, and the
	geometric margin is the gap between the two smallest power distances. The diagram is fit
	in raw activation space (PCA-reduced), where the class covariance mismatch actually lives
	(PNS equalises it). For each layer we report assignment accuracy and mean margin, and save
	a power map: a 2D LDA projection with the power cells, sites, weight rings, and points.

	python power_diagram.py
	python power_diagram.py --layers 24 26 28 30 32 34 --pca-dims 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

CLASS_NAMES = ["Refusal", "Jailbreak", "Benign"]
CLASS_COLORS = {
	"Refusal": "#1ecb96",
	"Jailbreak": "#e5484d",
	"Benign": "#f5c518",
}
GRID_RESOLUTION = 300

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# Load labels.csv in row_index order.
def load_labels(data_dir: Path) -> np.ndarray:
	frame = pd.read_csv(data_dir / "labels.csv")
	if "row_index" in frame.columns:
		frame = frame.sort_values("row_index").reset_index(drop=True)
	return frame["class_label"].to_numpy()


# Use the requested layers, or fall back to the layers step1 recorded in collection_meta.json.
def resolve_layers(data_dir: Path, requested):
	if requested:
		return requested
	meta_path = data_dir / "collection_meta.json"
	if meta_path.exists():
		with meta_path.open(encoding="utf-8") as handle:
			return json.load(handle).get("layers", [])
	return []


# Reduce a layer's raw activations to a fixed number of PCA dimensions.
def raw_pca_space(layer: int, data_dir: Path, pca_dims: int) -> np.ndarray:
	activations = np.load(data_dir / f"activations_layer{layer}.npy")
	dims = min(pca_dims, activations.shape[1], activations.shape[0])
	return PCA(n_components=dims, random_state=0).fit_transform(activations)


# Fit one power-diagram site per class: centroid and scalar weight = trace of the covariance.
def fit_sites(points: np.ndarray, labels: np.ndarray) -> dict[str, tuple]:
	sites: dict[str, tuple] = {}
	for name in CLASS_NAMES:
		mask = labels == name
		if mask.sum() < 2:
			continue
		mean = points[mask].mean(axis=0)
		covariance = np.cov(points[mask], rowvar=False)
		sites[name] = (mean, float(np.trace(covariance)))
	return sites


# Power distances of every point to every site: ||x - mu_c||^2 - w_c.
def power_distances(points: np.ndarray, sites: dict[str, tuple]) -> tuple[list[str], np.ndarray]:
	names = list(sites)
	distances = np.empty((points.shape[0], len(names)))
	for column, name in enumerate(names):
		mean, weight = sites[name]
		distances[:, column] = np.sum((points - mean) ** 2, axis=1) - weight
	return names, distances


# Assign each point to its minimum-power-distance class and compute its geometric margin.
def assign_and_margin(names: list[str], distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	order = np.argsort(distances, axis=1)
	predicted = np.array([names[i] for i in order[:, 0]])
	smallest = np.take_along_axis(distances, order[:, :1], axis=1)[:, 0]
	second = np.take_along_axis(distances, order[:, 1:2], axis=1)[:, 0]
	return predicted, second - smallest


# Render the 2D power map: cells by power assignment, sites, weight rings, and points.
def plot_power_map(points2d: np.ndarray, labels: np.ndarray, layer: int, accuracy: float, out_path: Path) -> None:
	sites = fit_sites(points2d, labels)
	names = list(sites)
	pad = 0.1 * (points2d.max(axis=0) - points2d.min(axis=0) + 1e-6)
	lo = points2d.min(axis=0) - pad
	hi = points2d.max(axis=0) + pad
	grid_x, grid_y = np.meshgrid(
		np.linspace(lo[0], hi[0], GRID_RESOLUTION),
		np.linspace(lo[1], hi[1], GRID_RESOLUTION),
	)
	grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
	_, grid_distances = power_distances(grid, sites)
	grid_assignment = grid_distances.argmin(axis=1).reshape(grid_x.shape)

	fig, ax = plt.subplots(figsize=(7.5, 6.5))
	cell_colors = np.array([_hex_to_rgb(CLASS_COLORS[name]) for name in names])
	ax.imshow(
		cell_colors[grid_assignment],
		origin="lower",
		extent=(lo[0], hi[0], lo[1], hi[1]),
		aspect="auto",
		alpha=0.25,
	)
	for name in CLASS_NAMES:
		mask = labels == name
		if mask.any():
			ax.scatter(points2d[mask, 0], points2d[mask, 1], s=8, alpha=0.7, color=CLASS_COLORS[name], label=name)
	for name in names:
		mean, weight = sites[name]
		ax.scatter([mean[0]], [mean[1]], marker="X", s=160, color=CLASS_COLORS[name], edgecolor="black", linewidth=1.2, zorder=5)
		if weight > 0:
			ring = plt.Circle((mean[0], mean[1]), np.sqrt(weight), color=CLASS_COLORS[name], fill=False, linewidth=1.5, alpha=0.9)
			ax.add_patch(ring)
	ax.set_xlabel("LDA-1")
	ax.set_ylabel("LDA-2")
	ax.set_title(f"Power map, layer {layer} (power-distance accuracy {accuracy:.3f}); ring = sqrt(weight)")
	ax.legend(loc="upper right", fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Convert a hex color to an RGB triple in 0..1.
def _hex_to_rgb(value: str) -> tuple[float, float, float]:
	value = value.lstrip("#")
	return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# Fit and evaluate the power diagram for one layer, and save its 2D power map.
def process_layer(layer: int, labels: np.ndarray, data_dir: Path, figures_dir: Path, pca_dims: int) -> dict:
	print(f"processing layer {layer}", file=sys.stderr)
	points = raw_pca_space(layer, data_dir, pca_dims)
	sites = fit_sites(points, labels)
	names, distances = power_distances(points, sites)
	predicted, margin = assign_and_margin(names, distances)
	accuracy = float((predicted == labels).mean())

	points2d = LinearDiscriminantAnalysis(n_components=2).fit_transform(points, labels)
	plot_power_map(points2d, labels, layer, accuracy, figures_dir / f"power_map_layer{layer}.png")

	row = {"layer": layer, "power_accuracy": accuracy, "mean_margin": float(margin.mean())}
	for name in names:
		row[f"weight_{name}"] = sites[name][1]
	return row


# Parse arguments, build the power diagram per layer, save the CSV and power maps.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	layers = resolve_layers(args.data_dir, args.layers)
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	rows = [process_layer(layer, labels, args.data_dir, args.figures_dir, args.pca_dims) for layer in layers]

	result = pd.DataFrame(rows)
	csv_path = args.data_dir / "power_diagram.csv"
	result.to_csv(csv_path, index=False)
	print(f"wrote {csv_path}", file=sys.stderr)
	print(result.to_string(index=False))


if __name__ == "__main__":
	main()
