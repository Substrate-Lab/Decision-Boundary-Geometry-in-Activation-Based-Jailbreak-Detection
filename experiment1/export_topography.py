#!/usr/bin/env python3
"""Export the power-diagram topography grid for each layer so R can render it.

	Reuses the multi-site power diagram from power_diagram.py: it fits the sub-sites in the
	2D LDA projection, lays a regular grid over that plane, and records at each grid node the
	geometric margin height (only where a harmful class wins) and the winning class. The grid
	is written as a tidy CSV per layer plus the prompt point cloud, which topography.R reads to
	draw a smooth 3D terrain (blue Benign ground, green Refusal and red Jailbreak peaks).

	python export_topography.py
	python export_topography.py --layers 24 30 34 --cells 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from power_diagram import (
	CLASS_NAMES,
	class_min_distance,
	fit_subsites,
	load_labels,
	power_distances,
	raw_pca_space,
	resolve_layers,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
EXPORT_DIR = Path(__file__).resolve().parent / "data" / "topography"


# Build the regular margin grid and winning class for one layer in LDA space.
def build_grid(layer: int, labels: np.ndarray, data_dir: Path, pca_dims: int, sites_per_class: int, cells: int):
	points = raw_pca_space(layer, data_dir, pca_dims)
	points2d = LinearDiscriminantAnalysis(n_components=2).fit_transform(points, labels)
	means, weights, site_classes = fit_subsites(points2d, labels, sites_per_class)

	span = np.ptp(points2d, axis=0)
	lo = points2d.min(axis=0) - 0.06 * span
	hi = points2d.max(axis=0) + 0.06 * span
	axis_x = np.linspace(lo[0], hi[0], cells)
	axis_y = np.linspace(lo[1], hi[1], cells)
	grid_x, grid_y = np.meshgrid(axis_x, axis_y)
	grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

	distances = power_distances(grid, means, weights)
	nearest_class = site_classes[distances.argmin(axis=1)]
	per_class = np.stack([class_min_distance(distances, site_classes, name) for name in CLASS_NAMES], axis=1)
	sorted_distance = np.sort(per_class, axis=1)
	margin = sorted_distance[:, 1] - sorted_distance[:, 0]
	is_harmful = np.isin(nearest_class, ["Refusal", "Jailbreak"])
	elevation = np.where(is_harmful, margin, 0.0)
	positive = elevation[elevation > 0]
	ceiling = np.percentile(positive, 90) if positive.size else 0.0
	heights = np.clip(elevation / ceiling, 0.0, 1.0) if ceiling > 0 else np.zeros_like(elevation)
	winner = np.where(is_harmful, nearest_class, "Benign")

	grid_frame = pd.DataFrame({"gx": grid[:, 0], "gy": grid[:, 1], "height": heights, "winner": winner})
	point_frame = pd.DataFrame({"x": points2d[:, 0], "y": points2d[:, 1], "label": labels})
	return grid_frame, point_frame


# Parse arguments and write the grid and point CSVs for each requested layer.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--sites-per-class", type=int, default=12)
	parser.add_argument("--cells", type=int, default=60)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	layers = resolve_layers(args.data_dir, args.layers)
	args.export_dir.mkdir(parents=True, exist_ok=True)
	for layer in layers:
		print(f"exporting layer {layer}", file=sys.stderr)
		grid_frame, point_frame = build_grid(layer, labels, args.data_dir, args.pca_dims, args.sites_per_class, args.cells)
		grid_frame.to_csv(args.export_dir / f"grid_layer{layer}.csv", index=False)
		point_frame.to_csv(args.export_dir / f"points_layer{layer}.csv", index=False)


if __name__ == "__main__":
	main()
