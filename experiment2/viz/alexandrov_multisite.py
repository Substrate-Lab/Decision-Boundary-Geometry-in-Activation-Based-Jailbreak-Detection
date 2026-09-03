#!/usr/bin/env python3
"""The multi-site power diagram Experiment 1 actually reported, with solved weights.

	Experiment 1's power_diagram.csv records layer 30 with n_sites = 36 -- twelve k-means
	sub-sites per class, the `power_multi` arm. That is the geometry drawn here: each class owns
	twelve weighted sites, so its territory is a union of convex power cells and the class-level
	boundary is piecewise linear, not a single hyperplane.

	Sites are fit on a train split and the diagram is scored on a held-out split, which is what
	makes the comparison non-trivial: k-means centroids are by construction the Voronoi sites of
	their own training points, so on train there is nothing to correct. On held-out data the
	tight class over-collects and the solved weights are what put the mass back.

	The points are synthetic, calibrated to Experiment 1's measured layer-30 norms (see
	lib/exp1_stats.py). The real activations are gitignored and not in the repo.

	Usage:
		.venv/bin/python experiment2/viz/alexandrov_multisite.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from experiment2.lib.alexandrov import cell_assignment, solve_weights, squared_distances
from experiment2.lib.exp1_stats import (
	CLASS_NAMES, calibrated_cloud, covariance_similarity, load_reported_run, norm_stats,
)

CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
PALE = ["#dce9f8", "#fbe1d6", "#d5efe6"]
INK = "#0b0b0b"
MUTED = "#52514e"
SEED = 0


# Twelve k-means sub-sites per class, matching Experiment 1's 36-site run.
def fit_sites(points, labels, per_class):
	sites, owners, masses = [], [], []
	for index in range(len(CLASS_NAMES)):
		member = points[labels == index]
		# A class can hold fewer training points than the requested sub-site count; k-means
		# raises rather than degrading, so clamp the way discriminants.py already does.
		clusters = min(per_class, len(member))
		if clusters < 1:
			continue
		assignment = KMeans(n_clusters=clusters, n_init=10, random_state=SEED).fit_predict(member)
		for cluster in range(clusters):
			rows = member[assignment == cluster]
			if len(rows) == 0:
				continue
			sites.append(rows.mean(axis=0))
			owners.append(index)
			masses.append(len(rows))
	masses = np.array(masses, dtype=float)
	return np.array(sites), np.array(owners), masses / masses.sum()


# Site index and class index for every node of a display grid, computed in row blocks.
def label_grid(points, sites, owners, weights, resolution):
	pad_x = 0.06 * np.ptp(points[:, 0])
	pad_y = 0.06 * np.ptp(points[:, 1])
	x_axis = np.linspace(points[:, 0].min() - pad_x, points[:, 0].max() + pad_x, resolution)
	y_axis = np.linspace(points[:, 1].min() - pad_y, points[:, 1].max() + pad_y, resolution)
	site_grid = np.empty((resolution, resolution), dtype=np.int16)
	for start in range(0, resolution, 64):
		stop = min(start + 64, resolution)
		mesh_x, mesh_y = np.meshgrid(x_axis, y_axis[start:stop])
		nodes = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
		power = squared_distances(nodes, sites) - weights[None, :]
		site_grid[start:stop] = np.argmin(power, axis=1).reshape(stop - start, resolution)
	return x_axis, y_axis, site_grid, owners[site_grid]


# Boolean mask of cells whose right or lower neighbour carries a different index.
def edge_mask(field):
	edges = np.zeros(field.shape, dtype=bool)
	edges[:, :-1] |= field[:, :-1] != field[:, 1:]
	edges[:-1, :] |= field[:-1, :] != field[1:, :]
	return edges


# Thicken a mask by one pixel in each direction, so class boundaries read heavier.
def thicken(mask):
	grown = mask.copy()
	grown[:, 1:] |= mask[:, :-1]
	grown[1:, :] |= mask[:-1, :]
	return grown


# Draw one panel: pale class territory, thin sub-cell edges, heavy class boundary, points.
def draw_panel(axis, points, labels, sites, owners, weights, resolution, title):
	x_axis, y_axis, site_grid, class_grid = label_grid(points, sites, owners, weights, resolution)
	extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
	axis.imshow(class_grid, origin="lower", extent=extent, aspect="equal",
	            cmap=ListedColormap(PALE), vmin=0, vmax=2, interpolation="nearest")

	sub_edges = edge_mask(site_grid) & ~edge_mask(class_grid)
	overlay = np.zeros(site_grid.shape + (4,))
	overlay[sub_edges] = (0.42, 0.42, 0.42, 0.55)
	axis.imshow(overlay, origin="lower", extent=extent, aspect="equal", interpolation="nearest")

	class_edges = thicken(edge_mask(class_grid))
	overlay = np.zeros(site_grid.shape + (4,))
	overlay[class_edges] = (0.04, 0.04, 0.04, 1.0)
	axis.imshow(overlay, origin="lower", extent=extent, aspect="equal", interpolation="nearest")

	for index, name in enumerate(CLASS_NAMES):
		member = points[labels == index]
		axis.scatter(member[:, 0], member[:, 1], s=4, c=CLASS_COLORS[index],
		             linewidths=0, alpha=0.75, label=name, zorder=3)
	axis.scatter(sites[:, 0], sites[:, 1], s=18, c="black", marker="o",
	             edgecolors="white", linewidths=0.7, zorder=4)
	axis.set_title(title, fontsize=10.5)
	axis.set_xticks([])
	axis.set_yticks([])
	for spine in axis.spines.values():
		spine.set_edgecolor("#bbbbbb")


# Rigid centre-and-rotate onto the cloud's principal axes, for framing only. Power diagrams
# are equivariant under rigid motion, so this moves the picture without changing the geometry.
def display_frame(points, sites):
	centre = points.mean(axis=0)
	centred = points - centre
	_, _, components = np.linalg.svd(centred, full_matrices=False)
	rotation = components.T
	return centred @ rotation, (sites - centre) @ rotation


# Build the cloud at a given angular separation and score the plain 36-site diagram on held-out.
def build_split(separation, layer, model, n_per_class, radial_share, per_class):
	points, labels = calibrated_cloud(
		layer, model, n_per_class=n_per_class, dims=2,
		radial_share=radial_share, separation=separation, seed=SEED,
	)
	rng = np.random.default_rng(SEED)
	shuffled = rng.permutation(len(points))
	split = len(points) // 2
	train, test = shuffled[:split], shuffled[split:]
	sites, owners, targets = fit_sites(points[train], labels[train], per_class)
	squared = squared_distances(points[test], sites)
	assigned = np.argmin(squared, axis=1)
	accuracy = float(np.mean(owners[assigned] == labels[test]))
	return accuracy, (points[test], labels[test], sites, owners, targets)


# Bisect the angular separation until the plain diagram reproduces Experiment 1's accuracy.
def calibrate_separation(target_accuracy, layer, model, n_per_class, radial_share, per_class,
                         low=0.005, high=1.0, iterations=24):
	for _ in range(iterations):
		middle = 0.5 * (low + high)
		accuracy, _ = build_split(middle, layer, model, n_per_class, radial_share, per_class)
		if accuracy > target_accuracy:
			high = middle
		else:
			low = middle
	separation = 0.5 * (low + high)
	accuracy, bundle = build_split(separation, layer, model, n_per_class, radial_share, per_class)
	return separation, accuracy, bundle


# Build the calibrated cloud, fit sites on train, solve weights, and write the figure.
def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", type=str, default=None,
	                    help="Experiment 1 model to read; defaults to the only collected run.")
	parser.add_argument("--per-class-sites", type=int, default=None)
	parser.add_argument("--n-per-class", type=int, default=500)
	parser.add_argument("--grid", type=int, default=560)
	parser.add_argument("--radial-share", type=float, default=0.35)
	parser.add_argument("--out", type=str, default="experiment2/figures/alexandrov_multisite.png")
	args = parser.parse_args()

	run = load_reported_run(args.model)
	per_class = args.per_class_sites or run["n_sites"] // len(CLASS_NAMES)
	similarity = covariance_similarity(run["layer"], args.model)

	# Match Experiment 1's reported operating point rather than inventing a separation.
	separation, achieved, bundle = calibrate_separation(
		run["power_accuracy"], run["layer"], args.model, args.n_per_class,
		args.radial_share, per_class,
	)
	test_points, test_labels, sites, owners, targets = bundle
	print(f"calibrated angular separation {separation:.4f} -> plain-diagram accuracy "
	      f"{achieved:.4f} (experiment 1 reported {run['power_accuracy']:.4f})")
	test_masses = np.full(len(test_points), 1.0 / len(test_points))

	# Targets are the class priors, spread across each class's sites by the shape
	# k-means found on train. Per-site targets therefore sum to the prior within every class.
	prior = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
	targets = targets.astype(float).copy()
	for index in range(len(CLASS_NAMES)):
		owned = owners == index
		targets[owned] = targets[owned] / targets[owned].sum() * prior[index]

	squared = squared_distances(test_points, sites)
	zero = np.zeros(len(sites))
	voronoi_sites, _ = cell_assignment(squared, zero, test_masses)
	weights, info = solve_weights(test_points, sites, targets, test_masses)
	solved_sites = info["assigned"]

	def class_mass(site_index):
		return np.array([test_masses[owners[site_index] == c].sum() for c in range(3)])

	voronoi_mass = class_mass(voronoi_sites)
	solved_mass = class_mass(solved_sites)
	voronoi_accuracy = float(np.mean(owners[voronoi_sites] == test_labels))
	solved_accuracy = float(np.mean(owners[solved_sites] == test_labels))

	print(f"layer {run['layer']}, {len(sites)} sites ({per_class}/class), held-out n={len(test_points)}")
	print(f"{'class':<11}{'prior':>8}{'voronoi':>9}{'solved':>9}")
	for index, name in enumerate(CLASS_NAMES):
		print(f"{name:<11}{prior[index]:>8.4f}{voronoi_mass[index]:>9.4f}{solved_mass[index]:>9.4f}")
	print(f"L1 class-mass error   voronoi {np.abs(voronoi_mass - prior).sum():.4f}"
	      f"   alexandrov {np.abs(solved_mass - prior).sum():.4f}")
	print(f"held-out accuracy     voronoi {voronoi_accuracy:.4f}   alexandrov {solved_accuracy:.4f}")
	print(f"(experiment 1 reported power_accuracy {run['power_accuracy']:.4f} at 36 sites)")

	view_points, view_sites = display_frame(test_points, sites)
	figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
	draw_panel(axes[0], view_points, test_labels, view_sites, owners, zero, args.grid,
	           f"Voronoi, w = 0   (class-mass L1 {np.abs(voronoi_mass - prior).sum():.3f})")
	draw_panel(axes[1], view_points, test_labels, view_sites, owners, weights, args.grid,
	           f"Alexandrov, solved w   (class-mass L1 {np.abs(solved_mass - prior).sum():.3f})")
	axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.92, markerscale=2.0)

	figure.suptitle(
		f"{run['model']} — layer {run['layer']} power diagram, {len(sites)} sites "
		f"({per_class} per class), held-out split",
		fontsize=12.5, y=0.97,
	)
	stats = norm_stats(run["layer"], args.model)
	caption = (
		"Synthetic 2D cloud calibrated to Experiment 1 layer-30 norms  "
		+ "  ".join(f"{n[:4]} {stats[n]['mean_norm']:.0f}±{stats[n]['std_norm']:.1f}" for n in CLASS_NAMES)
		+ f"   |   measured covariance similarity: raw {similarity['raw']:.3f}, PNS {similarity['pns']:.3f}"
	)
	figure.text(0.5, 0.045, caption, ha="center", fontsize=8.5, color=MUTED)
	figure.tight_layout(rect=[0, 0.07, 1, 0.95])

	destination = REPO_ROOT / args.out
	destination.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(destination, dpi=190)
	print(f"wrote {destination}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
