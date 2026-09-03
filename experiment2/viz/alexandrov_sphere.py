#!/usr/bin/env python3
"""The same 36-site diagram after the spherical projection, drawn on a cap of S^2.

	Experiment 1's layer-30 norms sit within 1.6% of each other (Refusal 133.1, Jailbreak 131.0,
	Benign 132.1) at spreads of 4-6%, so the three classes occupy nearly the same shell. That is
	the hyperspherical premise, and it is why the radial coordinate is a plausible thing to
	discard: l2-normalising moves every activation onto the shell the classes already share.

	On the sphere the power diagram is Sugihara's spherical Laguerre construction. With sites
	also on the sphere, ||x-y||^2 - w = 2 - 2*x.y - w, so minimising power distance maximises a
	weighted cosine and every cell boundary is the intersection of a plane with S^2 -- a circular
	arc. Cells stay spherically convex and the boundaries stay exact.

	Why a cap and not the globe: the angular separation is calibrated so the plain diagram
	reproduces Experiment 1's reported accuracy (0.717 at 36 sites), and at that separation the
	three classes occupy a small cap. Drawn on a whole sphere the cells become meaningless
	far-field lunes, so the frame is rotated to put the class centre at the pole and only the
	populated cap is meshed.

	Points are synthetic and calibrated to Experiment 1's measured norms; the real activations
	are gitignored and absent from the repo.

	Usage:
		.venv/bin/python experiment2/viz/alexandrov_sphere.py --png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from experiment2.lib.alexandrov import cell_assignment, solve_weights, squared_distances
from experiment2.lib.exp1_stats import (
	CLASS_NAMES, calibrated_cloud, covariance_similarity, load_reported_run, norm_stats,
)

CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
SEED = 0


# Project onto the unit sphere: the spherical arm's first step, discarding the radial coordinate.
def to_sphere(points: np.ndarray) -> np.ndarray:
	return points / np.linalg.norm(points, axis=1, keepdims=True)


# Twelve sub-sites per class, each pushed back onto the sphere so the Sugihara form applies.
def fit_sphere_sites(points, labels, per_class):
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
	sites = to_sphere(np.array(sites))
	masses = np.array(masses, dtype=float)
	return sites, np.array(owners), masses / masses.sum()


# Rotation carrying a unit vector to the north pole, so the populated cap can be meshed.
def rotation_to_pole(direction: np.ndarray) -> np.ndarray:
	direction = direction / np.linalg.norm(direction)
	pole = np.array([0.0, 0.0, 1.0])
	axis = np.cross(direction, pole)
	sine = np.linalg.norm(axis)
	cosine = float(direction @ pole)
	if sine < 1e-12:
		return np.eye(3) if cosine > 0 else -np.eye(3)
	axis = axis / sine
	cross = np.array([
		[0.0, -axis[2], axis[1]],
		[axis[2], 0.0, -axis[0]],
		[-axis[1], axis[0], 0.0],
	])
	angle = np.arctan2(sine, cosine)
	return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


# Build the cloud at a given separation and score the plain spherical diagram on held-out data.
def build_split(separation, layer, model, n_per_class, radial_share, per_class):
	ambient, labels = calibrated_cloud(
		layer, model, n_per_class=n_per_class, dims=3,
		radial_share=radial_share, separation=separation, seed=SEED,
	)
	points = to_sphere(ambient)
	rng = np.random.default_rng(SEED)
	shuffled = rng.permutation(len(points))
	split = len(points) // 2
	train, test = shuffled[:split], shuffled[split:]
	sites, owners, shape = fit_sphere_sites(points[train], labels[train], per_class)
	assigned = np.argmin(squared_distances(points[test], sites), axis=1)
	accuracy = float(np.mean(owners[assigned] == labels[test]))
	return accuracy, (points[test], labels[test], sites, owners, shape)


# Bisect the angular separation until the plain diagram reproduces Experiment 1's accuracy.
def calibrate_separation(target, layer, model, n_per_class, radial_share, per_class,
                         iterations=22):
	low, high = 0.005, 1.2
	for _ in range(iterations):
		middle = 0.5 * (low + high)
		accuracy, _ = build_split(middle, layer, model, n_per_class, radial_share, per_class)
		if accuracy > target:
			high = middle
		else:
			low = middle
	separation = 0.5 * (low + high)
	accuracy, bundle = build_split(separation, layer, model, n_per_class, radial_share, per_class)
	return separation, accuracy, bundle


# Mesh the populated cap and label every vertex with its owning class.
def cap_mesh(sites, owners, weights, cap_angle, n_lat=170, n_lon=340):
	lat = np.linspace(0.0, cap_angle, n_lat)
	lon = np.linspace(-np.pi, np.pi, n_lon)
	lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
	x = np.sin(lat_grid) * np.cos(lon_grid)
	y = np.sin(lat_grid) * np.sin(lon_grid)
	z = np.cos(lat_grid)
	vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
	power = squared_distances(vertices, sites) - weights[None, :]
	owning = owners[np.argmin(power, axis=1)].reshape(lat_grid.shape)
	return x, y, z, owning.astype(float)


# Discrete three-step colorscale so the surface reads as flat class territory.
def discrete_colorscale():
	steps = []
	for index, color in enumerate(CLASS_COLORS):
		steps.append([index / 3.0, color])
		steps.append([(index + 1) / 3.0, color])
	return steps


# Draw one cap: class territory as surface colour, points and sites just above it.
def add_cap(figure, column, points, labels, sites, owners, weights, cap_angle, show_legend):
	x, y, z, owning = cap_mesh(sites, owners, weights, cap_angle)
	figure.add_trace(
		go.Surface(
			x=x, y=y, z=z, surfacecolor=owning,
			colorscale=discrete_colorscale(), cmin=-0.5, cmax=2.5,
			showscale=False, opacity=0.42, hoverinfo="skip",
			lighting=dict(ambient=0.98, diffuse=0.05, specular=0.0),
		),
		row=1, col=column,
	)
	shown = points * 1.004
	for index, name in enumerate(CLASS_NAMES):
		member = shown[labels == index]
		figure.add_trace(
			go.Scatter3d(
				x=member[:, 0], y=member[:, 1], z=member[:, 2], mode="markers", name=name,
				legendgroup=name, showlegend=show_legend,
				marker=dict(size=2.6, color=CLASS_COLORS[index]),
				hovertemplate=f"{name}<extra></extra>",
			),
			row=1, col=column,
		)
	anchor = sites * 1.011
	figure.add_trace(
		go.Scatter3d(
			x=anchor[:, 0], y=anchor[:, 1], z=anchor[:, 2], mode="markers",
			marker=dict(size=4.2, color="black"), showlegend=False,
			text=[CLASS_NAMES[o] for o in owners],
			hovertemplate="site (%{text})<extra></extra>",
		),
		row=1, col=column,
	)


# Calibrate, project, solve the weights, and write the figure.
def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model", type=str, default=None,
	                    help="Experiment 1 model to read; defaults to the only collected run.")
	parser.add_argument("--n-per-class", type=int, default=500)
	parser.add_argument("--radial-share", type=float, default=0.35)
	parser.add_argument("--out", type=str, default="experiment2/figures/alexandrov_sphere.html")
	parser.add_argument("--png", action="store_true")
	args = parser.parse_args()

	run = load_reported_run(args.model)
	per_class = run["n_sites"] // len(CLASS_NAMES)
	stats = norm_stats(run["layer"], args.model)
	similarity = covariance_similarity(run["layer"], args.model)

	separation, achieved, bundle = calibrate_separation(
		run["power_accuracy"], run["layer"], args.model, args.n_per_class,
		args.radial_share, per_class,
	)
	test_points, test_labels, sites, owners, shape = bundle
	print(f"calibrated angular separation {separation:.4f} -> plain-diagram accuracy "
	      f"{achieved:.4f} (experiment 1 reported {run['power_accuracy']:.4f})")

	prior = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
	targets = shape.astype(float).copy()
	for index in range(len(CLASS_NAMES)):
		owned = owners == index
		targets[owned] = targets[owned] / targets[owned].sum() * prior[index]

	test_masses = np.full(len(test_points), 1.0 / len(test_points))
	squared = squared_distances(test_points, sites)
	zero = np.zeros(len(sites))
	voronoi_sites, _ = cell_assignment(squared, zero, test_masses)
	weights, info = solve_weights(test_points, sites, targets, test_masses)

	def class_mass(site_index):
		return np.array([test_masses[owners[site_index] == c].sum() for c in range(3)])

	voronoi_mass, solved_mass = class_mass(voronoi_sites), class_mass(info["assigned"])
	voronoi_accuracy = float(np.mean(owners[voronoi_sites] == test_labels))
	solved_accuracy = float(np.mean(owners[info["assigned"]] == test_labels))

	# Put the class centre at the pole and mesh only as far out as the data reaches.
	centre = to_sphere(test_points.mean(axis=0)[None, :])[0]
	rotation = rotation_to_pole(centre)
	view_points = test_points @ rotation.T
	view_sites = sites @ rotation.T
	# The data's true angular extent, and a padded version used only for meshing. Keep them
	# distinct: reporting the padded value would overstate how much of the sphere the classes
	# actually occupy.
	data_cap = float(np.arccos(np.clip(view_points[:, 2], -1.0, 1.0)).max())
	cap_angle = data_cap * 1.25

	radial = np.array([stats[n]["std_norm"] for n in CLASS_NAMES])
	print(f"layer {run['layer']} on S^2, {len(sites)} sites ({per_class}/class), held-out n={len(test_points)}")
	print(f"norm spread ratio across classes: {radial.max() / radial.min():.2f}x "
	      f"({radial.min():.2f} to {radial.max():.2f});  data cap half-angle "
	      f"{np.degrees(data_cap):.1f} deg (mesh padded to {np.degrees(cap_angle):.1f} deg)")
	print(f"{'class':<11}{'prior':>8}{'voronoi':>9}{'solved':>9}")
	for index, name in enumerate(CLASS_NAMES):
		print(f"{name:<11}{prior[index]:>8.4f}{voronoi_mass[index]:>9.4f}{solved_mass[index]:>9.4f}")
	print(f"L1 class-mass error   voronoi {np.abs(voronoi_mass - prior).sum():.4f}"
	      f"   alexandrov {np.abs(solved_mass - prior).sum():.4f}")
	print(f"held-out accuracy     voronoi {voronoi_accuracy:.4f}   alexandrov {solved_accuracy:.4f}")

	figure = make_subplots(
		rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
		subplot_titles=(
			f"Spherical Laguerre, w = 0   (class-mass L1 {np.abs(voronoi_mass - prior).sum():.3f})",
			f"Alexandrov, solved w   (class-mass L1 {np.abs(solved_mass - prior).sum():.3f})",
		),
		horizontal_spacing=0.02,
	)
	add_cap(figure, 1, view_points, test_labels, view_sites, owners, zero, cap_angle, True)
	add_cap(figure, 2, view_points, test_labels, view_sites, owners, weights, cap_angle, False)

	axis_off = dict(showbackground=False, showgrid=False, showticklabels=False,
	                zeroline=False, title="", visible=False)
	scene = dict(xaxis=axis_off, yaxis=axis_off, zaxis=axis_off,
	             camera=dict(eye=dict(x=0.0, y=-0.42, z=1.62), up=dict(x=0, y=0, z=1)),
	             aspectmode="data")
	figure.update_layout(
		scene=scene, scene2=scene,
		paper_bgcolor="white", font=dict(color=INK, size=12),
		height=650, width=1240, margin=dict(t=84, b=96, l=8, r=8),
		title=dict(
			text=f"{run['model']} — layer {run['layer']} activations projected to the unit sphere, "
			     f"{len(sites)} sites ({per_class} per class), data cap half-angle "
			     f"{np.degrees(data_cap):.1f}°",
			x=0.5, xanchor="center", font=dict(size=13),
		),
		legend=dict(orientation="h", yanchor="top", y=-0.005, xanchor="center", x=0.5),
	)
	norms = "  ".join(f"{n[:4]} {stats[n]['mean_norm']:.0f}±{stats[n]['std_norm']:.1f}"
	                  for n in CLASS_NAMES)
	figure.add_annotation(
		x=0.5, y=-0.085, xref="paper", yref="paper", showarrow=False,
		text=(f"Synthetic cloud calibrated to Experiment 1 layer-30 norms ({norms})"
		      f"   |   measured covariance similarity: raw {similarity['raw']:.3f}, "
		      f"PNS {similarity['pns']:.3f}"),
		font=dict(size=10, color="#52514e"),
	)

	destination = REPO_ROOT / args.out
	destination.parent.mkdir(parents=True, exist_ok=True)
	figure.write_html(str(destination), include_plotlyjs="cdn")
	print(f"wrote {destination}")
	if args.png:
		figure.write_image(str(destination.with_suffix(".png")), scale=2)
		print(f"wrote {destination.with_suffix('.png')}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
