#!/usr/bin/env python3
"""A 2D, CPU-fast picture of the Alexandrov weight solve.

	Purpose: make the weight solve visible before it is trusted in 2048 dimensions. Three
	synthetic classes stand in for the real ones and reproduce the covariance mismatch the paper
	is built on — Refusal tight and stereotyped, Jailbreak diffuse and elongated across attack
	styles, Benign in between. Sites are the class centroids; targets are the true class priors.

	Two panels, identical sites, only the weights differ:

		w = 0        plain Voronoi. Boundaries fall at centroid midpoints, so the tight class
		             annexes territory from the diffuse one and the cells carry the wrong mass.
		w = solved   the Alexandrov/Minkowski map. Each cell carries its prescribed mass, and
		             the boundaries are still exact hyperplanes.

	The visible gap between the panels is the "tight class encroaches on the diffuse one"
	failure mode — a systematic, directional error, not noise.

	Usage:
		.venv/bin/python experiment2/viz/alexandrov_2d_demo.py
		.venv/bin/python experiment2/viz/alexandrov_2d_demo.py --n-per-class 800 --grid 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from experiment2.lib.alexandrov import cell_assignment, solve_weights, squared_distances

# Categorical slots 1-3 of the validated palette; these three clear the all-pairs CVD floors.
CLASS_NAMES = ["Refusal", "Jailbreak", "Benign"]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
METHOD_COLOR = "#4a3aa7"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SEED = 0


# Three Gaussian classes with deliberately mismatched covariance, mirroring the hypothesis.
def synthetic_classes(n_per_class: int):
	rng = np.random.default_rng(SEED)
	means = np.array([[0.0, 0.0], [3.2, 0.9], [-0.4, 3.4]])
	covariances = [
		np.array([[0.16, 0.02], [0.02, 0.14]]),      # Refusal: tight, near-isotropic
		np.array([[2.30, 1.05], [1.05, 0.95]]),      # Jailbreak: diffuse, elongated
		np.array([[0.62, -0.18], [-0.18, 0.55]]),    # Benign: intermediate
	]
	points = np.vstack([
		rng.multivariate_normal(mean, covariance, size=n_per_class)
		for mean, covariance in zip(means, covariances)
	])
	labels = np.repeat(np.arange(len(CLASS_NAMES)), n_per_class)
	return points, labels


# Cell index for every node of a background grid spanning the point cloud.
def cell_grid(points: np.ndarray, sites: np.ndarray, weights: np.ndarray, resolution: int):
	pad = 0.6
	x_axis = np.linspace(points[:, 0].min() - pad, points[:, 0].max() + pad, resolution)
	y_axis = np.linspace(points[:, 1].min() - pad, points[:, 1].max() + pad, resolution)
	mesh_x, mesh_y = np.meshgrid(x_axis, y_axis)
	nodes = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
	power = squared_distances(nodes, sites) - weights[None, :]
	return x_axis, y_axis, np.argmin(power, axis=1).reshape(resolution, resolution)


# Discrete three-step colorscale so heatmap values 0/1/2 read as flat class regions.
def discrete_colorscale():
	steps = []
	for index, color in enumerate(CLASS_COLORS):
		steps.append([index / 3.0, color])
		steps.append([(index + 1) / 3.0, color])
	return steps


# Draw one panel: shaded cells, the point cloud, and directly-labelled sites.
def add_panel(figure, column, points, labels, sites, weights, resolution, show_legend):
	x_axis, y_axis, grid = cell_grid(points, sites, weights, resolution)
	figure.add_trace(
		go.Heatmap(
			x=x_axis, y=y_axis, z=grid,
			colorscale=discrete_colorscale(), zmin=-0.5, zmax=2.5,
			opacity=0.22, showscale=False, hoverinfo="skip",
		),
		row=1, col=column,
	)
	for index, name in enumerate(CLASS_NAMES):
		member = points[labels == index]
		figure.add_trace(
			go.Scattergl(
				x=member[:, 0], y=member[:, 1], mode="markers", name=name,
				legendgroup=name, showlegend=show_legend,
				marker=dict(size=5, color=CLASS_COLORS[index],
				            line=dict(width=0.8, color=SURFACE)),
				hovertemplate=f"{name}<br>%{{x:.2f}}, %{{y:.2f}}<extra></extra>",
			),
			row=1, col=column,
		)
	figure.add_trace(
		go.Scattergl(
			x=sites[:, 0], y=sites[:, 1], mode="markers", text=CLASS_NAMES,
			marker=dict(size=15, symbol="diamond", color=CLASS_COLORS,
			            line=dict(width=2.5, color=TEXT_PRIMARY)),
			showlegend=False,
			hovertemplate="site %{text}<extra></extra>",
		),
		row=1, col=column,
	)
	# Labels sit clear of the diamond; direct labelling is the relief the aqua slot's
	# sub-3:1 contrast against the light surface obliges.
	span = points[:, 1].max() - points[:, 1].min()
	figure.add_trace(
		go.Scattergl(
			x=sites[:, 0], y=sites[:, 1] + 0.10 * span, mode="text",
			text=CLASS_NAMES, textposition="top center",
			textfont=dict(size=13, color=TEXT_PRIMARY),
			showlegend=False, hoverinfo="skip",
		),
		row=1, col=column,
	)


# Grouped bars comparing the mass each cell actually carries against its target.
def add_mass_panel(figure, column, voronoi_masses, solved_masses, targets):
	figure.add_trace(
		go.Bar(x=CLASS_NAMES, y=voronoi_masses, name="Voronoi (w=0)",
		       marker_color=TEXT_SECONDARY, showlegend=True,
		       hovertemplate="Voronoi %{x}<br>mass %{y:.3f}<extra></extra>"),
		row=1, col=column,
	)
	figure.add_trace(
		go.Bar(x=CLASS_NAMES, y=solved_masses, name="Alexandrov (solved w)",
		       marker_color=METHOD_COLOR, showlegend=True,
		       hovertemplate="Alexandrov %{x}<br>mass %{y:.3f}<extra></extra>"),
		row=1, col=column,
	)
	figure.add_shape(
		type="line", x0=-0.5, x1=len(CLASS_NAMES) - 0.5,
		y0=targets[0], y1=targets[0],
		line=dict(color=TEXT_PRIMARY, width=2, dash="dot"),
		row=1, col=column,
	)
	figure.add_annotation(
		x=len(CLASS_NAMES) - 0.5, y=targets[0], text="target",
		showarrow=False, xanchor="right", yanchor="bottom",
		font=dict(size=11, color=TEXT_PRIMARY), row=1, col=column,
	)


# Build the synthetic classes, solve the weights, and write the figure.
def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--n-per-class", type=int, default=500)
	parser.add_argument("--grid", type=int, default=320)
	parser.add_argument("--out", type=str, default="experiment2/figures/alexandrov_2d.html")
	parser.add_argument("--png", action="store_true", help="Also write a static PNG (needs kaleido).")
	args = parser.parse_args()

	points, labels = synthetic_classes(args.n_per_class)
	sites = np.vstack([points[labels == index].mean(axis=0) for index in range(len(CLASS_NAMES))])
	point_masses = np.full(len(points), 1.0 / len(points))
	targets = np.array([np.mean(labels == index) for index in range(len(CLASS_NAMES))])

	squared = squared_distances(points, sites)
	zero_weights = np.zeros(len(sites))
	_, voronoi_masses = cell_assignment(squared, zero_weights, point_masses)
	weights, info = solve_weights(points, sites, targets, point_masses)

	print(f"solved in {info['iterations']} L-BFGS iterations")
	print(f"{'class':<12}{'target':>9}{'voronoi':>10}{'alexandrov':>13}{'weight':>10}")
	for index, name in enumerate(CLASS_NAMES):
		print(f"{name:<12}{targets[index]:>9.4f}{voronoi_masses[index]:>10.4f}"
		      f"{info['achieved_masses'][index]:>13.4f}{weights[index]:>10.4f}")
	voronoi_error = float(np.abs(voronoi_masses - targets).sum())
	print(f"\nL1 mass error   voronoi {voronoi_error:.4f}   alexandrov {info['mass_error_l1']:.4f}")

	figure = make_subplots(
		rows=1, cols=3, column_widths=[0.36, 0.36, 0.28],
		subplot_titles=(
			"Plain Voronoi \u2014 w = 0",
			"Alexandrov power diagram \u2014 solved w",
			"Mass carried by each cell",
		),
	)
	add_panel(figure, 1, points, labels, sites, zero_weights, args.grid, show_legend=True)
	add_panel(figure, 2, points, labels, sites, weights, args.grid, show_legend=False)
	add_mass_panel(figure, 3, voronoi_masses, info["achieved_masses"], targets)

	figure.update_layout(
		template="simple_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
		font=dict(color=TEXT_PRIMARY, size=13), height=560, width=1500,
		title=dict(
			text="Solved site weights move the boundary onto the mass-correct position",
			font=dict(size=18),
		),
		legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="left", x=0),
		barmode="group", margin=dict(t=90, b=90),
	)
	for column in (1, 2):
		figure.update_xaxes(showgrid=False, zeroline=False, row=1, col=column)
		figure.update_yaxes(showgrid=False, zeroline=False, scaleanchor=f"x{column}",
		                    scaleratio=1, row=1, col=column)
	figure.update_yaxes(title_text="cell mass share", row=1, col=3)

	destination = REPO_ROOT / args.out
	destination.parent.mkdir(parents=True, exist_ok=True)
	figure.write_html(str(destination), include_plotlyjs="cdn")
	print(f"wrote {destination}")
	if args.png:
		image_path = destination.with_suffix(".png")
		figure.write_image(str(image_path), scale=2)
		print(f"wrote {image_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
