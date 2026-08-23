#!/usr/bin/env python3
"""Fit five decision-boundary methods on the synthetic strata and save boundary figures.

	python step2_fit_boundaries.py
	python step2_fit_boundaries.py --data-dir ./data --figures-dir ./figures
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

STRATA = ["low", "high"]
METHOD_COLORS = {
	"bayes_optimal": "#1b1b1b",
	"qda": "#2e7d32",
	"linear_probe": "#c62828",
	"plain_voronoi": "#1565c0",
	"power_diagram": "#ef6c00",
}


# Container pairing a method name with its decision-score function g(points) -> scores.
@dataclass
class BoundaryMethod:
	name: str
	score: Callable[[np.ndarray], np.ndarray]


# Load the shared params.json written by step1.
def load_params(data_dir: Path) -> dict:
	path = data_dir / "params.json"
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


# Load one stratum's training frame (columns x, y, label).
def load_train_frame(data_dir: Path, stratum: str) -> pd.DataFrame:
	path = data_dir / f"stratum_{stratum}_train.csv"
	return pd.read_csv(path)


# Encode labels as a numeric array with B (Jailbreak) as the positive class 1 and A as 0.
def encode_labels(labels: np.ndarray) -> np.ndarray:
	return (labels == "B").astype(int)


# Estimate per-class sample means and covariances from training points and numeric labels.
def estimate_class_stats(points: np.ndarray, numeric_labels: np.ndarray) -> dict:
	points_a = points[numeric_labels == 0]
	points_b = points[numeric_labels == 1]
	mu_hat_a = points_a.mean(axis=0)
	mu_hat_b = points_b.mean(axis=0)
	sigma_hat_a = np.cov(points_a, rowvar=False)
	sigma_hat_b = np.cov(points_b, rowvar=False)
	return {
		"mu_hat_A": mu_hat_a,
		"mu_hat_B": mu_hat_b,
		"Sigma_hat_A": sigma_hat_a,
		"Sigma_hat_B": sigma_hat_b,
	}


# Gaussian log-likelihood of points under a mean and covariance (up to no dropped terms).
def gaussian_log_likelihood(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
	mean = np.asarray(mean, dtype=float)
	covariance = np.asarray(covariance, dtype=float)
	inverse = np.linalg.inv(covariance)
	centered = points - mean
	mahalanobis_sq = np.einsum("ni,ij,nj->n", centered, inverse, centered)
	log_norm = np.log(np.linalg.det(2.0 * np.pi * covariance))
	return -0.5 * mahalanobis_sq - 0.5 * log_norm


# Build the Bayes-optimal score using the TRUE parameters and equal priors.
def build_bayes_optimal(true_params: dict, k: int) -> BoundaryMethod:
	mu_a = np.asarray(true_params["mu_A"], dtype=float)
	mu_b = np.asarray(true_params["mu_B"], dtype=float)
	covariance_a = np.eye(2)
	covariance_b = float(k) * np.eye(2)

	def score(points: np.ndarray) -> np.ndarray:
		loglik_b = gaussian_log_likelihood(points, mu_b, covariance_b)
		loglik_a = gaussian_log_likelihood(points, mu_a, covariance_a)
		return loglik_b - loglik_a

	return BoundaryMethod(name="bayes_optimal", score=score)


# Build the QDA score from a fitted QuadraticDiscriminantAnalysis classifier.
def build_qda(points: np.ndarray, numeric_labels: np.ndarray) -> BoundaryMethod:
	classifier = QuadraticDiscriminantAnalysis()
	classifier.fit(points, numeric_labels)

	def score(query: np.ndarray) -> np.ndarray:
		return np.ravel(classifier.decision_function(query))

	return BoundaryMethod(name="qda", score=score)


# Build the linear-probe score from a fitted LogisticRegression classifier.
def build_linear_probe(points: np.ndarray, numeric_labels: np.ndarray) -> BoundaryMethod:
	classifier = LogisticRegression()
	classifier.fit(points, numeric_labels)

	def score(query: np.ndarray) -> np.ndarray:
		return np.ravel(classifier.decision_function(query))

	return BoundaryMethod(name="linear_probe", score=score)


# Build the plain (unweighted) Voronoi score: closer to B is positive.
def build_plain_voronoi(stats: dict) -> BoundaryMethod:
	mu_a = stats["mu_hat_A"]
	mu_b = stats["mu_hat_B"]

	def score(points: np.ndarray) -> np.ndarray:
		dist_a = np.sum((points - mu_a) ** 2, axis=1)
		dist_b = np.sum((points - mu_b) ** 2, axis=1)
		return dist_a - dist_b

	return BoundaryMethod(name="plain_voronoi", score=score)


# Build the power-diagram score using site weights equal to each class covariance trace.
def build_power_diagram(stats: dict) -> BoundaryMethod:
	mu_a = stats["mu_hat_A"]
	mu_b = stats["mu_hat_B"]
	weight_a = np.trace(stats["Sigma_hat_A"])
	weight_b = np.trace(stats["Sigma_hat_B"])

	def score(points: np.ndarray) -> np.ndarray:
		dist_a = np.sum((points - mu_a) ** 2, axis=1)
		dist_b = np.sum((points - mu_b) ** 2, axis=1)
		return (dist_a - weight_a) - (dist_b - weight_b)

	return BoundaryMethod(name="power_diagram", score=score)


# Fit all five methods and return them in a stable order.
def fit_methods(train_points: np.ndarray, numeric_labels: np.ndarray, true_params: dict, k: int) -> list[BoundaryMethod]:
	stats = estimate_class_stats(train_points, numeric_labels)
	return [
		build_bayes_optimal(true_params, k),
		build_qda(train_points, numeric_labels),
		build_linear_probe(train_points, numeric_labels),
		build_plain_voronoi(stats),
		build_power_diagram(stats),
	]


# Build the 2D evaluation grid (mesh_x, mesh_y, flat_points) from the params grid spec.
def build_grid(true_params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	grid = true_params["grid"]
	grid_min = float(grid["min"])
	grid_max = float(grid["max"])
	resolution = float(grid["resolution"])
	n_steps = int(round((grid_max - grid_min) / resolution)) + 1
	axis = np.linspace(grid_min, grid_max, n_steps)
	mesh_x, mesh_y = np.meshgrid(axis, axis)
	flat_points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
	return mesh_x, mesh_y, flat_points


# Evaluate a method's score on the flat grid points and reshape to the mesh shape.
def evaluate_on_grid(method: BoundaryMethod, flat_points: np.ndarray, mesh_shape: tuple[int, int]) -> np.ndarray:
	scores = method.score(flat_points)
	return scores.reshape(mesh_shape)


# Extract sampled g=0 contour points for a method as an (M,2) array (may be empty).
def extract_boundary_points(mesh_x: np.ndarray, mesh_y: np.ndarray, score_grid: np.ndarray) -> np.ndarray:
	figure = plt.figure()
	axes = figure.add_subplot(111)
	contour_set = axes.contour(mesh_x, mesh_y, score_grid, levels=[0.0])
	segments = []
	for path in contour_set.get_paths():
		vertices = path.vertices
		if vertices.size:
			segments.append(vertices)
	plt.close(figure)
	if segments:
		return np.vstack(segments)
	return np.empty((0, 2))


# Draw a training subsample plus each method's g=0 boundary and save the figure.
def plot_boundaries(
	train_points: np.ndarray,
	numeric_labels: np.ndarray,
	methods: list[BoundaryMethod],
	mesh_x: np.ndarray,
	mesh_y: np.ndarray,
	score_grids: dict,
	stratum: str,
	k: int,
	figures_dir: Path,
	subsample: int = 800,
) -> Path:
	figures_dir.mkdir(parents=True, exist_ok=True)
	figure = plt.figure(figsize=(8, 8))
	axes = figure.add_subplot(111)
	rng = np.random.default_rng(0)
	n_points = train_points.shape[0]
	take = min(subsample, n_points)
	indices = rng.choice(n_points, size=take, replace=False)
	sample_points = train_points[indices]
	sample_labels = numeric_labels[indices]
	axes.scatter(
		sample_points[sample_labels == 0, 0], sample_points[sample_labels == 0, 1],
		s=8, alpha=0.4, color="#4a4a4a", label="A (Refusal)",
	)
	axes.scatter(
		sample_points[sample_labels == 1, 0], sample_points[sample_labels == 1, 1],
		s=8, alpha=0.4, color="#2e7d32", label="B (Jailbreak)",
	)
	for method in methods:
		color = METHOD_COLORS.get(method.name, "#000000")
		axes.contour(mesh_x, mesh_y, score_grids[method.name], levels=[0.0], colors=[color], linewidths=1.6)
		axes.plot([], [], color=color, linewidth=1.6, label=method.name)
	axes.set_xlabel("x")
	axes.set_ylabel("y")
	axes.set_aspect("equal", adjustable="box")
	axes.set_title(f"Decision boundaries — stratum {stratum} (k={k})")
	axes.legend(loc="upper right", fontsize=8)
	path = figures_dir / f"boundaries_{stratum}.png"
	figure.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(figure)
	return path


# Write sampled boundary points for all methods of one stratum to a CSV.
def write_boundary_points(
	methods: list[BoundaryMethod],
	mesh_x: np.ndarray,
	mesh_y: np.ndarray,
	score_grids: dict,
	stratum: str,
	data_dir: Path,
) -> Path:
	rows = []
	for method in methods:
		boundary = extract_boundary_points(mesh_x, mesh_y, score_grids[method.name])
		for point in boundary:
			rows.append({"method": method.name, "x": float(point[0]), "y": float(point[1])})
	frame = pd.DataFrame(rows, columns=["method", "x", "y"])
	path = data_dir / f"boundary_points_{stratum}.csv"
	frame.to_csv(path, index=False)
	return path


# Fit, evaluate, plot, and export boundaries for a single stratum.
def run_stratum(stratum: str, true_params: dict, data_dir: Path, figures_dir: Path) -> None:
	print(f"Stratum {stratum}: loading training data ...", file=sys.stderr)
	frame = load_train_frame(data_dir, stratum)
	train_points = frame[["x", "y"]].to_numpy(dtype=float)
	numeric_labels = encode_labels(frame["label"].to_numpy())
	k = int(true_params["strata"][stratum])

	print(f"Stratum {stratum}: fitting five methods ...", file=sys.stderr)
	methods = fit_methods(train_points, numeric_labels, true_params, k)

	print(f"Stratum {stratum}: building grid ...", file=sys.stderr)
	mesh_x, mesh_y, flat_points = build_grid(true_params)
	mesh_shape = mesh_x.shape

	score_grids = {}
	for method in methods:
		print(f"Stratum {stratum}: scoring {method.name} on grid ...", file=sys.stderr)
		score_grids[method.name] = evaluate_on_grid(method, flat_points, mesh_shape)

	print(f"Stratum {stratum}: writing boundary points ...", file=sys.stderr)
	points_path = write_boundary_points(methods, mesh_x, mesh_y, score_grids, stratum, data_dir)
	print(f"  wrote {points_path}", file=sys.stderr)

	print(f"Stratum {stratum}: plotting boundaries ...", file=sys.stderr)
	figure_path = plot_boundaries(train_points, numeric_labels, methods, mesh_x, mesh_y, score_grids, stratum, k, figures_dir)
	print(f"  wrote {figure_path}", file=sys.stderr)


# Parse arguments, then fit and plot boundaries for both strata.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
		help="Directory with CSVs and params.json (default: ./data next to this script).",
	)
	parser.add_argument(
		"--figures-dir", type=Path, default=Path(__file__).resolve().parent / "figures",
		help="Directory to write boundary figures (default: ./figures next to this script).",
	)
	args = parser.parse_args()

	print(f"Reading params from {args.data_dir} ...", file=sys.stderr)
	true_params = load_params(args.data_dir)
	for stratum in STRATA:
		run_stratum(stratum, true_params, args.data_dir, args.figures_dir)


if __name__ == "__main__":
	main()
