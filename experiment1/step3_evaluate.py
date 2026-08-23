#!/usr/bin/env python3
"""Evaluate five decision-boundary methods against the Bayes-optimal ground truth.

	python step3_evaluate.py
	python step3_evaluate.py --data-dir ./data --figures-dir ./figures
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
from scipy.spatial import cKDTree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

STRATA_ORDER = ["low", "high"]
METHOD_ORDER = ["bayes_optimal", "qda", "linear_probe", "plain_voronoi", "power_diagram"]
BOUNDARY_SAMPLE_SIZE = 1000


# One fitted method exposing a real-valued decision score over (N, 2) points.
@dataclass
class BoundaryMethod:
	name: str
	score: Callable[[np.ndarray], np.ndarray]


# Load the shared params.json describing means, strata, and the grid.
def load_params(data_dir: Path) -> dict:
	with (data_dir / "params.json").open(encoding="utf-8") as handle:
		return json.load(handle)


# Read one (stratum, split) CSV into a DataFrame with columns x, y, label.
def load_frame(data_dir: Path, stratum: str, split: str) -> pd.DataFrame:
	return pd.read_csv(data_dir / f"stratum_{stratum}_{split}.csv")


# Split a frame into (N, 2) points and numeric labels with B = Jailbreak = 1.
def split_points_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	points = frame[["x", "y"]].to_numpy(dtype=float)
	numeric_labels = (frame["label"].to_numpy() == "B").astype(int)
	return points, numeric_labels


# Estimate per-class means and covariances from training points.
def estimate_class_stats(train_points: np.ndarray, numeric_labels: np.ndarray):
	points_a = train_points[numeric_labels == 0]
	points_b = train_points[numeric_labels == 1]
	mu_hat_a = points_a.mean(axis=0)
	mu_hat_b = points_b.mean(axis=0)
	sigma_hat_a = np.cov(points_a, rowvar=False)
	sigma_hat_b = np.cov(points_b, rowvar=False)
	return mu_hat_a, mu_hat_b, sigma_hat_a, sigma_hat_b


# Gaussian log-likelihood of each point under a mean and covariance.
def gaussian_log_likelihood(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
	difference = points - mean
	precision = np.linalg.inv(covariance)
	mahalanobis_squared = np.einsum("ij,jk,ik->i", difference, precision, difference)
	determinant = np.linalg.det(2.0 * np.pi * covariance)
	return -0.5 * mahalanobis_squared - 0.5 * np.log(determinant)


# Fit all five methods on the training data in stable order.
def fit_methods(train_points: np.ndarray, numeric_labels: np.ndarray, true_params: dict, k: float) -> list[BoundaryMethod]:
	mu_hat_a, mu_hat_b, sigma_hat_a, sigma_hat_b = estimate_class_stats(train_points, numeric_labels)
	true_mu_a = np.asarray(true_params["mu_A"], dtype=float)
	true_mu_b = np.asarray(true_params["mu_B"], dtype=float)
	true_sigma_a = np.eye(2)
	true_sigma_b = k * np.eye(2)
	weight_a = float(np.trace(sigma_hat_a))
	weight_b = float(np.trace(sigma_hat_b))
	qda_model = QuadraticDiscriminantAnalysis().fit(train_points, numeric_labels)
	logistic_model = LogisticRegression().fit(train_points, numeric_labels)

	# Bayes-optimal quadratic discriminant using the true parameters.
	def bayes_optimal_score(points: np.ndarray) -> np.ndarray:
		return gaussian_log_likelihood(points, true_mu_b, true_sigma_b) - gaussian_log_likelihood(points, true_mu_a, true_sigma_a)

	# QDA decision function with B = 1 as the positive class.
	def qda_score(points: np.ndarray) -> np.ndarray:
		return qda_model.decision_function(points)

	# Linear logistic-regression decision function with B = 1 as positive.
	def linear_probe_score(points: np.ndarray) -> np.ndarray:
		return logistic_model.decision_function(points)

	# Plain nearest-mean (Voronoi) score: squared-distance difference.
	def plain_voronoi_score(points: np.ndarray) -> np.ndarray:
		return np.sum((points - mu_hat_a) ** 2, axis=1) - np.sum((points - mu_hat_b) ** 2, axis=1)

	# Power-diagram score: Voronoi score with trace-based weights subtracted.
	def power_diagram_score(points: np.ndarray) -> np.ndarray:
		return (np.sum((points - mu_hat_a) ** 2, axis=1) - weight_a) - (np.sum((points - mu_hat_b) ** 2, axis=1) - weight_b)

	return [
		BoundaryMethod("bayes_optimal", bayes_optimal_score),
		BoundaryMethod("qda", qda_score),
		BoundaryMethod("linear_probe", linear_probe_score),
		BoundaryMethod("plain_voronoi", plain_voronoi_score),
		BoundaryMethod("power_diagram", power_diagram_score),
	]


# Build the evaluation grid axes from the params grid specification.
def build_grid(params: dict) -> tuple[np.ndarray, np.ndarray]:
	grid = params["grid"]
	minimum = float(grid["min"])
	maximum = float(grid["max"])
	resolution = float(grid["resolution"])
	count = int(round((maximum - minimum) / resolution)) + 1
	axis = np.linspace(minimum, maximum, count)
	return axis, axis.copy()


# Extract the g = 0 boundary as an (N, 2) set of contour points.
def contour_points(score_values_on_grid: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
	figure = plt.figure()
	axes = figure.add_subplot(111)
	contour_set = axes.contour(grid_x, grid_y, score_values_on_grid, levels=[0.0])
	segments = [segment for segment in contour_set.allsegs[0] if len(segment) > 0]
	plt.close(figure)
	if not segments:
		return np.empty((0, 2))
	return np.vstack(segments)


# Fraction of test points whose thresholded prediction differs from the truth.
def misclassification_rate(scores: np.ndarray, numeric_labels: np.ndarray) -> float:
	prediction = (scores > 0).astype(int)
	return float(np.mean(prediction != numeric_labels))


# Area under the ROC curve for the B = 1 positive class.
def compute_auc(numeric_labels: np.ndarray, scores: np.ndarray) -> float:
	return float(roc_auc_score(numeric_labels, scores))


# Root-mean-square perpendicular distance from Bayes to candidate boundary.
def boundary_rms(bayes_boundary_points: np.ndarray, candidate_boundary_points: np.ndarray) -> float:
	if len(bayes_boundary_points) == 0 or len(candidate_boundary_points) == 0:
		return float("nan")
	tree = cKDTree(candidate_boundary_points)
	distances, _ = tree.query(bayes_boundary_points)
	return float(np.sqrt(np.mean(distances ** 2)))


# Area over the grid where the method and Bayes predictions disagree.
def area_error(method_predictions: np.ndarray, bayes_predictions: np.ndarray, total_area: float) -> float:
	return float(np.mean(method_predictions != bayes_predictions) * total_area)


# Subsample a boundary point set down to about the requested count.
def subsample_points(points: np.ndarray, count: int) -> np.ndarray:
	if len(points) <= count:
		return points
	indices = np.linspace(0, len(points) - 1, count).astype(int)
	return points[indices]


# Fit and evaluate every method on one stratum, returning metric rows.
def evaluate_stratum(data_dir: Path, stratum: str, params: dict) -> list[dict]:
	k = float(params["strata"][stratum])
	print(f"  evaluating stratum {stratum} (k = {k}) ...", file=sys.stderr)
	train_points, train_labels = split_points_labels(load_frame(data_dir, stratum, "train"))
	test_points, test_labels = split_points_labels(load_frame(data_dir, stratum, "test"))
	methods = fit_methods(train_points, train_labels, params, k)

	grid_x, grid_y = build_grid(params)
	mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
	grid_points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
	grid_shape = mesh_x.shape
	total_area = (float(params["grid"]["max"]) - float(params["grid"]["min"])) ** 2

	bayes_scores_grid = methods[0].score(grid_points).reshape(grid_shape)
	bayes_predictions = bayes_scores_grid > 0
	bayes_boundary = subsample_points(contour_points(bayes_scores_grid, grid_x, grid_y), BOUNDARY_SAMPLE_SIZE)

	rows: list[dict] = []
	for method in methods:
		test_scores = method.score(test_points)
		grid_scores = method.score(grid_points).reshape(grid_shape)
		candidate_boundary = contour_points(grid_scores, grid_x, grid_y)
		rows.append({
			"stratum": stratum,
			"method": method.name,
			"misclassification_rate": misclassification_rate(test_scores, test_labels),
			"auc": compute_auc(test_labels, test_scores),
			"boundary_rms": boundary_rms(bayes_boundary, candidate_boundary),
			"area_error": area_error(grid_scores > 0, bayes_predictions, total_area),
		})
	return rows


# Save a grouped bar chart of boundary RMS per method across strata.
def plot_metrics_summary(frame: pd.DataFrame, figures_dir: Path) -> Path:
	figures_dir.mkdir(parents=True, exist_ok=True)
	strata = [stratum for stratum in STRATA_ORDER if stratum in set(frame["stratum"])]
	positions = np.arange(len(METHOD_ORDER))
	width = 0.8 / max(len(strata), 1)
	figure, axes = plt.subplots(figsize=(10, 6))
	for index, stratum in enumerate(strata):
		subset = frame[frame["stratum"] == stratum].set_index("method")
		values = [float(subset.loc[method, "boundary_rms"]) for method in METHOD_ORDER]
		offset = (index - (len(strata) - 1) / 2.0) * width
		axes.bar(positions + offset, values, width, label=f"stratum {stratum}")
	axes.set_xticks(positions)
	axes.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
	axes.set_ylabel("boundary RMS distance to Bayes boundary")
	axes.set_title("Boundary distance error by method and covariance stratum")
	axes.legend()
	path = figures_dir / "metrics_summary.png"
	figure.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(figure)
	return path


# Parse arguments, evaluate both strata, write metrics.csv, and save the figure.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
		help="Directory holding params.json and the stratum CSVs (default: ./data).",
	)
	parser.add_argument(
		"--figures-dir", type=Path, default=Path(__file__).resolve().parent / "figures",
		help="Directory to write the summary figure (default: ./figures).",
	)
	args = parser.parse_args()

	print(f"Evaluating boundary methods from {args.data_dir} ...", file=sys.stderr)
	params = load_params(args.data_dir)
	rows: list[dict] = []
	for stratum in STRATA_ORDER:
		rows.extend(evaluate_stratum(args.data_dir, stratum, params))

	columns = ["stratum", "method", "misclassification_rate", "auc", "boundary_rms", "area_error"]
	frame = pd.DataFrame(rows, columns=columns)

	metrics_path = args.data_dir / "metrics.csv"
	frame.to_csv(metrics_path, index=False)
	print(f"  wrote {metrics_path} ({len(frame)} rows)", file=sys.stderr)

	print(frame.to_string(index=False))

	figure_path = plot_metrics_summary(frame, args.figures_dir)
	print(f"  wrote {figure_path}", file=sys.stderr)


if __name__ == "__main__":
	main()
