#!/usr/bin/env python3
"""Layer-transition graph: between-class covariance similarity across the collected layers.

	For each layer we measure how similar the three classes' covariance matrices are (a
	single centroid and covariance per class, not multi-centroid). The output is one graph
	with layers on the y-axis and covariance similarity on the x-axis: low similarity means
	high covariance mismatch, which is where a covariance-aware boundary matters most. It is
	measured in both raw-PCA space and the PNS (unwrapped) space to show what unwrapping does.
	Per-layer nearest-centroid accuracy and separation ratio are also written to the CSV.

	python layer_transition.py
	python layer_transition.py --layers 2 4 8 16 24 32 --pca-dims 10
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
from sklearn.decomposition import PCA

from common import CLASS_NAMES, load_labels, resolve_layers

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# Reduce raw activations to a fixed number of PCA dimensions for a fair cross-layer comparison.
def raw_pca_space(layer: int, data_dir: Path, pca_dims: int) -> np.ndarray:
	activations = np.load(data_dir / f"activations_layer{layer}.npy")
	dims = min(pca_dims, activations.shape[1], activations.shape[0])
	return PCA(n_components=dims, random_state=0).fit_transform(activations)


# Load the PNS unwrapped scores for one layer.
def pns_space(layer: int, data_dir: Path) -> np.ndarray:
	return np.load(data_dir / f"pns_layer{layer}.npy")


# Compute one centroid (mean vector) per class present in the data.
def class_centroids(points: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
	centroids: dict[str, np.ndarray] = {}
	for name in CLASS_NAMES:
		mask = labels == name
		if mask.any():
			centroids[name] = points[mask].mean(axis=0)
	return centroids


# Fraction of points whose nearest single class centroid is their true class.
def nearest_centroid_accuracy(points: np.ndarray, labels: np.ndarray, centroids: dict[str, np.ndarray]) -> float:
	names = list(centroids)
	stacked = np.stack([centroids[name] for name in names])
	distances = np.linalg.norm(points[:, None, :] - stacked[None, :, :], axis=2)
	predicted = np.array([names[i] for i in distances.argmin(axis=1)])
	return float((predicted == labels).mean())


# Ratio of mean between-centroid distance to mean within-class spread (higher = more separated and tighter).
def separation_ratio(points: np.ndarray, labels: np.ndarray, centroids: dict[str, np.ndarray]) -> float:
	names = list(centroids)
	between = []
	for i in range(len(names)):
		for j in range(i + 1, len(names)):
			between.append(np.linalg.norm(centroids[names[i]] - centroids[names[j]]))
	within = []
	for name in names:
		mask = labels == name
		spread = np.sqrt(np.mean(np.sum((points[mask] - centroids[name]) ** 2, axis=1)))
		within.append(spread)
	mean_within = float(np.mean(within))
	if mean_within == 0:
		return float("nan")
	return float(np.mean(between) / mean_within)


# Mean cosine similarity between the classes' covariance matrices (1 = identical spread, low = mismatch).
def covariance_similarity(points: np.ndarray, labels: np.ndarray) -> float:
	covariances = []
	for name in CLASS_NAMES:
		mask = labels == name
		if mask.sum() >= 2:
			covariances.append(np.cov(points[mask], rowvar=False).ravel())
	similarities = []
	for i in range(len(covariances)):
		for j in range(i + 1, len(covariances)):
			denom = np.linalg.norm(covariances[i]) * np.linalg.norm(covariances[j])
			if denom > 0:
				similarities.append(float(np.dot(covariances[i], covariances[j]) / denom))
	if not similarities:
		return float("nan")
	return float(np.mean(similarities))


# Compute nearest-centroid accuracy, separation ratio, and covariance similarity for one space.
def evaluate_space(points: np.ndarray, labels: np.ndarray) -> dict:
	centroids = class_centroids(points, labels)
	return {
		"accuracy": nearest_centroid_accuracy(points, labels, centroids),
		"separation_ratio": separation_ratio(points, labels, centroids),
		"covariance_similarity": covariance_similarity(points, labels),
	}


# Build the metric rows for every layer in both the raw-PCA and PNS spaces.
def compute_transition(layers: list[int], labels: np.ndarray, data_dir: Path, pca_dims: int) -> list[dict]:
	rows: list[dict] = []
	for layer in layers:
		print(f"evaluating layer {layer}", file=sys.stderr)
		raw = evaluate_space(raw_pca_space(layer, data_dir, pca_dims), labels)
		pns = evaluate_space(pns_space(layer, data_dir), labels)
		rows.append({
			"layer": layer,
			"raw_accuracy": raw["accuracy"],
			"pns_accuracy": pns["accuracy"],
			"raw_separation_ratio": raw["separation_ratio"],
			"pns_separation_ratio": pns["separation_ratio"],
			"raw_covariance_similarity": raw["covariance_similarity"],
			"pns_covariance_similarity": pns["covariance_similarity"],
		})
	return rows


# Plot between-class covariance similarity per layer, with layers on the y-axis and similarity on the x-axis.
def plot_covariance_similarity(rows: list[dict], out_path: Path) -> None:
	layers = [row["layer"] for row in rows]
	raw_sim = [row["raw_covariance_similarity"] for row in rows]
	pns_sim = [row["pns_covariance_similarity"] for row in rows]
	positions = list(range(len(layers)))

	fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.45 * len(layers))))
	ax.plot(raw_sim, positions, marker="o", color="#1f77b4", label="raw (PCA)")
	ax.plot(pns_sim, positions, marker="s", color="#d62728", label="PNS")
	ax.set_yticks(positions)
	ax.set_yticklabels([str(layer) for layer in layers])
	ax.set_ylabel("layer")
	ax.set_xlabel("between-class covariance similarity (1 = identical spread, low = mismatch)")
	ax.set_title("Class covariance similarity by layer")
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments, compute the per-layer separation metrics, save the CSV and figures.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels = np.array(load_labels(args.data_dir))
	layers = resolve_layers(args.data_dir, args.layers)
	rows = compute_transition(layers, labels, args.data_dir, args.pca_dims)

	frame = pd.DataFrame(rows)
	csv_path = args.data_dir / "layer_transition.csv"
	frame.to_csv(csv_path, index=False)
	print(f"wrote {csv_path}", file=sys.stderr)
	print(frame.to_string(index=False))

	args.figures_dir.mkdir(parents=True, exist_ok=True)
	plot_covariance_similarity(rows, args.figures_dir / "layer_covariance_similarity.png")


if __name__ == "__main__":
	main()
