#!/usr/bin/env python3
"""Diagnostics for the missing middle-layer separation hump.

	Tests three candidate explanations, per layer, on the saved activations:
	  1. Behavioral labels bias toward late layers -> also measure PROMPT labels
	     (harmful vs benign ground truth), which should reflect concept, not decision.
	  2. PCA-10 discards middle-layer signal -> sweep the probe over more PCA dims.
	  3. Nearest-centroid is too weak -> use a cross-validated logistic probe.

	python layer_diagnostics.py
	python layer_diagnostics.py --pca-dims 10 50 100
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import load_labels_frame, resolve_layers

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
SEED = 0
PROBE_DIMS = [10, 50, 100]


# Balance a two-class index set by subsampling the larger class to the smaller count.
def balanced_indices(binary_labels: np.ndarray, seed: int) -> np.ndarray:
	rng = random.Random(seed)
	groups = {value: np.where(binary_labels == value)[0].tolist() for value in set(binary_labels)}
	smallest = min(len(idx) for idx in groups.values())
	kept: list[int] = []
	for value, idx in groups.items():
		shuffled = list(idx)
		rng.shuffle(shuffled)
		kept.extend(shuffled[:smallest])
	return np.array(sorted(kept))


# Cross-validated balanced accuracy of a logistic probe on PCA-reduced, standardized activations.
def probe_accuracy(points: np.ndarray, labels: np.ndarray, dims: int) -> float:
	dims = min(dims, points.shape[1], points.shape[0] - 1)
	pipeline = Pipeline([
		("scale", StandardScaler()),
		("pca", PCA(n_components=dims, random_state=SEED)),
		("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
	])
	scores = cross_val_score(pipeline, points, labels, cv=5, scoring="balanced_accuracy")
	return float(scores.mean())


# Compute the diagnostic metrics for one layer.
def evaluate_layer(layer: int, data_dir: Path, behavioral: np.ndarray, prompt_binary: np.ndarray, prompt_index: np.ndarray) -> dict:
	print(f"evaluating layer {layer}", file=sys.stderr)
	points = np.load(data_dir / f"activations_layer{layer}.npy")
	row = {"layer": layer}
	row["behavioral_probe_d100"] = probe_accuracy(points, behavioral, 100)
	row["prompt_probe_d100"] = probe_accuracy(points[prompt_index], prompt_binary[prompt_index], 100)
	for dims in PROBE_DIMS:
		row[f"prompt_probe_d{dims}"] = probe_accuracy(points[prompt_index], prompt_binary[prompt_index], dims)
	return row


# Plot behavioral vs prompt probe accuracy across layers, with chance references.
def plot_label_comparison(rows: list[dict], out_path: Path) -> None:
	layers = [row["layer"] for row in rows]
	fig, ax = plt.subplots(figsize=(11.0, 5.5))
	ax.plot(layers, [r["behavioral_probe_d100"] for r in rows], marker="o", color="#7b2fbe", label="behavioral 3-way probe")
	ax.plot(layers, [r["prompt_probe_d100"] for r in rows], marker="s", color="#1f77b4", label="prompt harmful/benign probe")
	ax.axhline(1.0 / 3.0, linestyle="--", color="#b0b0b0", label="chance (3-way)")
	ax.axhline(0.5, linestyle=":", color="#b0b0b0", label="chance (2-way)")
	ax.set_xlabel("layer")
	ax.set_ylabel("cross-validated balanced accuracy")
	ax.set_title("Behavioral vs prompt-label separability by layer (probe, 100 PCA dims)")
	ax.set_xticks(layers)
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Plot the prompt-label probe accuracy across layers at several PCA dimensionalities.
def plot_dimension_sweep(rows: list[dict], out_path: Path) -> None:
	layers = [row["layer"] for row in rows]
	fig, ax = plt.subplots(figsize=(11.0, 5.5))
	colors = {10: "#1f77b4", 50: "#2ca02c", 100: "#d62728"}
	for dims in PROBE_DIMS:
		ax.plot(layers, [r[f"prompt_probe_d{dims}"] for r in rows], marker="o", color=colors.get(dims, "#555555"), label=f"{dims} PCA dims")
	ax.axhline(0.5, linestyle=":", color="#b0b0b0", label="chance (2-way)")
	ax.set_xlabel("layer")
	ax.set_ylabel("prompt harmful/benign probe accuracy")
	ax.set_title("Prompt-label separability by layer across PCA dimensionality")
	ax.set_xticks(layers)
	ax.legend(fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments, run the diagnostics for every layer, save the CSV and figures.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	frame = load_labels_frame(args.data_dir)
	behavioral = frame["class_label"].to_numpy()
	prompt_binary = np.where(frame["is_harmful_prompt"].to_numpy(), "harmful", "benign")
	prompt_index = balanced_indices(prompt_binary, SEED)
	layers = resolve_layers(args.data_dir)

	rows = [evaluate_layer(layer, args.data_dir, behavioral, prompt_binary, prompt_index) for layer in layers]

	result = pd.DataFrame(rows)
	csv_path = args.data_dir / "layer_diagnostics.csv"
	result.to_csv(csv_path, index=False)
	print(f"wrote {csv_path}", file=sys.stderr)
	print(result.to_string(index=False))

	args.figures_dir.mkdir(parents=True, exist_ok=True)
	plot_label_comparison(rows, args.figures_dir / "layer_diag_labels.png")
	plot_dimension_sweep(rows, args.figures_dir / "layer_diag_dims.png")


if __name__ == "__main__":
	main()
