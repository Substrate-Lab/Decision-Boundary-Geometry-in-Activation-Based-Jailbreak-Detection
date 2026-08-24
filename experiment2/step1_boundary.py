#!/usr/bin/env python3
"""Fit the covariance-aware boundary per layer and record cell assignment and margin.

	Step 1 of Experiment 2. For each layer and each selected discriminant, fit on a stratified
	train split and evaluate on a held-out test split, assigning every test point to its cell and
	recording the geometric margin = (second-smallest score) - (smallest score). Large margin
	means the point sits deep inside its cell; near zero means it sits on the boundary. Writes a
	per-point margin CSV per (space, layer, model) and a boundary_summary.csv across all runs.

	python step1_boundary.py
	python step1_boundary.py --models power_multi qda --space rawpca --layers 24 30 34
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from discriminants import build_discriminant
from space import DEFAULT_DATA_DIR, load_labels, resolve_layers, working_space

DEFAULT_MODELS = ["power_multi", "qda", "lda"]
SEED = 0
TEST_SIZE = 0.3

OUTPUT_DIR = Path(__file__).resolve().parent / "data"


# Fit one model on the train split and score the test split for one layer.
def evaluate(points: np.ndarray, labels: np.ndarray, model_name: str, sites_per_class: int):
	train_x, test_x, train_y, test_y, _, test_index = train_test_split(
		points, labels, np.arange(len(labels)), test_size=TEST_SIZE, random_state=SEED, stratify=labels,
	)
	model = build_discriminant(model_name, sites_per_class=sites_per_class).fit(train_x, train_y)
	predicted, margin = model.assign_and_margin(test_x)
	frame = pd.DataFrame({
		"row_index": test_index,
		"true": test_y,
		"predicted": predicted,
		"margin": margin,
		"correct": (predicted == test_y).astype(int),
	})
	return frame


# Parse arguments, fit every model on every layer, and write margins and the summary.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
	parser.add_argument("--space", choices=["rawpca", "pns"], default="rawpca")
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--sites-per-class", type=int, default=12)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	summary_rows = []
	for layer in resolve_layers(args.data_dir, args.layers):
		points = working_space(layer, args.data_dir, args.space, args.pca_dims)
		print(f"layer {layer}: points {points.shape}", file=sys.stderr)
		for model_name in args.models:
			frame = evaluate(points, labels, model_name, args.sites_per_class)
			out_path = args.output_dir / f"margins_{args.space}_layer{layer}_{model_name}.csv"
			frame.to_csv(out_path, index=False)
			accuracy = float(frame["correct"].mean())
			summary_rows.append({
				"space": args.space,
				"layer": layer,
				"model": model_name,
				"n_test": len(frame),
				"accuracy": accuracy,
				"mean_margin": float(frame["margin"].mean()),
			})
			print(f"  {model_name}: accuracy {accuracy:.4f} mean_margin {frame['margin'].mean():.4g}", file=sys.stderr)

	summary = pd.DataFrame(summary_rows)
	summary_path = args.output_dir / "boundary_summary.csv"
	summary.to_csv(summary_path, index=False)
	print(f"wrote {summary_path} ({len(summary)} rows)", file=sys.stderr)
	print(summary.to_string(index=False))


if __name__ == "__main__":
	main()
