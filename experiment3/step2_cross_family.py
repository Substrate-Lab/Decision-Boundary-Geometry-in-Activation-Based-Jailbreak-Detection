#!/usr/bin/env python3
"""Cross-family generalization: train on some source datasets, test on a held-out one.

	Step 2 of Experiment 3. The deployment condition is an unseen jailbreak family, so this leaves
	one source dataset out at a time, trains each detector on the other three, and measures the
	AUC on the held-out family. The in-distribution AUC (k-fold on the training datasets) is the
	reference, and the drop to the held-out AUC is the generalization gap. The paper's claim is
	that the covariance-aware geometry degrades more gracefully than a discriminative linear probe;
	the auc_drop column is where that claim is checked.

	python step2_cross_family.py
	python step2_cross_family.py --layers 24 28 30 34 --positive Jailbreak --negative Benign
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment2"))
from space import DEFAULT_DATA_DIR, resolve_layers, working_space

from detectors import build_detector, default_detectors

SEED = 0
FOLDS = 5
HORIZON_LAYERS = [24, 28, 30, 34]

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# Load labels.csv in row_index order and return class labels and source datasets.
def load_labels_and_source(data_dir: Path):
	frame = pd.read_csv(data_dir / "labels.csv")
	if "row_index" in frame.columns:
		frame = frame.sort_values("row_index").reset_index(drop=True)
	return frame["class_label"].to_numpy(), frame["dataset"].to_numpy()


# In-distribution k-fold AUC of one detector on the training rows.
def indist_auc(points, labels, detector_name, positive, negative, sites_per_class):
	folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
	target = (labels == positive).astype(int)
	scores = []
	for train_index, test_index in folds.split(points, target):
		detector = build_detector(detector_name, positive, negative, sites_per_class=sites_per_class)
		detector.fit(points[train_index], labels[train_index])
		scores.append(roc_auc_score(target[test_index], detector.score(points[test_index])))
	return float(np.mean(scores))


# Held-out-family AUC of one detector trained on the training rows.
def ood_auc(train_points, train_labels, test_points, test_labels, detector_name, positive, negative, sites_per_class):
	detector = build_detector(detector_name, positive, negative, sites_per_class=sites_per_class)
	detector.fit(train_points, train_labels)
	target = (test_labels == positive).astype(int)
	return float(roc_auc_score(target, detector.score(test_points)))


# Draw the mean AUC drop per detector.
def plot_auc_drop(frame: pd.DataFrame, positive: str, negative: str, out_path: Path) -> None:
	means = frame.groupby("detector")["auc_drop"].mean().sort_values()
	fig, ax = plt.subplots(figsize=(7.5, 5.0))
	ax.bar(means.index, means.values, color="#4a5568")
	ax.set_ylabel("mean AUC drop (in-dist - held-out)")
	ax.set_title(f"Cross-family generalization gap: {positive} vs {negative}")
	ax.tick_params(axis="x", rotation=30)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments, run leave-one-dataset-out per layer, and write the results.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--detectors", nargs="+", default=default_detectors())
	parser.add_argument("--positive", default="Jailbreak")
	parser.add_argument("--negative", default="Benign")
	parser.add_argument("--space", choices=["rawpca", "pns"], default="rawpca")
	parser.add_argument("--layers", type=int, nargs="+", default=HORIZON_LAYERS)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--sites-per-class", type=int, default=12)
	parser.add_argument("--min-test", type=int, default=20)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels, source = load_labels_and_source(args.data_dir)
	keep = np.isin(labels, [args.positive, args.negative])
	datasets = sorted(np.unique(source[keep]).tolist())
	args.output_dir.mkdir(parents=True, exist_ok=True)
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	rows = []
	for layer in resolve_layers(args.data_dir, args.layers):
		full = working_space(layer, args.data_dir, args.space, args.pca_dims)
		points = full[keep]
		layer_labels = labels[keep]
		layer_source = source[keep]
		print(f"layer {layer}: {points.shape[0]} points", file=sys.stderr)
		for held_out in datasets:
			test_mask = layer_source == held_out
			train_mask = ~test_mask
			test_labels = layer_labels[test_mask]
			if min(np.sum(test_labels == args.positive), np.sum(test_labels == args.negative)) < args.min_test:
				print(f"  skip {held_out}: too few held-out points", file=sys.stderr)
				continue
			for name in args.detectors:
				reference = indist_auc(points[train_mask], layer_labels[train_mask], name, args.positive, args.negative, args.sites_per_class)
				held = ood_auc(points[train_mask], layer_labels[train_mask], points[test_mask], test_labels, name, args.positive, args.negative, args.sites_per_class)
				rows.append({
					"space": args.space,
					"layer": layer,
					"held_out": held_out,
					"detector": name,
					"indist_auc": reference,
					"ood_auc": held,
					"auc_drop": reference - held,
				})

	frame = pd.DataFrame(rows)
	out_path = args.output_dir / f"cross_family_{args.space}_{args.positive}_vs_{args.negative}.csv"
	frame.to_csv(out_path, index=False)
	print(f"wrote {out_path} ({len(frame)} rows)", file=sys.stderr)
	if not frame.empty:
		plot_auc_drop(frame, args.positive, args.negative, args.figures_dir / f"cross_family_{args.space}_{args.positive}_vs_{args.negative}.png")
		print(frame.groupby("detector")[["indist_auc", "ood_auc", "auc_drop"]].mean().round(4).to_string())


if __name__ == "__main__":
	main()
