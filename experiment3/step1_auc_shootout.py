#!/usr/bin/env python3
"""Per-layer ROC-AUC shootout: geometry detectors vs a linear probe.

	Step 1 of Experiment 3, and the core validating experiment (plan §7 caveat 2). For every layer
	it runs a stratified k-fold ROC-AUC of each detector on a binary detection task (default
	Jailbreak vs Benign, the hard detection-relevant pair) and plots AUC against layer. The claim
	to be tested is that the covariance-aware power diagram's advantage over the flat linear probe
	*widens* at the late-layer execution horizon (~L24-34); this figure is where that shows up or
	fails to.

	python step1_auc_shootout.py
	python step1_auc_shootout.py --positive Jailbreak --negative Benign --space rawpca
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
from space import DEFAULT_DATA_DIR, load_labels, resolve_layers, working_space

from detectors import build_detector, default_detectors

SEED = 0
FOLDS = 5

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# Mean and std cross-validated AUC for one detector on one layer's two-class data.
def cross_val_auc(points, labels, detector_name, positive, negative, sites_per_class):
	folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
	target = (labels == positive).astype(int)
	scores = []
	for train_index, test_index in folds.split(points, target):
		detector = build_detector(detector_name, positive, negative, sites_per_class=sites_per_class)
		detector.fit(points[train_index], labels[train_index])
		predicted = detector.score(points[test_index])
		scores.append(roc_auc_score(target[test_index], predicted))
	return float(np.mean(scores)), float(np.std(scores))


# Draw AUC against layer, one line per detector.
def plot_auc_by_layer(frame: pd.DataFrame, positive: str, negative: str, out_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(8.5, 5.5))
	for name in frame["detector"].unique():
		part = frame[frame["detector"] == name].sort_values("layer")
		ax.plot(part["layer"], part["auc"], marker="o", markersize=3, label=name)
	ax.axvspan(24, 34, color="#1ecb96", alpha=0.08)
	ax.set_xlabel("layer")
	ax.set_ylabel(f"ROC-AUC ({positive} vs {negative})")
	ax.set_title(f"Per-layer detection AUC: {positive} vs {negative}")
	ax.legend(loc="lower right", fontsize=8)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments, run the shootout per layer, and write the AUC table and figure.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--detectors", nargs="+", default=default_detectors())
	parser.add_argument("--positive", default="Jailbreak")
	parser.add_argument("--negative", default="Benign")
	parser.add_argument("--space", choices=["rawpca", "pns"], default="rawpca")
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--sites-per-class", type=int, default=12)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	keep = np.isin(labels, [args.positive, args.negative])
	args.output_dir.mkdir(parents=True, exist_ok=True)
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	rows = []
	for layer in resolve_layers(args.data_dir, args.layers):
		points = working_space(layer, args.data_dir, args.space, args.pca_dims)[keep]
		layer_labels = labels[keep]
		print(f"layer {layer}: {points.shape[0]} points", file=sys.stderr)
		for name in args.detectors:
			mean_auc, std_auc = cross_val_auc(points, layer_labels, name, args.positive, args.negative, args.sites_per_class)
			rows.append({"space": args.space, "layer": layer, "detector": name, "auc": mean_auc, "auc_std": std_auc})
			print(f"  {name}: auc {mean_auc:.4f} +/- {std_auc:.4f}", file=sys.stderr)

	frame = pd.DataFrame(rows)
	out_path = args.output_dir / f"auc_by_layer_{args.space}_{args.positive}_vs_{args.negative}.csv"
	frame.to_csv(out_path, index=False)
	print(f"wrote {out_path} ({len(frame)} rows)", file=sys.stderr)
	plot_auc_by_layer(frame, args.positive, args.negative, args.figures_dir / f"auc_by_layer_{args.space}_{args.positive}_vs_{args.negative}.png")


if __name__ == "__main__":
	main()
