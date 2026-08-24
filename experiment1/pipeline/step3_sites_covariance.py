#!/usr/bin/env python3
"""Build class sites and covariances in PNS space for Experiment 1 (Step 3).

	For each layer's Principal Nested Spheres (polar) embedding from step2, compute each
	class's site (mean vector) and full covariance matrix, then quantify spread to test the
	core hypothesis: Refusal is tight (small covariance), Jailbreak is diffuse (large
	covariance), Benign sits in between. Reads pns_layer{L}.npy + labels.csv, writes
	sites_layer{L}.npz, class_spread.csv, and per-layer trace-of-covariance bar charts.

	Dependencies: numpy, pandas, matplotlib (nothing is executed here).

	python step3_sites_covariance.py
	python step3_sites_covariance.py --layers 8 16 24
	python step3_sites_covariance.py --data-dir data --figures-dir figures
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import CLASS_NAMES, DATA_DIR, FIGURES_DIR, load_labels_frame, resolve_layers

TINY = 1e-12


# Per-class statistics computed in PNS coordinate space for one layer.
@dataclass
class ClassStats:
	class_label: str
	n: int
	mu: np.ndarray
	cov: np.ndarray
	trace_cov: float
	log_generalized_variance: float
	logdet_sign: float
	mean_norm: float
	well_conditioned: bool


# Load the PNS (polar) embedding array for one layer.
def load_pns_scores(layer: int, data_dir: Path) -> np.ndarray:
	scores_path = data_dir / f"pns_layer{layer}.npy"
	scores = np.load(scores_path)
	print(f"loaded {scores_path} shape {scores.shape}", file=sys.stderr)
	return scores


# Compute a stable log generalized variance (log-determinant) of a covariance matrix.
def stable_logdet(cov: np.ndarray) -> tuple[float, float]:
	sign, logdet = np.linalg.slogdet(cov)
	if sign > 0 and np.isfinite(logdet):
		return float(sign), float(logdet)
	eigenvalues = np.linalg.eigvalsh(cov)
	clipped = np.clip(eigenvalues, TINY, np.inf)
	return 1.0, float(np.sum(np.log(clipped)))


# Compute mean, full covariance, and spread metrics for the rows selected by mask.
def class_statistics(scores: np.ndarray, mask: np.ndarray, class_label: str) -> ClassStats:
	rows = scores[mask]
	n = int(rows.shape[0])
	n_dims = int(scores.shape[1])
	mu = rows.mean(axis=0) if n > 0 else np.zeros(n_dims, dtype=float)
	well_conditioned = n > n_dims
	if n >= 2:
		cov = np.cov(rows, rowvar=False)
		cov = np.atleast_2d(cov)
	else:
		cov = np.zeros((n_dims, n_dims), dtype=float)
	trace_cov = float(np.trace(cov))
	sign, logdet = stable_logdet(cov)
	mean_norm = float(np.linalg.norm(mu))
	return ClassStats(
		class_label=class_label,
		n=n,
		mu=mu,
		cov=cov,
		trace_cov=trace_cov,
		log_generalized_variance=logdet,
		logdet_sign=sign,
		mean_norm=mean_norm,
		well_conditioned=bool(well_conditioned),
	)


# Compute per-class sites and spread rows for one layer, returning the npz dict and csv rows.
def compute_layer(layer: int, labels: pd.DataFrame, data_dir: Path) -> tuple[dict, list[dict]]:
	scores = load_pns_scores(layer, data_dir)
	label_values = labels["class_label"].to_numpy()
	present = [name for name in CLASS_NAMES if np.any(label_values == name)]
	npz_dict: dict[str, np.ndarray] = {"class_order": np.array(present, dtype=object)}
	csv_rows: list[dict] = []
	for class_label in present:
		mask = label_values == class_label
		stats = class_statistics(scores, mask, class_label)
		npz_dict[f"mean_{class_label}"] = stats.mu
		npz_dict[f"cov_{class_label}"] = stats.cov
		csv_rows.append({
			"layer": layer,
			"class_label": stats.class_label,
			"n": stats.n,
			"trace_cov": stats.trace_cov,
			"log_generalized_variance": stats.log_generalized_variance,
			"mean_norm": stats.mean_norm,
			"well_conditioned": stats.well_conditioned,
		})
		print(
			f"  layer {layer} {class_label}: n={stats.n} trace={stats.trace_cov:.4g} "
			f"logGV={stats.log_generalized_variance:.4g} well_conditioned={stats.well_conditioned}",
			file=sys.stderr,
		)
	return npz_dict, csv_rows


# Save a bar chart of trace(covariance) by class for one layer.
def plot_class_spread(layer: int, rows: list[dict], figures_dir: Path) -> None:
	figures_dir.mkdir(parents=True, exist_ok=True)
	ordered = [row for name in CLASS_NAMES for row in rows if row["class_label"] == name]
	names = [row["class_label"] for row in ordered]
	traces = [row["trace_cov"] for row in ordered]
	colors = {"Refusal": "#2a7d4f", "Jailbreak": "#b23a48", "Benign": "#4a5568"}
	bar_colors = [colors.get(name, "#4a5568") for name in names]
	figure, axis = plt.subplots(figsize=(6, 4))
	axis.bar(names, traces, color=bar_colors)
	axis.set_ylabel("trace(covariance) in PNS space")
	axis.set_title(f"Class spread by trace of covariance (layer {layer})")
	for index, value in enumerate(traces):
		axis.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
	figure.tight_layout()
	out_path = figures_dir / f"class_spread_layer{layer}.png"
	figure.savefig(out_path, dpi=150)
	plt.close(figure)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments, compute sites and spread per layer, and write npz, csv, and figures.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"--layers", type=int, nargs="+", default=None,
		help="Layers to process (default: the layers recorded in collection_meta.json).",
	)
	parser.add_argument(
		"--data-dir", default=str(DATA_DIR),
		help="Directory holding labels.csv and pns_layer{L}.npy (default: %(default)s).",
	)
	parser.add_argument(
		"--figures-dir", default=str(FIGURES_DIR),
		help="Directory for output figures (default: %(default)s).",
	)
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	figures_dir = Path(args.figures_dir)

	labels = load_labels_frame(data_dir)
	all_rows: list[dict] = []
	for layer in resolve_layers(data_dir, args.layers):
		print(f"processing layer {layer}", file=sys.stderr)
		npz_dict, csv_rows = compute_layer(layer, labels, data_dir)
		npz_path = data_dir / f"sites_layer{layer}.npz"
		np.savez(npz_path, **npz_dict)
		print(f"wrote {npz_path}", file=sys.stderr)
		plot_class_spread(layer, csv_rows, figures_dir)
		all_rows.extend(csv_rows)

	spread_df = pd.DataFrame(all_rows, columns=[
		"layer", "class_label", "n", "trace_cov",
		"log_generalized_variance", "mean_norm", "well_conditioned",
	])
	spread_path = data_dir / "class_spread.csv"
	spread_df.to_csv(spread_path, index=False)
	print(f"wrote {spread_path} ({len(spread_df)} rows)", file=sys.stderr)
	print(spread_df.to_string(index=False))


if __name__ == "__main__":
	main()
