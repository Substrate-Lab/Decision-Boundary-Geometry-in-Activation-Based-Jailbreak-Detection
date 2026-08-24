#!/usr/bin/env python3
"""Fractal diagnostic: covariance participation ratio per class per layer.

	Step 3 of Experiment 3. The effective (fractal) dimension of a class is the covariance
	participation ratio D = trace(Sigma)^2 / ||Sigma||_F^2 = (sum eigenvalues)^2 / sum(eigenvalues^2),
	the number of directions the class actually spreads across. Box-counting is deliberately not
	used: it fails in high dimension with few points. The hypothesis is that the diffuse class
	(Jailbreak) occupies many more effective directions than the stereotyped class (Refusal), so
	D_fractal(Jailbreak) should sit well above D_fractal(Refusal) across layers.

	python step3_fractal.py
	python step3_fractal.py --space rawpca --layers 24 30 34
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment2"))
from space import DEFAULT_DATA_DIR, load_labels, resolve_layers, working_space

CLASS_COLORS = {"Refusal": "#1ecb96", "Jailbreak": "#e5484d", "Benign": "#f5c518"}
CLASS_NAMES = ["Refusal", "Jailbreak", "Benign"]

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# Covariance participation ratio: (sum of eigenvalues)^2 over sum of squared eigenvalues.
def participation_ratio(rows: np.ndarray) -> float:
	if rows.shape[0] < 2:
		return float("nan")
	eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(rows, rowvar=False)), 0.0, None)
	denominator = float(np.sum(eigenvalues ** 2))
	if denominator <= 0:
		return float("nan")
	return float(np.sum(eigenvalues) ** 2 / denominator)


# Draw the fractal dimension of each class against layer.
def plot_fractal(frame: pd.DataFrame, out_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(8.0, 5.0))
	for name in CLASS_NAMES:
		part = frame[frame["class_label"] == name].sort_values("layer")
		if part.empty:
			continue
		ax.plot(part["layer"], part["d_fractal"], marker="o", markersize=3, color=CLASS_COLORS[name], label=name)
	ax.set_xlabel("layer")
	ax.set_ylabel("fractal dimension  trace(Sigma)^2 / ||Sigma||_F^2")
	ax.set_title("Effective dimension per class across layers")
	ax.legend(loc="upper left", fontsize=9)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments, compute the fractal dimension per class per layer, and write the results.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--space", choices=["rawpca", "pns"], default="rawpca")
	parser.add_argument("--layers", type=int, nargs="+", default=None)
	parser.add_argument("--pca-dims", type=int, default=10)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	rows = []
	for layer in resolve_layers(args.data_dir, args.layers):
		points = working_space(layer, args.data_dir, args.space, args.pca_dims)
		for name in CLASS_NAMES:
			mask = labels == name
			if not mask.any():
				continue
			rows.append({
				"space": args.space,
				"layer": layer,
				"class_label": name,
				"n": int(mask.sum()),
				"d_fractal": participation_ratio(points[mask]),
			})
		summary = ", ".join(f"{r['class_label']} {r['d_fractal']:.2f}" for r in rows if r["layer"] == layer)
		print(f"layer {layer}: {summary}", file=sys.stderr)

	frame = pd.DataFrame(rows)
	out_path = args.output_dir / f"fractal_{args.space}.csv"
	frame.to_csv(out_path, index=False)
	print(f"wrote {out_path} ({len(frame)} rows)", file=sys.stderr)
	plot_fractal(frame, args.figures_dir / f"fractal_{args.space}.png")


if __name__ == "__main__":
	main()
