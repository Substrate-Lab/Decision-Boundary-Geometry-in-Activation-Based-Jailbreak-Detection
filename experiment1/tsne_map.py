#!/usr/bin/env python3
"""t-SNE visualization of layer activations, colored by class.

	A qualitative 2D embedding (PCA to 50 dims, then t-SNE) to *show* whether the classes
	separate. By default it plots the detection-relevant pair, Benign vs Jailbreak. t-SNE is
	view-only: it warps distances and densities, so do NOT fit boundaries or measure
	covariance in this space -- keep the quantitative geometry in raw/PCA (see power_diagram.py).

	python tsne_map.py
	python tsne_map.py --layers 24 30 34 --classes Benign Jailbreak Refusal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import CLASS_COLORS, load_labels

DEFAULT_LAYERS = [8, 16, 24, 30, 36]
DEFAULT_CLASSES = ["Benign", "Jailbreak"]
PCA_PREDIMS = 50
SEED = 0

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


# Embed activations into 2D via PCA-then-t-SNE.
def tsne_embed(points: np.ndarray) -> np.ndarray:
	dims = min(PCA_PREDIMS, points.shape[1], points.shape[0])
	reduced = PCA(n_components=dims, random_state=SEED).fit_transform(points)
	tsne = TSNE(n_components=2, perplexity=30.0, init="pca", learning_rate="auto", random_state=SEED)
	return tsne.fit_transform(reduced)


# Compute and save a t-SNE scatter for one layer, restricted to the selected classes.
def process_layer(layer: int, labels: np.ndarray, classes: list[str], data_dir: Path, figures_dir: Path) -> None:
	print(f"embedding layer {layer}", file=sys.stderr)
	activations = np.load(data_dir / f"activations_layer{layer}.npy")
	keep = np.isin(labels, classes)
	embedding = tsne_embed(activations[keep])
	kept_labels = labels[keep]

	fig, ax = plt.subplots(figsize=(6.8, 6.2))
	for name in classes:
		mask = kept_labels == name
		if mask.any():
			ax.scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.7, color=CLASS_COLORS.get(name, "#555555"), label=name)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlabel("t-SNE 1")
	ax.set_ylabel("t-SNE 2")
	ax.set_title(f"t-SNE of layer {layer} activations ({' vs '.join(classes)})")
	ax.legend(loc="upper right", fontsize=9)
	fig.tight_layout()
	out_path = figures_dir / f"tsne_layer{layer}_{'_'.join(classes)}.png"
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"wrote {out_path}", file=sys.stderr)


# Parse arguments and render a t-SNE map for each requested layer.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
	parser.add_argument("--classes", type=str, nargs="+", default=DEFAULT_CLASSES)
	parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
	parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
	args = parser.parse_args()

	labels = load_labels(args.data_dir)
	args.figures_dir.mkdir(parents=True, exist_ok=True)
	for layer in args.layers:
		process_layer(layer, labels, args.classes, args.data_dir, args.figures_dir)


if __name__ == "__main__":
	main()
