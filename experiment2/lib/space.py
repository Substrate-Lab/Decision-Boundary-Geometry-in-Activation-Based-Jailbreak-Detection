#!/usr/bin/env python3
"""Load the working representation for a layer, shared across the Experiment 2 steps.

	Experiment 2 reuses the activations collected in Experiment 1. Two working spaces are
	supported: "rawpca" reduces the raw last-token activations with PCA, and "pns" loads the
	Principal Nested Spheres scores written by Experiment 1 step 2. The tension is that PNS
	raises covariance similarity (shrinking the mismatch the boundary exploits), so both spaces
	are kept selectable to measure the effect rather than assume one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SEED = 0
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = EXPERIMENT_DIR.parent / "experiment1" / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "data"
FIGURES_DIR = EXPERIMENT_DIR / "figures"


# Load labels.csv in row_index order and return the class labels.
def load_labels(data_dir: Path) -> np.ndarray:
	frame = pd.read_csv(data_dir / "labels.csv")
	if "row_index" in frame.columns:
		frame = frame.sort_values("row_index").reset_index(drop=True)
	return frame["class_label"].to_numpy()


# Use the requested layers, or fall back to the layers recorded in collection_meta.json.
def resolve_layers(data_dir: Path, requested):
	if requested:
		return requested
	meta_path = data_dir / "collection_meta.json"
	if meta_path.exists():
		with meta_path.open(encoding="utf-8") as handle:
			return json.load(handle).get("layers", [])
	return []


# Return the working representation for one layer in the chosen space.
def working_space(layer: int, data_dir: Path, space: str, pca_dims: int) -> np.ndarray:
	if space == "pns":
		return np.load(data_dir / f"pns_layer{layer}.npy")
	if space == "rawpca":
		activations = np.load(data_dir / f"activations_layer{layer}.npy")
		dims = min(pca_dims, activations.shape[1], activations.shape[0])
		return PCA(n_components=dims, random_state=SEED).fit_transform(activations)
	raise ValueError(f"unknown space: {space}")
