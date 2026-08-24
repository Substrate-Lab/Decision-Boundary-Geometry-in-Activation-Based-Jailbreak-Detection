#!/usr/bin/env python3
"""Shared helpers for the Experiment 1 analysis and visualization scripts.

	One place for the class names, class colors, label loading, and layer resolution that the
	per-layer scripts all need, so they stay in sync instead of each keeping its own copy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = ["Refusal", "Jailbreak", "Benign"]
CLASS_COLORS = {
	"Refusal": "#1ecb96",
	"Jailbreak": "#e5484d",
	"Benign": "#f5c518",
}


# Load labels.csv in row_index order as a DataFrame.
def load_labels_frame(data_dir: Path) -> pd.DataFrame:
	frame = pd.read_csv(data_dir / "labels.csv")
	if "row_index" in frame.columns:
		frame = frame.sort_values("row_index").reset_index(drop=True)
	return frame


# Load the class labels in row_index order.
def load_labels(data_dir: Path) -> np.ndarray:
	return load_labels_frame(data_dir)["class_label"].to_numpy()


# Use the requested layers, or fall back to the layers step1 recorded in collection_meta.json.
def resolve_layers(data_dir: Path, requested=None):
	if requested:
		return requested
	meta_path = data_dir / "collection_meta.json"
	if meta_path.exists():
		with meta_path.open(encoding="utf-8") as handle:
			return json.load(handle).get("layers", [])
	return []
