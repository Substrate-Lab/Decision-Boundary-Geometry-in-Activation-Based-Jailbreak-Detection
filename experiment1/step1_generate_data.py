#!/usr/bin/env python3
"""Generate synthetic 2D Gaussian data for two classes across two covariance strata.

	python step1_generate_data.py
	python step1_generate_data.py --data-dir ./data --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MU_A = [-2.0, 0.0]
MU_B = [2.0, 0.0]
STRATA = {"low": 1, "high": 10}
N_TRAIN_PER_CLASS = 5000
N_TEST_PER_CLASS = 5000
GRID = {"min": -10.0, "max": 10.0, "resolution": 0.01}
RANDOM_SEED = 0
CLASS_PRIORS = {"A": 0.5, "B": 0.5}
CLASS_MEANINGS = {"A": "Refusal", "B": "Jailbreak"}

STRATUM_OFFSETS = {"low": 0, "high": 1}
SPLIT_OFFSETS = {"train": 0, "test": 1}
CLASS_OFFSETS = {"A": 0, "B": 1}


# Derive a distinct, deterministic sub-seed for one (stratum, split, class) draw.
def make_sub_seed(base_seed: int, stratum: str, split: str, class_label: str) -> int:
	offset = (
		STRATUM_OFFSETS[stratum] * 100
		+ SPLIT_OFFSETS[split] * 10
		+ CLASS_OFFSETS[class_label]
	)
	return base_seed * 1000 + offset


# Sample n points from a 2D Gaussian given mean, covariance, and a numpy Generator.
def sample_gaussian(mean, covariance, n_per_class: int, generator: np.random.Generator):
	return generator.multivariate_normal(mean, covariance, size=n_per_class)


# Build a labeled DataFrame (x, y, label) for one stratum+split by sampling both classes.
def build_split_frame(stratum: str, split: str, k: int, n_per_class: int, base_seed: int) -> pd.DataFrame:
	covariance_a = np.eye(2)
	covariance_b = k * np.eye(2)
	generator_a = np.random.default_rng(make_sub_seed(base_seed, stratum, split, "A"))
	generator_b = np.random.default_rng(make_sub_seed(base_seed, stratum, split, "B"))
	points_a = sample_gaussian(MU_A, covariance_a, n_per_class, generator_a)
	points_b = sample_gaussian(MU_B, covariance_b, n_per_class, generator_b)
	frame_a = pd.DataFrame({"x": points_a[:, 0], "y": points_a[:, 1], "label": "A"})
	frame_b = pd.DataFrame({"x": points_b[:, 0], "y": points_b[:, 1], "label": "B"})
	return pd.concat([frame_a, frame_b], ignore_index=True)


# Write the self-describing params.json capturing every shared parameter.
def write_params(data_dir: Path, base_seed: int) -> None:
	params = {
		"mu_A": MU_A,
		"mu_B": MU_B,
		"Sigma_A": {"form": "k * identity", "k": 1},
		"Sigma_B": {"form": "k * identity", "k": "stratum-dependent"},
		"strata": STRATA,
		"n_train_per_class": N_TRAIN_PER_CLASS,
		"n_test_per_class": N_TEST_PER_CLASS,
		"grid": GRID,
		"random_seed": base_seed,
		"class_priors": CLASS_PRIORS,
		"class_meanings": CLASS_MEANINGS,
	}
	path = data_dir / "params.json"
	with path.open("w", encoding="utf-8") as handle:
		json.dump(params, handle, indent=2)
		handle.write("\n")
	print(f"  wrote {path}", file=sys.stderr)


# Generate and write all four CSVs and params.json into the data directory.
def generate_all(data_dir: Path, base_seed: int) -> None:
	data_dir.mkdir(parents=True, exist_ok=True)
	splits = {"train": N_TRAIN_PER_CLASS, "test": N_TEST_PER_CLASS}
	for stratum, k in STRATA.items():
		for split, n_per_class in splits.items():
			frame = build_split_frame(stratum, split, k, n_per_class, base_seed)
			path = data_dir / f"stratum_{stratum}_{split}.csv"
			frame.to_csv(path, index=False)
			print(f"  wrote {path} ({len(frame)} rows)", file=sys.stderr)
	write_params(data_dir, base_seed)


# Parse arguments, then generate the synthetic data and parameter file.
def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument(
		"--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
		help="Directory to write CSVs and params.json (default: ./data next to this script).",
	)
	parser.add_argument(
		"--seed", type=int, default=RANDOM_SEED,
		help="Base random seed (default: 0).",
	)
	args = parser.parse_args()

	print(f"Generating synthetic data into {args.data_dir} ...", file=sys.stderr)
	generate_all(args.data_dir, args.seed)


if __name__ == "__main__":
	main()
