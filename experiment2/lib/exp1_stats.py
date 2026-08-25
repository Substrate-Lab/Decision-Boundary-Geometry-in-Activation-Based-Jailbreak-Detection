#!/usr/bin/env python3
"""Experiment 1's measured layer statistics, and clouds calibrated to them.

	The raw activations (experiment1/data/activations_layer*.npy) are gitignored and absent from
	a fresh clone, so nothing here plots real activations. What Experiment 1 *did* commit is its
	measured geometry, and that is enough to build a low-dimensional stand-in whose radial
	structure matches the real thing rather than being invented:

		analytics/run_*.json  activation_geometry[layer][class] -> mean_norm, std_norm, n
		layer_transition.csv  raw/pns covariance similarity, separation ratio, accuracy
		power_diagram.csv     the reported run: layer 30, 36 sites, accuracy, mean margin

	Every cloud produced here is synthetic and labelled as such wherever it is drawn. It is
	calibrated, not real: per-class norms and norm spreads are matched to the measured values,
	the classes are placed as separated directions, and the tangential spread is set to hold a
	prescribed radial share of the total variance. That makes the radial/tangential split of §3
	an explicit knob, which is what the spherical arm is supposed to act on.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP1_DATA = REPO_ROOT / "experiment1" / "data"
CLASS_NAMES = ["Refusal", "Jailbreak", "Benign"]


# Model ids are recorded as whatever was passed to --model, often a local path.
def short_model_name(model_id: str) -> str:
	return re.split(r"[\\/]", str(model_id).strip())[-1]


# Every analytics run present, oldest first, each tagged with its model.
def available_runs() -> list:
	runs = []
	for path in sorted((EXP1_DATA / "analytics").glob("run_*.json")):
		with path.open(encoding="utf-8") as handle:
			payload = json.load(handle)
		runs.append({
			"path": path,
			"run_id": payload.get("run_id", path.stem),
			"model_id": payload.get("model_id", ""),
			"model": short_model_name(payload.get("model_id", "")),
			"payload": payload,
		})
	return runs


# Model names Experiment 1 has actually collected, for error messages.
def available_models() -> list:
	return [run["model"] for run in available_runs()]


# The analytics run for a model, or the only run when no model is named.
def resolve_run(model: str | None = None) -> dict:
	runs = available_runs()
	if not runs:
		raise FileNotFoundError(
			f"no analytics runs under {EXP1_DATA / 'analytics'} -- run "
			f"experiment1/pipeline/step1_collect_activations.py first"
		)
	if model is None:
		return runs[-1]
	matches = [run for run in runs if model.lower() in run["model"].lower()]
	if not matches:
		raise SystemExit(
			f"Experiment 1 has no collected run for model {model!r}.\n"
			f"  available: {', '.join(available_models())}\n"
			f"  to add one: python experiment1/pipeline/step1_collect_activations.py "
			f"--model <hf-id-or-path>\n"
			f"  (that step needs a CUDA GPU; nothing here can synthesise it.)"
		)
	return matches[-1]


# The per-layer per-class norm statistics for a model's run.
def load_analytics(model: str | None = None) -> dict:
	return resolve_run(model)["payload"]


# The layer CSVs carry no model column, so they belong to whichever run wrote them.
# Serving them for a different model would silently mix two models' numbers.
def _guard_csv_model(model: str | None) -> dict:
	runs = available_runs()
	run = resolve_run(model)
	if len(runs) > 1:
		raise SystemExit(
			f"experiment1/data/*.csv (power_diagram, layer_transition, layer_diagnostics) "
			f"carry no model column, but {len(runs)} analytics runs are present "
			f"({', '.join(available_models())}). Those CSVs cannot be attributed to a model. "
			f"Re-generate them per model, or add a `model` column, before plotting "
			f"cross-model results."
		)
	return run


# The reported power-diagram run: which layer, how many sites, what it scored.
def load_reported_run(model: str | None = None) -> dict:
	run = _guard_csv_model(model)
	frame = pd.read_csv(EXP1_DATA / "power_diagram.csv")
	row = frame.iloc[0]
	return {
		"layer": int(row["layer"]),
		"n_sites": int(row["n_sites"]),
		"power_accuracy": float(row["power_accuracy"]),
		"mean_margin": float(row["mean_margin"]),
		"model": run["model"],
		"run_id": run["run_id"],
	}


# Per-class mean norm and norm spread at one layer, in fixed class order.
def norm_stats(layer: int, model: str | None = None) -> dict:
	geometry = load_analytics(model)["activation_geometry"][str(layer)]
	return {name: geometry[name] for name in CLASS_NAMES}


# Raw and PNS covariance similarity at one layer. Measured by Experiment 1, not asserted:
# PNS *raises* similarity at every layer, shrinking the very mismatch a covariance-aware
# boundary exploits. That is an empirical result from layer_transition.csv; the draft-v1
# plan has no section stating it (its §5.2 only lists PNS as one of the four arms).
def covariance_similarity(layer: int, model: str | None = None) -> dict:
	_guard_csv_model(model)
	frame = pd.read_csv(EXP1_DATA / "layer_transition.csv")
	row = frame.loc[frame["layer"] == layer].iloc[0]
	return {
		"raw": float(row["raw_covariance_similarity"]),
		"pns": float(row["pns_covariance_similarity"]),
		"raw_accuracy": float(row["raw_accuracy"]),
		"pns_accuracy": float(row["pns_accuracy"]),
	}


# Unit direction per class: equally spaced in 2D, mutually separated on the sphere in 3D.
def class_directions(dims: int, separation: float) -> np.ndarray:
	if dims == 2:
		angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]) * separation
		return np.column_stack([np.cos(angles), np.sin(angles)])
	tilt = separation
	directions = []
	for index in range(len(CLASS_NAMES)):
		azimuth = 2.0 * np.pi * index / len(CLASS_NAMES)
		directions.append([
			np.sin(tilt) * np.cos(azimuth),
			np.sin(tilt) * np.sin(azimuth),
			np.cos(tilt),
		])
	return np.array(directions)


# A synthetic cloud whose radius matches Experiment 1 and whose tangential spread is set
# so the requested share of each class's variance sits in the radial direction.
def calibrated_cloud(
	layer: int,
	model: str | None = None,
	n_per_class: int = 500,
	dims: int = 2,
	radial_share: float = 0.35,
	separation: float = 1.0,
	seed: int = 0,
):
	rng = np.random.default_rng(seed)
	stats = norm_stats(layer, model)
	directions = class_directions(dims, separation)

	points = []
	labels = []
	for index, name in enumerate(CLASS_NAMES):
		mean_norm = stats[name]["mean_norm"]
		std_norm = stats[name]["std_norm"]
		# Split total variance: std_norm is the radial part, and holding it to `radial_share`
		# of the total fixes the tangential standard deviation.
		tangential_std = std_norm * np.sqrt((1.0 - radial_share) / radial_share)

		axis = directions[index]
		basis = _orthonormal_complement(axis)
		radial = rng.normal(mean_norm, std_norm, size=n_per_class)
		tangential = rng.normal(0.0, tangential_std, size=(n_per_class, dims - 1))
		# Set the direction from the tangential offset, then scale it to the sampled radius,
		# so ||x|| is exactly the sampled norm. Adding the tangential part to a radial vector
		# instead would inflate ||x|| above the measured mean by the tangential contribution.
		offset = mean_norm * axis[None, :] + tangential @ basis
		direction = offset / np.linalg.norm(offset, axis=1, keepdims=True)
		points.append(radial[:, None] * direction)
		labels.append(np.full(n_per_class, index))

	return np.vstack(points), np.concatenate(labels)


# An orthonormal basis for the subspace orthogonal to `axis`, as rows.
def _orthonormal_complement(axis: np.ndarray) -> np.ndarray:
	completed, _ = np.linalg.qr(np.column_stack([axis, np.eye(axis.shape[0])]))
	return completed[:, 1:].T
