#!/usr/bin/env python3
"""Per-run analytics for Experiment 1 collection runs (Unwrapping the Geometry).

	A small shared module the collection steps call to time named stages, capture the
	environment, summarise the label rows and a first-look activation geometry, and write
	a per-run JSON report plus an appended row into a flat runs_index.csv so runs stay
	comparable over time.

	from analytics import (
		timed, capture_environment, gpu_memory_stats, summarize_labels,
		activation_geometry, write_run_report, print_summary,
	)
"""
from __future__ import annotations

import csv
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

CLASS_NAMES = ["Refusal", "Jailbreak", "Benign"]


# Collect elapsed seconds for named stages into one timings dict.
@dataclass
class RunTimer:
	timings: dict = field(default_factory=dict)

	# Time a named stage and store its elapsed seconds under stage_name.
	@contextmanager
	def timed(self, stage_name: str):
		start = perf_counter()
		try:
			yield
		finally:
			self.timings[stage_name] = perf_counter() - start


# Time a named stage, storing its elapsed seconds into the given timings dict.
@contextmanager
def timed(timings: dict, stage_name: str):
	start = perf_counter()
	try:
		yield
	finally:
		timings[stage_name] = perf_counter() - start


# Coerce a value to a finite native float, mapping non-finite or missing to None.
def _safe_float(value) -> float | None:
	if value is None:
		return None
	number = float(value)
	if not np.isfinite(number):
		return None
	return number


# Compute a JSON-safe rate as numerator/denominator, guarding divide-by-zero.
def _rate(numerator: int, denominator: int) -> float | None:
	if denominator <= 0:
		return None
	return float(numerator) / float(denominator)


# Return p50/p90/max/mean of a list of numbers as JSON-safe native floats.
def _length_stats(values: list) -> dict:
	if not values:
		return {"p50": None, "p90": None, "max": None, "mean": None}
	array = np.asarray(values, dtype=np.float64)
	return {
		"p50": _safe_float(np.percentile(array, 50)),
		"p90": _safe_float(np.percentile(array, 90)),
		"max": _safe_float(array.max()),
		"mean": _safe_float(array.mean()),
	}


# Capture python/torch/transformers versions and GPU facts, guarding every optional import.
def capture_environment() -> dict:
	environment: dict = {
		"python_version": sys.version.split()[0],
		"torch_version": None,
		"transformers_version": None,
		"cuda_available": False,
		"gpu_name": None,
		"gpu_capability": None,
		"total_vram_gb": None,
	}
	try:
		import torch

		environment["torch_version"] = torch.__version__
		if torch.cuda.is_available():
			environment["cuda_available"] = True
			properties = torch.cuda.get_device_properties(0)
			environment["gpu_name"] = torch.cuda.get_device_name(0)
			environment["gpu_capability"] = f"{properties.major}.{properties.minor}"
			environment["total_vram_gb"] = _safe_float(properties.total_memory / (1024 ** 3))
	except Exception as error:
		print(f"analytics: could not read torch environment: {error}", file=sys.stderr)
	try:
		import transformers

		environment["transformers_version"] = transformers.__version__
	except Exception:
		pass
	return environment


# Report peak allocated/reserved GPU memory in GB if torch+cuda are available, else zeros.
def gpu_memory_stats() -> dict:
	stats = {"peak_alloc_gb": 0.0, "peak_reserved_gb": 0.0}
	try:
		import torch

		if torch.cuda.is_available():
			stats["peak_alloc_gb"] = _safe_float(torch.cuda.max_memory_allocated() / (1024 ** 3))
			stats["peak_reserved_gb"] = _safe_float(torch.cuda.max_memory_reserved() / (1024 ** 3))
	except Exception as error:
		print(f"analytics: could not read gpu memory stats: {error}", file=sys.stderr)
	return stats


# Summarise step1 label rows into class balance, per-dataset crosstabs, refusal rates, and lengths.
def summarize_labels(label_rows: list) -> dict:
	n_total = len(label_rows)
	class_counts = {name: 0 for name in CLASS_NAMES}
	per_dataset_counts: dict = {}
	dataset_by_class: dict = {}
	refused_by_dataset: dict = {}
	total_by_dataset: dict = {}
	refused_by_category: dict = {}
	total_by_category: dict = {}
	n_harmful_prompts = 0
	n_benign_prompts = 0
	n_refused = 0
	n_refused_harmful = 0
	prompt_lengths: list = []
	response_lengths: list = []

	for row in label_rows:
		label = row.get("class_label")
		if label in class_counts:
			class_counts[label] += 1
		dataset = row.get("dataset")
		category = row.get("source_category")
		harmful = bool(row.get("is_harmful_prompt"))
		refused = bool(row.get("refused"))

		per_dataset_counts[dataset] = per_dataset_counts.get(dataset, 0) + 1
		if dataset not in dataset_by_class:
			dataset_by_class[dataset] = {name: 0 for name in CLASS_NAMES}
		if label in dataset_by_class[dataset]:
			dataset_by_class[dataset][label] += 1

		total_by_dataset[dataset] = total_by_dataset.get(dataset, 0) + 1
		total_by_category[category] = total_by_category.get(category, 0) + 1
		if refused:
			n_refused += 1
			refused_by_dataset[dataset] = refused_by_dataset.get(dataset, 0) + 1
			refused_by_category[category] = refused_by_category.get(category, 0) + 1
		if harmful:
			n_harmful_prompts += 1
			if refused:
				n_refused_harmful += 1
		else:
			n_benign_prompts += 1

		prompt_lengths.append(len(str(row.get("prompt", ""))))
		response_lengths.append(len(str(row.get("response", ""))))

	refusal_rate_by_dataset = {
		dataset: _rate(refused_by_dataset.get(dataset, 0), total_by_dataset[dataset])
		for dataset in total_by_dataset
	}
	refusal_rate_by_source_category = {
		category: _rate(refused_by_category.get(category, 0), total_by_category[category])
		for category in total_by_category
	}

	return {
		"n_total": n_total,
		"class_counts": class_counts,
		"n_harmful_prompts": n_harmful_prompts,
		"n_benign_prompts": n_benign_prompts,
		"per_dataset_counts": per_dataset_counts,
		"dataset_by_class": dataset_by_class,
		"refusal_rate_overall": _rate(n_refused, n_total),
		"refusal_rate_harmful": _rate(n_refused_harmful, n_harmful_prompts),
		"refusal_rate_by_dataset": refusal_rate_by_dataset,
		"refusal_rate_by_source_category": refusal_rate_by_source_category,
		"jailbreak_compliance_rate": _rate(class_counts["Jailbreak"], n_harmful_prompts),
		"prompt_char_len": _length_stats(prompt_lengths),
		"response_char_len": _length_stats(response_lengths),
	}


# Measure the per-class mean/std of activation L2 norms at each layer as a tight-vs-diffuse signal.
def activation_geometry(activations_by_layer: dict, label_rows: list) -> dict:
	labels = np.asarray([row.get("class_label") for row in label_rows])
	geometry: dict = {}
	for layer, vectors in activations_by_layer.items():
		array = np.asarray(vectors, dtype=np.float64)
		layer_key = str(layer)
		geometry[layer_key] = {}
		if array.ndim != 2 or array.shape[0] == 0:
			for name in CLASS_NAMES:
				geometry[layer_key][name] = {"mean_norm": None, "std_norm": None, "n": 0}
			continue
		norms = np.linalg.norm(array, axis=1)
		aligned = labels[: norms.shape[0]]
		for name in CLASS_NAMES:
			mask = aligned == name
			count = int(mask.sum())
			if count == 0:
				geometry[layer_key][name] = {"mean_norm": None, "std_norm": None, "n": 0}
				continue
			class_norms = norms[mask]
			geometry[layer_key][name] = {
				"mean_norm": _safe_float(class_norms.mean()),
				"std_norm": _safe_float(class_norms.std()),
				"n": count,
			}
	return geometry


# Sum the per-stage timings in a run record into a total seconds value.
def _seconds_total(run_record: dict) -> float:
	timings = run_record.get("timings") or {}
	total = 0.0
	for value in timings.values():
		number = _safe_float(value)
		if number is not None:
			total += number
	return total


# Write the pretty per-run JSON report and append a flat row to runs_index.csv.
def write_run_report(data_dir: Path, run_record: dict) -> Path:
	analytics_dir = Path(data_dir) / "analytics"
	analytics_dir.mkdir(parents=True, exist_ok=True)

	run_id = run_record.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	json_path = analytics_dir / f"run_{run_id}.json"
	with json_path.open("w", encoding="utf-8") as handle:
		json.dump(run_record, handle, indent=2, default=str)
	print(f"wrote {json_path}", file=sys.stderr)

	labels_summary = run_record.get("labels_summary") or {}
	class_counts = labels_summary.get("class_counts") or {}
	environment = run_record.get("environment") or {}
	gpu = run_record.get("gpu") or {}

	row = {
		"run_id": run_id,
		"model_id": run_record.get("model_id"),
		"sample_per_class": run_record.get("sample_per_class"),
		"load_in_4bit": run_record.get("load_in_4bit"),
		"n_total": labels_summary.get("n_total"),
		"n_Refusal": class_counts.get("Refusal"),
		"n_Jailbreak": class_counts.get("Jailbreak"),
		"n_Benign": class_counts.get("Benign"),
		"refusal_rate_harmful": labels_summary.get("refusal_rate_harmful"),
		"jailbreak_compliance_rate": labels_summary.get("jailbreak_compliance_rate"),
		"over_refusal_skipped": run_record.get("over_refusal_skipped"),
		"seconds_total": _seconds_total(run_record),
		"gpu_peak_alloc_gb": gpu.get("peak_alloc_gb"),
		"gpu_name": environment.get("gpu_name"),
	}
	fieldnames = list(row.keys())
	index_path = analytics_dir / "runs_index.csv"
	write_header = not index_path.exists()
	with index_path.open("a", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		if write_header:
			writer.writeheader()
		writer.writerow(row)
	print(f"appended run row to {index_path}", file=sys.stderr)
	return json_path


# Print a short readable digest of the run record to stderr.
def print_summary(run_record: dict) -> None:
	labels_summary = run_record.get("labels_summary") or {}
	class_counts = labels_summary.get("class_counts") or {}
	timings = run_record.get("timings") or {}
	gpu = run_record.get("gpu") or {}

	print("=== run analytics ===", file=sys.stderr)
	print(f"run_id: {run_record.get('run_id')}", file=sys.stderr)
	print(f"model: {run_record.get('model_id')}", file=sys.stderr)
	print(
		f"classes: Refusal={class_counts.get('Refusal', 0)} "
		f"Jailbreak={class_counts.get('Jailbreak', 0)} "
		f"Benign={class_counts.get('Benign', 0)} "
		f"(n_total={labels_summary.get('n_total', 0)})",
		file=sys.stderr,
	)
	print(
		f"refusal_rate_harmful={labels_summary.get('refusal_rate_harmful')} "
		f"jailbreak_compliance_rate={labels_summary.get('jailbreak_compliance_rate')}",
		file=sys.stderr,
	)
	print(f"over_refusal_skipped: {run_record.get('over_refusal_skipped')}", file=sys.stderr)
	for stage, seconds in timings.items():
		number = _safe_float(seconds)
		shown = f"{number:.1f}s" if number is not None else "n/a"
		print(f"  stage {stage}: {shown}", file=sys.stderr)
	print(f"total: {_seconds_total(run_record):.1f}s", file=sys.stderr)
	print(f"gpu peak alloc: {gpu.get('peak_alloc_gb')} GB", file=sys.stderr)
