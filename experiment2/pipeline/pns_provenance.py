#!/usr/bin/env python3
"""Provenance for every PNS array the Tier 1 numbers were computed from.

	dispersion_ratio.csv reports rho, D_B and the trace ratios. This says where each of those
	numbers came from: which file, how big, what it hashes to, and -- the part worth checking --
	whether the per-class moments recomputed here reproduce the moments already committed in
	sites_layer*.npz.

	That last column is the real provenance signal. The committed sites were fitted with the
	empirical covariance, not Ledoit-Wolf; recomputing with a shrunk estimator would have
	produced plausible-looking numbers that quietly disagreed with the fitted sites. sites_match
	being True means the moments behind rho are the same moments already in the repo.

	Writes experiment2/data/pns_provenance.csv.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from experiment2.pipeline.step4_dispersion_ratio import available_layers, discover_models

CLASS_NAMES = ["Refusal", "Jailbreak", "Benign"]


# First 12 hex characters of the file's SHA-256, enough to pin a file without the full digest.
def short_hash(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1 << 20), b""):
			digest.update(chunk)
	return digest.hexdigest()[:12]


# Do the moments recomputed from the scores reproduce the committed sites for this layer?
def sites_agreement(directory: Path, layer: int, points: np.ndarray, labels: np.ndarray):
	path = directory / f"sites_layer{layer}.npz"
	if not path.exists():
		# Not a mismatch — there is simply no committed reference for this layer. Recording it
		# as such keeps "unverified" from being read as "verified".
		return {"sites_file": "", "sites_sha256": "", "sites_match": "no_sites_file"}
	bundle = np.load(path, allow_pickle=True)
	agrees = True
	for name in CLASS_NAMES:
		rows = points[labels == name]
		agrees &= np.allclose(rows.mean(axis=0), bundle[f"mean_{name}"], atol=1e-9)
		agrees &= np.allclose(np.cov(rows, rowvar=False), bundle[f"cov_{name}"], atol=1e-9)
	return {
		"sites_file": str(path.relative_to(REPO_ROOT)),
		"sites_sha256": short_hash(path),
		"sites_match": "matched" if agrees else "MISMATCH",
	}


def main() -> int:
	rows = []
	for entry in discover_models():
		directory, model = entry["dir"], entry["model"]
		meta_path = directory / "collection_meta.json"
		meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
		labels_path = directory / "labels.csv"
		labels = pd.read_csv(labels_path)["class_label"].to_numpy()
		counts = {name: int(np.sum(labels == name)) for name in CLASS_NAMES}
		analytics = sorted((directory / "analytics").glob("run_*.json"))
		run_id = ""
		if analytics:
			run_id = json.loads(analytics[-1].read_text()).get("run_id", "")

		for layer in available_layers(directory):
			pns_path = directory / f"pns_layer{layer}.npy"
			points = np.load(pns_path)
			row = {
				"model": model,
				"model_id": meta.get("model_id", ""),
				"run_id": run_id,
				"hidden_dim_raw": meta.get("hidden_dim", ""),
				"layer": layer,
				"frame": "pns",
				"pns_file": str(pns_path.relative_to(REPO_ROOT)),
				"pns_sha256": short_hash(pns_path),
				"n_rows": int(points.shape[0]),
				"n_dims": int(points.shape[1]),
				"dtype": str(points.dtype),
				"bytes": pns_path.stat().st_size,
				"labels_file": str(labels_path.relative_to(REPO_ROOT)),
				"labels_sha256": short_hash(labels_path),
				"n_Refusal": counts["Refusal"],
				"n_Jailbreak": counts["Jailbreak"],
				"n_Benign": counts["Benign"],
				"covariance_estimator": "empirical (np.cov)",
				"rawpca_available": bool(any(directory.glob("activations_layer*.npy"))),
				"consumed_by": "experiment2/pipeline/step4_dispersion_ratio.py",
				"results_in": "experiment2/data/dispersion_ratio.csv",
			}
			row.update(sites_agreement(directory, layer, points, labels))
			rows.append(row)

	frame = pd.DataFrame(rows)
	destination = REPO_ROOT / "experiment2" / "data" / "pns_provenance.csv"
	frame.to_csv(destination, index=False)

	print(f"{len(frame)} PNS arrays across {frame['model'].nunique()} models\n")
	summary = frame.groupby("model").agg(
		layers=("layer", "count"),
		dims=("n_dims", "first"),
		rows=("n_rows", "first"),
		raw_hidden=("hidden_dim_raw", "first"),
		sites_verified=("sites_match", lambda s: int((s == "matched").sum())),
		rawpca=("rawpca_available", "first"),
	)
	print(summary.to_string())
	matched = int((frame["sites_match"] == "matched").sum())
	missing = int((frame["sites_match"] == "no_sites_file").sum())
	mismatched = frame[frame["sites_match"] == "MISMATCH"]
	print("\nmoments vs committed sites_layer*.npz:")
	print(f"  matched          {matched:>3}/{len(frame)}")
	print(f"  no reference     {missing:>3}/{len(frame)}  (no sites_layer*.npz for that layer —")
	print("                        unverified, not verified)")
	print(f"  MISMATCHED       {len(mismatched):>3}/{len(frame)}")
	for _, row in mismatched.iterrows():
		print(f"    {row['model']} layer {row['layer']} — numbers do not trace to committed sites")
	print(f"\nwrote {destination.relative_to(REPO_ROOT)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
