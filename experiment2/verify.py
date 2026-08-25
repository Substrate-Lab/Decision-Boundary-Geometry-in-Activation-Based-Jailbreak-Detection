#!/usr/bin/env python3
"""Self-check: re-derive every claim the Experiment 2 figures make, from source.

	The figures quote numbers (layer-30 norms, covariance similarity, the reported 36-site
	accuracy) and rely on mathematical identities (the Alexandrov gradient, the log/exp maps,
	the chordal/geodesic tie at w = 0). Both are places where a plausible-looking wrong value
	would survive unnoticed, so each is checked here against its source or against an
	independent computation rather than trusted.

	Run it after touching anything in experiment2/lib:

		.venv/bin/python experiment2/verify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from experiment2.lib.alexandrov import (
	cell_assignment, power_margin, solve_weights, solve_weights_from_cost, squared_distances,
)
from experiment2.lib.exp1_stats import (
	CLASS_NAMES, EXP1_DATA, available_models, calibrated_cloud, covariance_similarity,
	load_reported_run, norm_stats,
)
from experiment2.lib.spherical import (
	exp_map, from_tangent, geodesic_distances, karcher_mean, log_map, normalize, power_scores,
	tangent_basis, to_tangent,
)

FAILURES = []


# Record a pass or a failure without aborting, so one run reports every problem.
def check(name, condition, detail=""):
	status = "ok  " if condition else "FAIL"
	if not condition:
		FAILURES.append(name)
	print(f"  [{status}] {name}{('  — ' + detail) if detail else ''}")


# Every number the captions quote must come from a committed file, not from memory.
def check_provenance():
	print("provenance — figure captions vs committed data")
	raw = json.loads((sorted((EXP1_DATA / "analytics").glob("run_*.json"))[-1]).read_text())
	run = load_reported_run()
	frame = pd.read_csv(EXP1_DATA / "power_diagram.csv").iloc[0]

	check("reported layer matches power_diagram.csv", run["layer"] == int(frame["layer"]),
	      f"{run['layer']}")
	check("reported n_sites matches power_diagram.csv", run["n_sites"] == int(frame["n_sites"]),
	      f"{run['n_sites']}")
	check("reported accuracy matches power_diagram.csv",
	      abs(run["power_accuracy"] - float(frame["power_accuracy"])) < 1e-12,
	      f"{run['power_accuracy']:.4f}")
	check("n_sites is divisible by the class count", run["n_sites"] % len(CLASS_NAMES) == 0,
	      f"{run['n_sites'] // len(CLASS_NAMES)} per class")

	stats = norm_stats(run["layer"])
	source = raw["activation_geometry"][str(run["layer"])]
	for name in CLASS_NAMES:
		check(f"norm stats for {name} match analytics",
		      stats[name]["mean_norm"] == source[name]["mean_norm"]
		      and stats[name]["std_norm"] == source[name]["std_norm"],
		      f"{stats[name]['mean_norm']:.2f}±{stats[name]['std_norm']:.2f}")

	similarity = covariance_similarity(run["layer"])
	transition = pd.read_csv(EXP1_DATA / "layer_transition.csv")
	row = transition.loc[transition["layer"] == run["layer"]].iloc[0]
	check("covariance similarity matches layer_transition.csv",
	      abs(similarity["raw"] - float(row["raw_covariance_similarity"])) < 1e-12
	      and abs(similarity["pns"] - float(row["pns_covariance_similarity"])) < 1e-12,
	      f"raw {similarity['raw']:.3f}, PNS {similarity['pns']:.3f}")
	check("PNS raises covariance similarity at this layer (measured, not assumed)",
	      similarity["pns"] > similarity["raw"])
	check("only collected models are offered", available_models() == ["Qwen2.5-3B-Instruct"],
	      ", ".join(available_models()))


# The Alexandrov dual: its gradient, its optimum, and its gauge freedom.
def check_solver():
	print("\nsolver — the Alexandrov/Minkowski variational principle")
	rng = np.random.default_rng(0)
	points = rng.normal(size=(400, 3))
	sites = rng.normal(size=(5, 3))
	targets = rng.random(5)
	targets /= targets.sum()
	masses = np.full(len(points), 1.0 / len(points))
	cost = squared_distances(points, sites)

	# Gradient of the dual should be (target mass - achieved mass), by the envelope theorem.
	def dual(weights):
		return float(masses @ np.min(cost - weights[None, :], axis=1) + weights @ targets)

	probe = rng.normal(scale=0.3, size=5)
	_, achieved = cell_assignment(cost, probe, masses)
	analytic = targets - achieved
	step = 1e-6
	numeric = np.array([
		(dual(probe + step * np.eye(5)[i]) - dual(probe - step * np.eye(5)[i])) / (2 * step)
		for i in range(5)
	])
	check("dual gradient equals (target − achieved mass)",
	      np.allclose(analytic, numeric, atol=1e-6),
	      f"max diff {np.abs(analytic - numeric).max():.2e}")

	weights, info = solve_weights(points, sites, targets, masses)
	# Cell mass is a sum of atoms of size 1/n, so an arbitrary real target is only reachable to
	# that granularity. The floor is the claim; exactness is not.
	floor = len(sites) / len(points)
	check("solved mass reaches the atomic quantisation floor",
	      info["mass_error_l1"] <= floor + 1e-12,
	      f"L1 {info['mass_error_l1']:.2e} <= n_sites/n {floor:.2e}")
	check("achieved masses are whole multiples of 1/n",
	      np.allclose(info["achieved_masses"] * len(points),
	                  np.round(info["achieved_masses"] * len(points))))
	exact = solve_weights(points, sites, np.full(len(sites), 1.0 / len(sites)), masses)[1]
	check("representable targets are hit exactly", exact["mass_error_l1"] < 1e-12,
	      f"L1 {exact['mass_error_l1']:.2e}")
	check("weights are gauge-pinned to mean zero", abs(weights.mean()) < 1e-12)
	check("objective is invariant under a constant shift",
	      abs(dual(probe) - dual(probe + 3.7)) < 1e-9)
	check("cost-matrix solver reproduces the Euclidean solver",
	      np.allclose(weights, solve_weights_from_cost(cost, targets, masses)[0]))

	# A larger weight can only grow its own cell, never shrink it.
	grown = weights.copy()
	grown[0] += 5.0
	_, before = cell_assignment(cost, weights, masses)
	_, after = cell_assignment(cost, grown, masses)
	check("raising one weight grows that cell", after[0] >= before[0] - 1e-15,
	      f"{before[0]:.3f} → {after[0]:.3f}")

	assigned, margin = power_margin(points, sites, weights)
	check("power margin is non-negative", float(margin.min()) >= -1e-12)
	check("margin's owner matches the power-cell assignment",
	      np.array_equal(assigned, np.argmin(cost - weights[None, :], axis=1)))

	check("squared_distances matches a direct computation",
	      np.allclose(cost, ((points[:, None, :] - sites[None, :, :]) ** 2).sum(axis=2)))


# The sphere: charts, the Karcher mean, and the identities the figures rely on.
def check_geometry():
	print("\ngeometry — sphere charts and power distances")
	rng = np.random.default_rng(1)
	points = normalize(rng.normal(size=(600, 3)) + np.array([0.0, 0.0, 6.0]))
	base = karcher_mean(points)
	basis = tangent_basis(base)

	check("Karcher mean is a unit vector", abs(np.linalg.norm(base) - 1.0) < 1e-12)
	check("Karcher mean is a fixed point (tangent mean ≈ 0)",
	      float(np.linalg.norm(log_map(points, base).mean(axis=0))) < 1e-9)
	check("tangent basis is orthonormal", np.allclose(basis @ basis.T, np.eye(2)))
	check("tangent basis is orthogonal to the base point", np.allclose(basis @ base, 0.0))
	check("log/exp maps are mutual inverses",
	      np.allclose(from_tangent(to_tangent(points, base, basis), base, basis), points,
	                  atol=1e-12))
	check("exp map lands on the sphere",
	      np.allclose(np.linalg.norm(exp_map(log_map(points, base), base), axis=1), 1.0))
	# Degenerate inputs must fail loudly or return the mathematically right thing -- never NaN.
	try:
		normalize(np.zeros((1, 3)))
		zero_guarded = False
	except ValueError:
		zero_guarded = True
	check("normalize rejects zero-norm rows instead of returning NaN", zero_guarded)
	antipodal_mean = karcher_mean(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
	check("Karcher mean survives a zero centroid (antipodal pair)",
	      np.all(np.isfinite(antipodal_mean))
	      and abs(np.linalg.norm(antipodal_mean) - 1.0) < 1e-12,
	      f"{np.round(antipodal_mean, 3)}")
	pole = np.array([0.0, 0.0, 1.0])
	antipode = log_map(np.array([[0.0, 0.0, -1.0]]), pole)
	check("log map at the exact antipode has magnitude pi",
	      abs(float(np.linalg.norm(antipode)) - np.pi) < 1e-12,
	      f"{float(np.linalg.norm(antipode)):.6f}")
	check("log map at the base point is the zero vector",
	      float(np.linalg.norm(log_map(pole[None, :], pole))) < 1e-15)

	check("geodesic distance equals arccos of the inner product",
	      np.allclose(geodesic_distances(points, points[:5]),
	                  np.arccos(np.clip(points @ points[:5].T, -1.0, 1.0))))

	sites = normalize(rng.normal(size=(9, 3)) + np.array([0.0, 0.0, 6.0]))
	zero = np.zeros(len(sites))
	chordal = np.argmin(power_scores(points, sites, zero, "chordal"), axis=1)
	geodesic = np.argmin(power_scores(points, sites, zero, "geodesic"), axis=1)
	check("chordal and geodesic agree at w = 0 (both monotone in x·y)",
	      np.array_equal(chordal, geodesic))

	weights = rng.normal(scale=0.05, size=len(sites))
	differs = not np.array_equal(
		np.argmin(power_scores(points, sites, weights, "chordal"), axis=1),
		np.argmin(power_scores(points, sites, weights, "geodesic"), axis=1),
	)
	check("weights separate chordal from geodesic", differs)
	check("chordal power distance equals 2 − 2x·y − w",
	      np.allclose(power_scores(points, sites, weights, "chordal"),
	                  2.0 - 2.0 * points @ sites.T - weights[None, :]))


# The synthetic stand-in must match the statistics it claims to be calibrated to.
def check_calibration():
	print("\ncalibration — synthetic cloud vs Experiment 1's measured norms")
	run = load_reported_run()
	stats = norm_stats(run["layer"])
	for dims in (2, 3):
		points, labels = calibrated_cloud(run["layer"], dims=dims, n_per_class=2000)
		for index, name in enumerate(CLASS_NAMES):
			radius = np.linalg.norm(points[labels == index], axis=1)
			target_mean = stats[name]["mean_norm"]
			target_std = stats[name]["std_norm"]
			check(f"dims={dims} {name} mean norm within 1%",
			      abs(radius.mean() - target_mean) / target_mean < 0.01,
			      f"{radius.mean():.2f} vs {target_mean:.2f}")
			check(f"dims={dims} {name} norm spread within 10%",
			      abs(radius.std() - target_std) / target_std < 0.10,
			      f"{radius.std():.2f} vs {target_std:.2f}")
	check("Refusal is the tightest class, as Experiment 1 measured",
	      min(CLASS_NAMES, key=lambda n: stats[n]["std_norm"]) == "Refusal")


def main() -> int:
	check_provenance()
	check_solver()
	check_geometry()
	check_calibration()
	print()
	if FAILURES:
		print(f"{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
		return 1
	print("all checks passed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
