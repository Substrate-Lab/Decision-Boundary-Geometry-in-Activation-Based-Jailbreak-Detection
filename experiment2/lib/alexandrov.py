#!/usr/bin/env python3
"""Solved site weights for a Laguerre (power) diagram — the Alexandrov/Minkowski map.

	Weights are *solved*, not read off a covariance summary. Given sites
	y_i and target cell masses nu_i, find the weight vector w such that each power cell carries
	its prescribed mass:

		mu(Lag_i(w)) = nu_i          Lag_i(w) = { x : ||x-y_i||^2 - w_i <= ||x-y_j||^2 - w_j }

	This is the semi-discrete optimal-transport map. It is the maximiser of the concave dual

		Phi(w) = INT min_i ( ||x-y_i||^2 - w_i ) dmu(x)  +  SUM_i w_i nu_i

	whose gradient is exactly (target mass - achieved mass):

		dPhi/dw_i = nu_i - mu(Lag_i(w))

	(the cell-boundary terms cancel by the envelope theorem). Phi is concave — a minimum over
	affine functions of w plus a linear term — so any ascent method reaches the global optimum,
	and it is invariant under w -> w + c*1, so the solution is a ray and we pin it to mean zero.

	Why this matters for the paper: with solved weights the weight functional stops being a free
	parameter a reviewer can attack, and the Alexandrov/Minkowski variational principle (Gu, Luo,
	Sun, Yau, arXiv:1302.5472) makes the optimal-transport statement actually true. Boundaries
	stay exact hyperplanes, so the signed power distance remains a well-defined geometric margin.

	Scalar weights are the point. Substituting per-class Sigma^-1 would make the boundary
	quadratic and collapse the construction into QDA.

	One limit is worth stating plainly, because it bounds every mass number these figures
	report. With an empirical source measure a cell's mass is a sum of atoms of size 1/n, so a
	target that is not itself a multiple of 1/n cannot be met exactly; the dual is piecewise
	linear and its optimum sits at the nearest reachable vertex. The residual is a quantisation
	floor of order n_sites/n, not solver error — when the targets *are* representable (equal
	cells, n divisible by the site count) the solve is exact to machine precision. Report the
	achieved mass; never assume it equals the target.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# Squared Euclidean distance from every point to every site, computed once and reused.
def squared_distances(points: np.ndarray, sites: np.ndarray) -> np.ndarray:
	point_sq = np.sum(points ** 2, axis=1)[:, None]
	site_sq = np.sum(sites ** 2, axis=1)[None, :]
	cross = points @ sites.T
	return np.maximum(point_sq + site_sq - 2.0 * cross, 0.0)


# Assign each point to its lowest power-distance site and report that cell's mass share.
def cell_assignment(squared: np.ndarray, weights: np.ndarray, point_masses: np.ndarray):
	power = squared - weights[None, :]
	assigned = np.argmin(power, axis=1)
	masses = np.bincount(assigned, weights=point_masses, minlength=weights.shape[0])
	return assigned, masses


# Negated dual objective and its gradient, so a minimiser performs the concave ascent.
def _negative_dual(weights, squared, target_masses, point_masses):
	power = squared - weights[None, :]
	assigned = np.argmin(power, axis=1)
	lowest = power[np.arange(power.shape[0]), assigned]
	achieved = np.bincount(assigned, weights=point_masses, minlength=weights.shape[0])
	value = float(point_masses @ lowest + weights @ target_masses)
	return -value, -(target_masses - achieved)


# Solve for the weights from an arbitrary cost matrix, so non-Euclidean geometries
# (geodesic, tangent-chart, Bregman) use the same variational principle unchanged.
def solve_weights_from_cost(cost, target_masses, point_masses=None, max_iter: int = 2000):
	n_points, n_sites = cost.shape
	if point_masses is None:
		point_masses = np.full(n_points, 1.0 / n_points)
	point_masses = np.asarray(point_masses, dtype=float)
	point_masses = point_masses / point_masses.sum()
	target_masses = np.asarray(target_masses, dtype=float)
	target_masses = target_masses / target_masses.sum()

	result = minimize(
		_negative_dual,
		np.zeros(n_sites),
		args=(cost, target_masses, point_masses),
		jac=True,
		method="L-BFGS-B",
		options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12},
	)
	weights = result.x - result.x.mean()
	assigned, achieved = cell_assignment(cost, weights, point_masses)
	info = {
		"achieved_masses": achieved,
		"target_masses": target_masses,
		"mass_error_l1": float(np.abs(achieved - target_masses).sum()),
		"mass_error_max": float(np.abs(achieved - target_masses).max()),
		"iterations": int(result.nit),
		"assigned": assigned,
	}
	return weights, info


# Solve for the weights whose power cells carry the prescribed masses.
def solve_weights(
	points: np.ndarray,
	sites: np.ndarray,
	target_masses: np.ndarray,
	point_masses: np.ndarray | None = None,
	max_iter: int = 2000,
):
	# The Euclidean case is the squared-distance cost; the gauge is pinned inside.
	return solve_weights_from_cost(
		squared_distances(points, sites), target_masses, point_masses, max_iter,
	)


# Signed geometric margin: gap between the nearest and second-nearest power distances.
def power_margin(points: np.ndarray, sites: np.ndarray, weights: np.ndarray):
	power = squared_distances(points, sites) - weights[None, :]
	order = np.argsort(power, axis=1)
	rows = np.arange(power.shape[0])
	nearest = power[rows, order[:, 0]]
	runner_up = power[rows, order[:, 1]]
	return order[:, 0], runner_up - nearest
