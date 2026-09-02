#!/usr/bin/env python3
"""Bhattacharyya decomposition: how much of class separability is dispersion, not mean shift.

	For two Gaussians the Bhattacharyya distance splits exactly into a mean-separation term and
	a dispersion term, with no cross term:

		Sbar   = (S1 + S2) / 2
		D_mean = (1/8) (mu1-mu2)^T Sbar^-1 (mu1-mu2)
		D_disp = (1/2) [ log det Sbar - (log det S1 + log det S2)/2 ]
		D_B    = D_mean + D_disp
		rho    = D_disp / D_B

	Both terms are non-negative -- D_disp by AM-GM on the determinants, with equality iff
	S1 = S2 -- so rho lies in [0, 1] and reads directly as the share of separability that a
	mean-difference direction cannot capture. rho near 0 means a single direction suffices and
	the covariance story adds nothing; rho near 1 means the classes are separated by shape
	rather than location, which is precisely the regime a covariance-aware boundary is for.

	This replaces "which detector won" with a measurement that does not depend on any fitted
	detector, so it cannot be confounded by the choice of classifier or its regularisation.

	The same quantity bounds achievable error. The Bhattacharyya bound gives

		P_e <= (1/2) exp(-D_B)

	for equal priors, so any fitted detector whose error falls below that bound is evidence the
	Gaussian surrogate is wrong in that frame -- report the violation rather than the bound.

	Covariances are empirical (np.cov), which is what the committed sites_layer*.npz were fitted
	with; using a shrunk estimator here would silently disagree with them.
"""
from __future__ import annotations

import numpy as np


# Stable log-determinant, guarding the non-positive-definite case rather than returning nan.
def _logdet(matrix: np.ndarray) -> float:
	sign, value = np.linalg.slogdet(matrix)
	if sign <= 0 or not np.isfinite(value):
		eigenvalues = np.clip(np.linalg.eigvalsh(matrix), 1e-12, None)
		return float(np.sum(np.log(eigenvalues)))
	return float(value)


# Split the Bhattacharyya distance between two Gaussians into its mean and dispersion parts.
def bhattacharyya(mean_a, cov_a, mean_b, cov_b) -> dict:
	delta = np.asarray(mean_a, dtype=float) - np.asarray(mean_b, dtype=float)
	pooled = 0.5 * (np.asarray(cov_a, dtype=float) + np.asarray(cov_b, dtype=float))
	d_mean = float(0.125 * delta @ np.linalg.solve(pooled, delta))
	d_disp = float(0.5 * (_logdet(pooled) - 0.5 * (_logdet(cov_a) + _logdet(cov_b))))
	total = d_mean + d_disp
	return {
		"d_mean": d_mean,
		"d_disp": d_disp,
		"d_bhattacharyya": total,
		"rho": float(d_disp / total) if total > 0 else float("nan"),
		# Equal-prior Bhattacharyya bracket on the Bayes error of the Gaussian surrogate.
		# Upper: P_e <= 0.5*exp(-D_B). Lower: P_e >= 0.5*(1 - sqrt(1 - exp(-2*D_B))).
		# The lower one is the diagnostic: a fitted detector scoring beneath a floor derived
		# from the first two moments means the real classes are more separable than any
		# Gaussian matching those moments, i.e. the surrogate is wrong in that frame.
		"error_upper": float(0.5 * np.exp(-total)),
		"error_lower": float(0.5 * (1.0 - np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * total))))),
	}


# Empirical mean and covariance for one class, matching how sites_layer*.npz were fitted.
def class_moments(points: np.ndarray) -> tuple:
	return points.mean(axis=0), np.cov(points, rowvar=False)


# The decomposition for one class pair in one frame.
def pair_statistics(points: np.ndarray, labels: np.ndarray, class_a: str, class_b: str) -> dict:
	mean_a, cov_a = class_moments(points[labels == class_a])
	mean_b, cov_b = class_moments(points[labels == class_b])
	stats = bhattacharyya(mean_a, cov_a, mean_b, cov_b)
	stats["log_trace_ratio"] = float(np.log(np.trace(cov_a) / np.trace(cov_b)))
	stats["n_a"] = int(np.sum(labels == class_a))
	stats["n_b"] = int(np.sum(labels == class_b))
	return stats


# Percentile intervals from resampling prompts within each class, which is the unit of
# independence here -- resampling rows across classes would blur the class sizes.
def bootstrap_pair(points, labels, class_a, class_b, draws=1000, seed=0, keys=("rho",)):
	rng = np.random.default_rng(seed)
	index_a = np.flatnonzero(labels == class_a)
	index_b = np.flatnonzero(labels == class_b)
	collected = {key: [] for key in keys}
	for _ in range(draws):
		draw_a = points[rng.choice(index_a, len(index_a), replace=True)]
		draw_b = points[rng.choice(index_b, len(index_b), replace=True)]
		try:
			stats = bhattacharyya(*class_moments(draw_a), *class_moments(draw_b))
		except np.linalg.LinAlgError:
			continue
		for key in keys:
			collected[key].append(stats[key])
	summary = {}
	for key, values in collected.items():
		array = np.asarray(values, dtype=float)
		array = array[np.isfinite(array)]
		if array.size == 0:
			summary[f"{key}_lo"] = summary[f"{key}_hi"] = float("nan")
			continue
		summary[f"{key}_lo"] = float(np.percentile(array, 2.5))
		summary[f"{key}_hi"] = float(np.percentile(array, 97.5))
		summary[f"{key}_se"] = float(array.std(ddof=1))
	summary["draws_used"] = int(len(collected[keys[0]]))
	return summary


# Total variance of a point set. tr(Cov) is the sum of the per-dimension variances, so the
# off-diagonal covariances the full matrix computes are discarded work -- identical result,
# and it is the inner loop of every permutation below.
def total_variance(points: np.ndarray) -> float:
	return float(np.var(points, axis=0, ddof=1).sum())


# Scale-free spread contrast between two classes: Delta = log(tr S_a / tr S_b).
def log_trace_ratio(points: np.ndarray, labels: np.ndarray, class_a: str, class_b: str) -> float:
	return float(np.log(total_variance(points[labels == class_a])
	                    / total_variance(points[labels == class_b])))


# Null distribution of Delta under exchangeable class labels.
#
# The null being tested is that Refusal and Benign are interchangeable labels on the same
# activations, so only those two classes are pooled and reshuffled -- Jailbreak rows are held
# out entirely rather than mixed in, and the two class sizes are preserved, so the null differs
# from the observed statistic in label assignment alone.
#
# The p-value uses the (1 + count) / (1 + draws) form. Dividing by draws alone can return
# exactly zero, which would assert a precision the resampling does not have; this form floors
# p at 1/(1+draws) and is the honest statement of "no null draw reached the observed value".
def permutation_delta(points, labels, class_a, class_b, draws=1000, seed=0) -> dict:
	rng = np.random.default_rng(seed)
	observed = log_trace_ratio(points, labels, class_a, class_b)

	keep = np.isin(labels, [class_a, class_b])
	pooled = points[keep]
	n_a = int(np.sum(labels == class_a))

	null = np.empty(draws)
	for index in range(draws):
		order = rng.permutation(len(pooled))
		null[index] = float(np.log(total_variance(pooled[order[:n_a]])
		                           / total_variance(pooled[order[n_a:]])))

	extreme = int(np.sum(np.abs(null) >= abs(observed)))
	spread = null.std(ddof=1)
	return {
		"delta": observed,
		"null_mean": float(null.mean()),
		"null_sd": float(spread),
		"null_lo": float(np.percentile(null, 2.5)),
		"null_hi": float(np.percentile(null, 97.5)),
		"z_vs_null": float((observed - null.mean()) / spread) if spread > 0 else float("nan"),
		"p_two_sided": float((1 + extreme) / (1 + draws)),
		"p_floor": float(1.0 / (1 + draws)),
		"draws": int(draws),
	}


# Percentile interval for Delta itself, resampling prompts within each class.
def bootstrap_delta(points, labels, class_a, class_b, draws=1000, seed=0) -> dict:
	rng = np.random.default_rng(seed)
	index_a = np.flatnonzero(labels == class_a)
	index_b = np.flatnonzero(labels == class_b)
	values = np.empty(draws)
	for index in range(draws):
		draw_a = points[rng.choice(index_a, len(index_a), replace=True)]
		draw_b = points[rng.choice(index_b, len(index_b), replace=True)]
		values[index] = float(np.log(total_variance(draw_a) / total_variance(draw_b)))
	return {
		"delta_lo": float(np.percentile(values, 2.5)),
		"delta_hi": float(np.percentile(values, 97.5)),
		"delta_se": float(values.std(ddof=1)),
		# The sign is the claim, so report how often the resampled statistic keeps it.
		"sign_stability": float(np.mean(np.sign(values) == np.sign(values.mean()))),
	}
