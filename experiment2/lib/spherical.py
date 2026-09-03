#!/usr/bin/env python3
"""Intrinsic sphere geometry: Karcher mean, log/exp maps, and three power distances.

	A power diagram needs a
	Euclidean (or Bregman) structure -- a raw nested-polar vector cannot be fed to a squared
	norm, because that adds squared radians to squared magnitudes. The curved arm therefore
	outputs *tangent-space* coordinates via the log map at the Karcher mean, where Euclidean
	geometry is a legitimate local approximation and the tessellation is well defined.

	Three power distances are provided so the coordinate system can be treated as the
	independent variable it is:

		chordal    ||x-y||^2 - w        ambient Euclidean, restricted to the sphere.
		                                Sugihara's spherical Laguerre; boundaries are
		                                plane-cuts, i.e. circles on S^2.
		geodesic   d_g(x,y)^2 - w       intrinsic, d_g = arccos(x.y). Genuinely non-Euclidean.
		                                Note it agrees with chordal only when w = 0, since both
		                                are then monotone in x.y; weights break the tie.
		tangent    ||u-v||^2 - w        Euclidean in log-map coordinates at the Karcher mean.
		                                Boundaries are exact hyperplanes *there*, which is what
		                                keeps the convex-cell and signed-margin story intact.

	Tangent-space distortion is measured, not assumed negligible.
"""
from __future__ import annotations

import numpy as np


# Project onto the unit sphere. A zero row has no direction, so fail loudly rather than
# returning NaN, which would propagate silently through every chart built on it.
def normalize(points: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(points, axis=1, keepdims=True)
	if not np.all(norms > 0):
		raise ValueError(
			f"cannot project {int(np.sum(norms <= 0))} zero-norm row(s) onto the sphere"
		)
	return points / norms


# Karcher (intrinsic Frechet) mean on the sphere, by iterated log-average-exp.
# The extrinsic mean is the usual seed, but it is the zero vector for a set balanced about the
# origin (an antipodal pair, say). There the intrinsic mean is genuinely non-unique, so seed
# from an actual data point and let the iteration settle on one of them.
def karcher_mean(points: np.ndarray, iterations: int = 64, tol: float = 1e-12) -> np.ndarray:
	centroid = points.mean(axis=0)
	seed = centroid if np.linalg.norm(centroid) > 1e-12 else points[0]
	mean = normalize(seed[None, :])[0]
	for _ in range(iterations):
		tangent = log_map(points, mean)
		step = tangent.mean(axis=0)
		if np.linalg.norm(step) < tol:
			break
		mean = exp_map(step[None, :], mean)[0]
	return mean


# Riemannian log map at `base`, returning ambient tangent vectors orthogonal to `base`.
def log_map(points: np.ndarray, base: np.ndarray) -> np.ndarray:
	inner = np.clip(points @ base, -1.0, 1.0)
	angle = np.arccos(inner)
	residual = points - inner[:, None] * base[None, :]
	norm = np.linalg.norm(residual, axis=1, keepdims=True)
	scale = np.divide(angle[:, None], norm, out=np.zeros_like(norm), where=norm > 1e-15)
	tangent = residual * scale

	# At the exact antipode the residual vanishes while the true distance is pi. The direction
	# is genuinely undefined there, but the magnitude is not, so return pi along an arbitrary
	# tangent direction instead of silently collapsing the vector to zero.
	antipodal = (norm[:, 0] <= 1e-15) & (angle > 0.5 * np.pi)
	if np.any(antipodal):
		tangent[antipodal] = np.pi * tangent_basis(base)[0]
	return tangent


# Riemannian exp map at `base`, taking ambient tangent vectors back onto the sphere.
def exp_map(tangent: np.ndarray, base: np.ndarray) -> np.ndarray:
	norm = np.linalg.norm(tangent, axis=1, keepdims=True)
	direction = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 1e-15)
	return np.cos(norm) * base[None, :] + np.sin(norm) * direction


# Orthonormal basis of the tangent space at `base`, as rows.
def tangent_basis(base: np.ndarray) -> np.ndarray:
	completed, _ = np.linalg.qr(np.column_stack([base, np.eye(len(base))]))
	basis = completed[:, 1:]
	return basis.T


# Tangent coordinates of sphere points in a given basis at `base`.
def to_tangent(points: np.ndarray, base: np.ndarray, basis: np.ndarray) -> np.ndarray:
	return log_map(points, base) @ basis.T


# Sphere points from tangent coordinates in a given basis at `base`.
def from_tangent(coords: np.ndarray, base: np.ndarray, basis: np.ndarray) -> np.ndarray:
	return exp_map(coords @ basis, base)


# Pairwise geodesic distance between two sets of sphere points.
def geodesic_distances(points: np.ndarray, sites: np.ndarray) -> np.ndarray:
	return np.arccos(np.clip(points @ sites.T, -1.0, 1.0))


# Power scores under each of the three geometries; smallest entry per row wins the cell.
def power_scores(points, sites, weights, geometry, base=None, basis=None):
	if geometry == "chordal":
		inner = points @ sites.T
		return (2.0 - 2.0 * inner) - weights[None, :]
	if geometry == "geodesic":
		return geodesic_distances(points, sites) ** 2 - weights[None, :]
	if geometry == "tangent":
		point_coords = to_tangent(points, base, basis)
		site_coords = to_tangent(sites, base, basis)
		squared = (
			np.sum(point_coords ** 2, axis=1)[:, None]
			+ np.sum(site_coords ** 2, axis=1)[None, :]
			- 2.0 * point_coords @ site_coords.T
		)
		return np.maximum(squared, 0.0) - weights[None, :]
	raise ValueError(f"unknown geometry: {geometry}")


# How badly the tangent chart distorts distance: geodesic vs tangent-Euclidean, on sampled pairs.
def tangent_distortion(points, base, basis, n_pairs=20000, seed=0):
	rng = np.random.default_rng(seed)
	left = rng.integers(0, len(points), n_pairs)
	right = rng.integers(0, len(points), n_pairs)
	keep = left != right
	left, right = left[keep], right[keep]
	geodesic = np.arccos(np.clip(np.sum(points[left] * points[right], axis=1), -1.0, 1.0))
	coords = to_tangent(points, base, basis)
	flat = np.linalg.norm(coords[left] - coords[right], axis=1)
	relative = np.abs(flat - geodesic) / np.maximum(geodesic, 1e-12)
	return {"mean": float(relative.mean()), "p95": float(np.percentile(relative, 95)),
	        "max": float(relative.max())}
