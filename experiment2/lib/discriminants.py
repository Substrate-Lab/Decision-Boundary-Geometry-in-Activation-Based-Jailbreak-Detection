#!/usr/bin/env python3
"""Pluggable class discriminants for the Experiment 2 boundary.

	Each discriminant fits per-class statistics in a working space and returns, for any set of
	points, a score per class where the smallest score wins the cell. Four models are offered so
	the open question (is the covariance-aware form a real power diagram or just QDA?) can
	be tested empirically instead of hardcoded:

		lda          - shared pooled covariance, linear boundary (baseline).
		power_single - one scalar-weighted site per class, w = trace(Sigma); flat boundary,
		               a true single-site Laguerre power diagram.
		power_multi  - many k-means sub-sites per class, scalar weights; piecewise-linear
		               boundary that stays a real power diagram (the default).
		qda          - full per-class covariance, quadric boundary. This is exactly QDA / the
		               Gaussian log-likelihood discriminant, NOT a power diagram (see README).

	Covariances use Ledoit-Wolf shrinkage so the inverse and log-determinant stay defined when a
	class has few points relative to the dimension.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

CLASS_NAMES = ["Refusal", "Jailbreak", "Benign"]
SEED = 0


# Shrunk covariance, its precision, and stable log-determinant for a set of rows.
def shrunk_covariance(rows: np.ndarray):
	dims = rows.shape[1]
	if rows.shape[0] < 2:
		identity = np.eye(dims)
		return identity, identity, 0.0
	model = LedoitWolf().fit(rows)
	covariance = model.covariance_
	precision = model.precision_
	sign, logdet = np.linalg.slogdet(covariance)
	if sign <= 0 or not np.isfinite(logdet):
		eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 1e-12, None)
		logdet = float(np.sum(np.log(eigenvalues)))
	return covariance, precision, float(logdet)


# Shared interface: fit per-class statistics, then score every point against every class.
class Discriminant:
	# Record the class order present in the labels.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "Discriminant":
		self.classes_ = np.array([name for name in CLASS_NAMES if np.any(labels == name)])
		return self

	# Return an (n_points, n_classes) score array where the smallest entry per row wins.
	def scores(self, points: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	# Assign each point to its lowest-scoring class and report the geometric margin.
	def assign_and_margin(self, points: np.ndarray):
		scores = self.scores(points)
		order = np.argsort(scores, axis=1)
		assigned = self.classes_[order[:, 0]]
		best = np.take_along_axis(scores, order[:, :1], axis=1)[:, 0]
		second = np.take_along_axis(scores, order[:, 1:2], axis=1)[:, 0]
		return assigned, second - best


# Shared pooled-covariance Mahalanobis distance to each class mean (linear boundary).
class LdaDiscriminant(Discriminant):
	# Fit class means and one pooled shrunk precision from the class-centered data.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "LdaDiscriminant":
		super().fit(points, labels)
		self.means_ = {}
		centered = []
		for name in self.classes_:
			rows = points[labels == name]
			mean = rows.mean(axis=0)
			self.means_[name] = mean
			centered.append(rows - mean)
		_, self.precision_, _ = shrunk_covariance(np.vstack(centered))
		return self

	# Score each point by its pooled Mahalanobis distance to every class mean.
	def scores(self, points: np.ndarray) -> np.ndarray:
		columns = []
		for name in self.classes_:
			delta = points - self.means_[name]
			columns.append(np.einsum("ij,jk,ik->i", delta, self.precision_, delta))
		return np.column_stack(columns)


# Full per-class covariance discriminant: this is QDA, kept as an explicit baseline.
class QdaDiscriminant(Discriminant):
	# Fit each class mean, shrunk precision, and log-determinant.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "QdaDiscriminant":
		super().fit(points, labels)
		self.means_ = {}
		self.precisions_ = {}
		self.logdets_ = {}
		for name in self.classes_:
			rows = points[labels == name]
			self.means_[name] = rows.mean(axis=0)
			_, precision, logdet = shrunk_covariance(rows)
			self.precisions_[name] = precision
			self.logdets_[name] = logdet
		return self

	# Score each point by its per-class Mahalanobis distance plus the log-volume term.
	def scores(self, points: np.ndarray) -> np.ndarray:
		columns = []
		for name in self.classes_:
			delta = points - self.means_[name]
			mahalanobis = np.einsum("ij,jk,ik->i", delta, self.precisions_[name], delta)
			columns.append(mahalanobis + self.logdets_[name])
		return np.column_stack(columns)


# Single scalar-weighted site per class: a true Laguerre power diagram with a flat boundary.
class PowerSingleDiscriminant(Discriminant):
	# Fit one site per class with weight equal to the trace of its covariance.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "PowerSingleDiscriminant":
		super().fit(points, labels)
		self.sites_ = []
		self.weights_ = []
		self.site_classes_ = []
		for name in self.classes_:
			rows = points[labels == name]
			self.sites_.append(rows.mean(axis=0))
			covariance, _, _ = shrunk_covariance(rows)
			self.weights_.append(float(np.trace(covariance)))
			self.site_classes_.append(name)
		self.sites_ = np.array(self.sites_)
		self.weights_ = np.array(self.weights_)
		self.site_classes_ = np.array(self.site_classes_)
		return self

	# Score each point by the smallest power distance among the sites of each class.
	def scores(self, points: np.ndarray) -> np.ndarray:
		squared = np.sum((points[:, None, :] - self.sites_[None, :, :]) ** 2, axis=2)
		power = squared - self.weights_[None, :]
		columns = [power[:, self.site_classes_ == name].min(axis=1) for name in self.classes_]
		return np.column_stack(columns)


# Many scalar-weighted sub-sites per class: a piecewise-linear power diagram.
class PowerMultiDiscriminant(Discriminant):
	# Split each class into k-means sub-sites, each weighted by the trace of its sub-covariance.
	def __init__(self, sites_per_class: int = 12):
		self.sites_per_class = sites_per_class

	# Fit the sub-sites, weights, and their owning class labels.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "PowerMultiDiscriminant":
		super().fit(points, labels)
		sites = []
		weights = []
		site_classes = []
		for name in self.classes_:
			rows = points[labels == name]
			if len(rows) < 2:
				continue
			clusters = min(self.sites_per_class, len(rows))
			assignment = KMeans(n_clusters=clusters, n_init=10, random_state=SEED).fit_predict(rows)
			for cluster in range(clusters):
				members = rows[assignment == cluster]
				if len(members) == 0:
					continue
				sites.append(members.mean(axis=0))
				weight = float(np.trace(np.cov(members, rowvar=False))) if len(members) >= 2 else 0.0
				weights.append(weight)
				site_classes.append(name)
		self.sites_ = np.array(sites)
		self.weights_ = np.array(weights)
		self.site_classes_ = np.array(site_classes)
		return self

	# Score each point by the smallest power distance among the sub-sites of each class.
	def scores(self, points: np.ndarray) -> np.ndarray:
		squared = np.sum((points[:, None, :] - self.sites_[None, :, :]) ** 2, axis=2)
		power = squared - self.weights_[None, :]
		columns = [power[:, self.site_classes_ == name].min(axis=1) for name in self.classes_]
		return np.column_stack(columns)


# Build a discriminant by name.
def build_discriminant(name: str, sites_per_class: int = 12) -> Discriminant:
	if name == "lda":
		return LdaDiscriminant()
	if name == "qda":
		return QdaDiscriminant()
	if name == "power_single":
		return PowerSingleDiscriminant()
	if name == "power_multi":
		return PowerMultiDiscriminant(sites_per_class=sites_per_class)
	raise ValueError(f"unknown discriminant: {name}")
