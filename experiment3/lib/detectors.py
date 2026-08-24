#!/usr/bin/env python3
"""Binary jailbreak detectors for the Experiment 3 baseline shootout.

	Every detector fits on two classes and returns a scalar score per point where higher means
	more like the positive class, so a single ROC-AUC compares them all on equal footing. The
	baselines are a linear probe (logistic regression, the standard activation detector) and a
	plain unweighted Voronoi (nearest class mean). The geometry methods reuse the Experiment 2
	discriminants (qda, lda, power_single, power_multi) via a thin adapter: their two class scores
	are turned into one positive-class score as score(negative) - score(positive).

	The §6.1 point matters here: because the power methods use scalar weights they are genuinely
	distinct from the qda baseline, so "power vs qda" in the shootout is a real comparison, not a
	formula against itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment2.lib.discriminants import build_discriminant

DISCRIMINANT_NAMES = ["lda", "qda", "power_single", "power_multi"]


# Shared interface: fit on two classes, then score points toward the positive class.
class Detector:
	# Record the positive and negative class names.
	def __init__(self, positive: str, negative: str):
		self.positive = positive
		self.negative = negative

	# Fit the detector on the two-class training data.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "Detector":
		raise NotImplementedError

	# Return a score per point where higher means more like the positive class.
	def score(self, points: np.ndarray) -> np.ndarray:
		raise NotImplementedError


# Logistic-regression probe, the standard linear activation detector.
class LinearProbeDetector(Detector):
	# Fit logistic regression with the positive class as target 1.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "LinearProbeDetector":
		target = (labels == self.positive).astype(int)
		self.model_ = LogisticRegression(max_iter=1000).fit(points, target)
		return self

	# Score by the signed distance to the probe hyperplane.
	def score(self, points: np.ndarray) -> np.ndarray:
		return self.model_.decision_function(points)


# Plain unweighted Voronoi: nearest class mean, no covariance weight.
class PlainVoronoiDetector(Detector):
	# Fit the two class means.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "PlainVoronoiDetector":
		self.pos_mean_ = points[labels == self.positive].mean(axis=0)
		self.neg_mean_ = points[labels == self.negative].mean(axis=0)
		return self

	# Score by how much closer a point is to the positive mean than the negative mean.
	def score(self, points: np.ndarray) -> np.ndarray:
		to_pos = np.sum((points - self.pos_mean_) ** 2, axis=1)
		to_neg = np.sum((points - self.neg_mean_) ** 2, axis=1)
		return to_neg - to_pos


# Adapter turning an Experiment 2 discriminant into a positive-class score.
class DiscriminantDetector(Detector):
	# Record which discriminant to build.
	def __init__(self, name: str, positive: str, negative: str, sites_per_class: int = 12):
		super().__init__(positive, negative)
		self.name = name
		self.sites_per_class = sites_per_class

	# Fit the underlying discriminant on the two classes.
	def fit(self, points: np.ndarray, labels: np.ndarray) -> "DiscriminantDetector":
		self.model_ = build_discriminant(self.name, sites_per_class=self.sites_per_class).fit(points, labels)
		self.pos_column_ = int(np.where(self.model_.classes_ == self.positive)[0][0])
		self.neg_column_ = int(np.where(self.model_.classes_ == self.negative)[0][0])
		return self

	# Score by negative-class distance minus positive-class distance.
	def score(self, points: np.ndarray) -> np.ndarray:
		scores = self.model_.scores(points)
		return scores[:, self.neg_column_] - scores[:, self.pos_column_]


# Build a detector by name for a given positive/negative class pair.
def build_detector(name: str, positive: str, negative: str, sites_per_class: int = 12) -> Detector:
	if name == "linear_probe":
		return LinearProbeDetector(positive, negative)
	if name == "plain_voronoi":
		return PlainVoronoiDetector(positive, negative)
	if name in DISCRIMINANT_NAMES:
		return DiscriminantDetector(name, positive, negative, sites_per_class=sites_per_class)
	raise ValueError(f"unknown detector: {name}")


# The full detector roster used by the shootout.
def default_detectors():
	return ["linear_probe", "plain_voronoi", "lda", "qda", "power_single", "power_multi"]
