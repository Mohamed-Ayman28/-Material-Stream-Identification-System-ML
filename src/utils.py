"""utils.py

Shared helpers for the Material Stream Identification System.

This project includes a model-specific rejection mechanism so the system can
return "Unknown" for low-confidence / out-of-distribution inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RejectResult:
	label_idx: Optional[int]
	confidence: Optional[float]
	reason: str
	probabilities: Optional[Dict[int, float]] = None
	extra: Optional[Dict[str, Any]] = None


def _sigmoid(x: float) -> float:
	return float(1.0 / (1.0 + np.exp(-x)))


def detect_model_kind(model: Any) -> str:
	"""Best-effort model type detection without changing the model itself."""
	name = type(model).__name__.lower()
	module = type(model).__module__.lower()

	if 'neighbors' in module and 'kneighbors' in dir(model):
		return 'knn'
	if 'svm' in module and 'svc' in name:
		return 'svm'
	if 'ensemble' in module and 'voting' in name:
		return 'ensemble'
	if hasattr(model, 'predict_proba'):
		return 'probabilistic'
	return 'other'


def _top2(values: np.ndarray) -> Tuple[float, float]:
	"""Return (top1, top2) for a 1D array."""
	if values.size == 0:
		return 0.0, 0.0
	if values.size == 1:
		v = float(values[0])
		return v, 0.0
	# partial sort for speed/simplicity
	idx = np.argpartition(values, -2)[-2:]
	top = np.sort(values[idx])[::-1]
	return float(top[0]), float(top[1])


def predict_with_rejection(
	model: Any,
	X_feat: np.ndarray,
	*,
	model_kind: Optional[str] = None,
	# Tailored thresholds per model family
	svm_min_prob: float = 0.45,
	svm_min_prob_gap: float = 0.12,
	svm_min_margin_sigmoid: float = 0.50,
	knn_max_mean_distance: float = 50.0,
	ensemble_min_prob: float = 0.30,
	ensemble_min_prob_gap: float = 0.03,
	generic_min_prob: float = 0.50,
	generic_min_prob_gap: float = 0.12,
) -> RejectResult:
	"""Predict and reject low-confidence results.

	Returns label_idx=None when the input is rejected ("Unknown").
	The logic is tailored per model family:
	- KNN: mean neighbor distance gate (primary)
	- SVM: probability gate if available, otherwise margin-based heuristic
	- Ensemble: probability + top1-top2 gap
	- Other probabilistic: probability + gap
	"""
	kind = (model_kind or detect_model_kind(model)).lower()

	# KNN: distance-based rejection (robust even without predict_proba)
	if kind == 'knn' and hasattr(model, 'kneighbors'):
		try:
			neigh_dist, _ = model.kneighbors(
				X_feat,
				n_neighbors=getattr(model, 'n_neighbors', 5),
				return_distance=True,
			)
			mean_dist = float(np.mean(neigh_dist))
			pred = int(model.predict(X_feat)[0])
			conf = 1.0 / (1.0 + mean_dist)
			if mean_dist <= knn_max_mean_distance:
				return RejectResult(pred, float(conf), 'accepted', extra={'mean_distance': mean_dist})
			return RejectResult(None, float(conf), 'rejected_knn_distance', extra={'mean_distance': mean_dist})
		except Exception as e:
			# Fall back to plain prediction
			pred = int(model.predict(X_feat)[0])
			return RejectResult(pred, 1.0, 'accepted_fallback', extra={'error': str(e)})

	# Probabilistic models (including ensemble/SVM with probability=True)
	if hasattr(model, 'predict_proba'):
		try:
			probs = np.asarray(model.predict_proba(X_feat))[0]
			top1, top2 = _top2(probs)
			pred = int(np.argmax(probs))
			prob_map = {int(i): float(p) for i, p in enumerate(probs)}

			if kind == 'svm':
				min_prob = svm_min_prob
				min_gap = svm_min_prob_gap
			elif kind == 'ensemble':
				min_prob = ensemble_min_prob
				min_gap = ensemble_min_prob_gap
			else:
				min_prob = generic_min_prob
				min_gap = generic_min_prob_gap

			if top1 < min_prob:
				return RejectResult(None, float(top1), 'rejected_low_probability', probabilities=prob_map)
			if (top1 - top2) < min_gap:
				return RejectResult(None, float(top1), 'rejected_ambiguous_top2', probabilities=prob_map, extra={'top2': float(top2)})
			return RejectResult(pred, float(top1), 'accepted', probabilities=prob_map, extra={'top2': float(top2)})
		except Exception as e:
			# Continue to other fallbacks
			pass

	# SVM without probabilities: margin-based heuristic
	if kind == 'svm' and hasattr(model, 'decision_function'):
		try:
			dec = np.asarray(model.decision_function(X_feat))

			# Convert decision values into a single "margin" score.
			# - If (n_classes,) or (1, n_classes): use top1-top2 margin.
			# - Otherwise (ovo): use max absolute decision value.
			if dec.ndim == 1:
				scores = dec
			else:
				scores = dec[0]

			if scores.size >= 2:
				top1, top2 = _top2(scores)
				margin = float(top1 - top2)
			else:
				margin = float(np.max(np.abs(scores))) if scores.size else 0.0

			conf = _sigmoid(margin)
			pred = int(model.predict(X_feat)[0])
			if conf >= svm_min_margin_sigmoid:
				return RejectResult(pred, float(conf), 'accepted', extra={'margin': margin})
			return RejectResult(None, float(conf), 'rejected_low_margin', extra={'margin': margin})
		except Exception as e:
			pred = int(model.predict(X_feat)[0])
			return RejectResult(pred, 1.0, 'accepted_fallback', extra={'error': str(e)})

	# Last resort
	pred = int(model.predict(X_feat)[0])
	return RejectResult(pred, None, 'accepted_no_confidence')
