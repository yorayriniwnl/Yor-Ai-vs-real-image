"""Canonical leakage-safe training protocol for the AI-vs-real classifier.

This module contains the single source of truth for splitting, scaling, training,
and evaluation metadata. The scaler is fitted on the training partition only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROTOCOL_VERSION = 2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainingResult:
    model: SVC
    scaler: StandardScaler
    metadata: dict
    y_test: np.ndarray
    predictions: np.ndarray
    test_paths: list[str]


def _paths_hash(paths: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()


def train_model_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    paths: Sequence[str],
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TrainingResult:
    """Split first, fit preprocessing on train only, train SVM, and evaluate holdout."""
    X = np.asarray(X)
    y = np.asarray(y)
    paths = list(paths)

    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    if len(X) != len(y) or len(y) != len(paths):
        raise ValueError("X, y, and paths must have the same number of samples")
    if len(np.unique(y)) < 2:
        raise ValueError("training requires at least two classes")

    X_train, X_test, y_train, y_test, _paths_train, paths_test = train_test_split(
        X,
        y,
        paths,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metadata = {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation": (
            "deterministic 80/20 stratified holdout"
            if test_size == 0.2
            else f"deterministic {int((1 - test_size) * 100)}/{int(test_size * 100)} stratified holdout"
        ),
        "random_state": random_state,
        "scaler_fit_scope": "training_partition_only",
        "feature_length": int(X.shape[1]),
        "total_samples": int(len(y)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "class_counts": {
            "real": int((y == 0).sum()),
            "ai": int((y == 1).sum()),
        },
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_paths_sha256": _paths_hash(paths_test),
        "model": {
            "class": type(model).__name__,
            "kernel": model.kernel,
            "class_weight": "balanced",
        },
        "limitations": [
            "The holdout is drawn from the checked-in curated dataset and is not a measure of universal detector performance.",
            "The classifier uses handcrafted grayscale texture features and can fail on images outside this distribution.",
        ],
    }

    return TrainingResult(
        model=model,
        scaler=scaler,
        metadata=metadata,
        y_test=y_test,
        predictions=predictions,
        test_paths=list(paths_test),
    )
