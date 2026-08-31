#!/usr/bin/env python3
"""Reproduce the checked-in detector's deterministic held-out evaluation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import extract_features_from_gray


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
OUTPUT_PATH = ROOT / "outputs" / "model-evaluation.json"


def load_samples() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[tuple[np.ndarray, int, str]] = []
    for folder_name, label in (("real", 0), ("ai", 1)):
        folder = ROOT / "dataset" / folder_name
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            features = extract_features_from_gray(gray)
            if features is not None:
                rows.append((features, label, path.relative_to(ROOT).as_posix()))

    if not rows:
        raise RuntimeError("no supported dataset images produced features")
    return (
        np.array([features for features, _, _ in rows]),
        np.array([label for _, label, _ in rows]),
        [path for _, _, path in rows],
    )


def main() -> int:
    X, y, paths = load_samples()
    X_train, X_test, y_train, y_test, _, paths_test = train_test_split(
        X,
        y,
        paths,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel="rbf", class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    result = {
        "schema_version": 1,
        "evaluation": "deterministic 80/20 stratified holdout",
        "random_state": 42,
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
        "test_paths_sha256": hashlib.sha256("\n".join(sorted(paths_test)).encode("utf-8")).hexdigest(),
        "limitations": [
            "The holdout is drawn from the checked-in curated dataset and is not a measure of universal detector performance.",
            "The classifier uses handcrafted grayscale texture features and can fail on images outside this distribution.",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"evaluation written: {OUTPUT_PATH}")
    print(f"samples={result['total_samples']} train={result['train_samples']} test={result['test_samples']} accuracy={result['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
