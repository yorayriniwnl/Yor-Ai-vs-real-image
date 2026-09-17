#!/usr/bin/env python3
"""Train, evaluate, and export the deployable classifier with explicit lineage.

This is the only supported path for regenerating svm_model.pkl and scaler.pkl.
It splits the dataset before fitting StandardScaler so the holdout cannot leak
into preprocessing statistics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import extract_features_from_gray
from training_pipeline import train_model_from_arrays


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
EVALUATION_PATH = ROOT / "outputs" / "model-evaluation.json"
LINEAGE_PATH = ROOT / "outputs" / "model-artifact-lineage.json"
MODEL_PATH = ROOT / "svm_model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    X, y, paths = load_samples()
    result = train_model_from_arrays(X, y, paths, test_size=0.2, random_state=42)

    joblib.dump(result.model, MODEL_PATH)
    joblib.dump(result.scaler, SCALER_PATH)

    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8")

    lineage = {
        "schema_version": 1,
        "protocol_version": result.metadata["protocol_version"],
        "training_script": "scripts/train_and_export_model.py",
        "evaluation_file": "outputs/model-evaluation.json",
        "scaler_fit_scope": result.metadata["scaler_fit_scope"],
        "random_state": result.metadata["random_state"],
        "test_paths_sha256": result.metadata["test_paths_sha256"],
        "total_samples": result.metadata["total_samples"],
        "train_samples": result.metadata["train_samples"],
        "test_samples": result.metadata["test_samples"],
        "held_out_accuracy": result.metadata["accuracy"],
        "artifacts": {
            "svm_model.pkl": {"sha256": sha256_file(MODEL_PATH)},
            "scaler.pkl": {"sha256": sha256_file(SCALER_PATH)},
        },
        "claim_boundary": (
            "The held-out metric is attached to these exact artifact hashes only. "
            "Changing either binary requires regenerating this manifest."
        ),
    }
    LINEAGE_PATH.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")

    print(f"evaluation written: {EVALUATION_PATH}")
    print(f"lineage written: {LINEAGE_PATH}")
    print(
        "samples={total} train={train} test={test} accuracy={accuracy:.4f}".format(
            total=result.metadata["total_samples"],
            train=result.metadata["train_samples"],
            test=result.metadata["test_samples"],
            accuracy=result.metadata["accuracy"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
