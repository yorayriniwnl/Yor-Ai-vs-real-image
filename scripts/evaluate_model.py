#!/usr/bin/env python3
"""Reproduce the detector's deterministic held-out evaluation without exporting artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import extract_features_from_gray
from training_pipeline import train_model_from_arrays


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
    result = train_model_from_arrays(X, y, paths, test_size=0.2, random_state=42)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8")
    print(f"evaluation written: {OUTPUT_PATH}")
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
