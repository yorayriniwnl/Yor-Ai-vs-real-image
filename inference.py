from __future__ import annotations

import base64
import mimetypes
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

MAX_UPLOAD_MB = 25
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


@lru_cache(maxsize=1)
def load_model_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Commit svm_model.pkl and scaler.pkl before deploying."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def validate_upload(filename: str, file_bytes: bytes) -> None:
    if not filename:
        raise ValueError("Choose a JPG or PNG image to analyze.")

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only JPG, JPEG, and PNG images are supported.")

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError(f"File too large. Max size is {MAX_UPLOAD_MB}MB.")


def extract_features_from_gray(gray):
    if gray is None:
        return None

    gray = cv2.resize(gray, (224, 224))
    norm_image = gray / 255.0

    variance = np.var(norm_image)
    laplacian = cv2.Laplacian(norm_image, cv2.CV_64F)
    hf_variance = np.var(laplacian)

    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7

    reduced = gray // 16
    glcm = graycomatrix(
        reduced,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    glcm_matrix = glcm[:, :, 0, 0]
    entropy = -np.sum(glcm_matrix * np.log2(glcm_matrix + 1e-10))

    return np.hstack(
        [
            variance,
            hf_variance,
            hist,
            contrast,
            energy,
            homogeneity,
            correlation,
            entropy,
        ]
    )


def extract_features_from_image(image_bgr):
    if image_bgr is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return extract_features_from_gray(gray)


def predict_probabilities(model_obj, features_scaled):
    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(features_scaled)[0]
        classes = getattr(model_obj, "classes_", None)
        if classes is not None and 1 in classes:
            ai_idx = list(classes).index(1)
        elif len(proba) > 1:
            ai_idx = 1
        else:
            ai_idx = 0

        prob_ai = float(proba[ai_idx])
        confidence = float(max(prob_ai, 1.0 - prob_ai))
        return prob_ai, confidence

    if hasattr(model_obj, "decision_function"):
        score = model_obj.decision_function(features_scaled)
        score = float(score[0]) if np.ndim(score) else float(score)
        score = float(np.clip(score, -10.0, 10.0))
        prob_ai = float(1.0 / (1.0 + np.exp(-score)))
        confidence = float(max(prob_ai, 1.0 - prob_ai))
        return prob_ai, confidence

    pred = model_obj.predict(features_scaled)[0]
    prob_ai = 1.0 if pred == 1 else 0.0
    return float(prob_ai), 0.5


def analyze_image_bytes(file_bytes: bytes):
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    model, scaler = load_model_artifacts()
    features = extract_features_from_image(image)
    if features is None:
        raise ValueError("Could not extract features from the uploaded image.")

    features_scaled = scaler.transform([features])
    prob_ai, confidence = predict_probabilities(model, features_scaled)

    confidence_pct = int(round(confidence * 100.0))
    ai_probability_pct = int(round(prob_ai * 100.0))
    is_real = prob_ai < 0.5

    model_name = type(model).__name__
    if hasattr(model, "kernel"):
        model_name = f"{model_name} ({model.kernel})"

    return {
        "is_real": bool(is_real),
        "label": "Real photo" if is_real else "AI-generated",
        "confidence_pct": confidence_pct,
        "ai_probability_pct": ai_probability_pct,
        "model_name": model_name,
        "summary": f"Estimated AI probability: {ai_probability_pct}%.",
    }


def image_bytes_to_data_url(file_bytes: bytes, filename: str | None = None, mime_type: str | None = None) -> str:
    if not mime_type and filename:
        mime_type = mimetypes.guess_type(filename)[0]
    if not mime_type:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
