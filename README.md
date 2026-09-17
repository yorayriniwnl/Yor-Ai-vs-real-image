# YOR // Texture Forensics

![YOR Texture Forensics](assets/hero.svg)

`VERIFIED HOLDOUT` · `INFERENCE DEMO` · `ARTIFACT LINEAGE PENDING`

Texture Forensics is a small image-classification system that extracts grayscale texture, noise, and structure features before scoring them with an RBF Support Vector Machine. It is a research/demo detector, not an authenticity oracle: images outside the curated dataset can produce confident mistakes.

## Current evidence

The reproducible evaluation is [`outputs/model-evaluation.json`](outputs/model-evaluation.json). It uses the checked-in dataset, a deterministic stratified 80/20 split, `random_state=42`, an RBF SVM, and a scaler fitted on the training partition only.

| Measure | Result |
| --- | ---: |
| Total supported samples | 533 |
| Real / AI samples | 258 / 275 |
| Train / holdout samples | 426 / 107 |
| Held-out accuracy | 78.5% |
| Feature vector length | 17 |

This is one holdout from one curated dataset. It is not a universal detector benchmark, and the result should not be used to make consequential decisions about an image or its creator.

### Deployment-artifact boundary

The checked-in `svm_model.pkl` and `scaler.pkl` predate the current leakage-safe 80/20 evaluation protocol. Their historical training path fitted preprocessing before the split, so the public inference demo must **not** be described as the exact 78.5%-validated model until those binaries are regenerated and committed from the canonical exporter below.

The supported regeneration path is now:

```powershell
python -m pip install -r requirements.txt
python -m unittest tests.test_training_protocol
python scripts/train_and_export_model.py
```

That exporter splits first, fits `StandardScaler` on the training partition only, trains the RBF SVM, writes the evaluation JSON, and writes `outputs/model-artifact-lineage.json` with hashes for the exact `svm_model.pkl` and `scaler.pkl` artifacts. A held-out metric may be attached to deployable binaries only when that manifest exists and its hashes match.

The historical `final_train.py` and `svm_model.py` entrypoints are retained only as compatibility wrappers and now delegate to the canonical exporter.

![Texture forensics pipeline](assets/architecture.svg)

## Product surface

- A Flask page at `/` for a simple multipart upload flow.
- A JSON endpoint at `POST /api/predict` for the same inference path.
- A Streamlit/Three.js control-room surface in [`app.py`](app.py), using the shared [YOR visual token contract](design/yor-tokens.json).
- Upload validation for JPG, JPEG, and PNG files with a 25 MB limit.
- Confidence and estimated AI probability are shown as model outputs, not proof of provenance.

## Feature and inference flow

1. Decode the image, convert to grayscale, and resize it to `224 × 224`.
2. Extract variance, Laplacian high-frequency variance, uniform LBP histogram values, GLCM contrast, energy, homogeneity, correlation, and entropy.
3. Standardize the 17-value vector with the checked-in scaler.
4. Run the checked-in RBF SVM and expose the estimated AI probability plus confidence.

The inference flow is implemented and deployable, but its current checked-in binary artifacts remain lineage-pending until regenerated through `scripts/train_and_export_model.py`.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m unittest tests.test_training_protocol
python scripts/evaluate_model.py
python -m flask --app index run --debug
```

For the Streamlit surface:

```powershell
python -m streamlit run app.py
```

The dataset directory is checked in for reproducibility; verify the source and licensing records before redistributing it.

## Source and limitations

- The dataset is split into `dataset/real` and `dataset/ai`; exact source/licensing records should be completed before broader redistribution.
- The 78.5% evaluation is image-level and dataset-specific. It does not establish robustness to new generators, compression, screenshots, edits, social-media transforms, or adversarial examples.
- The currently checked-in deployment binaries are not yet cryptographically tied to that evaluation snapshot.
- Confidence is derived from the model’s probability/decision output. It is not calibrated as a real-world probability of authorship.
- The model uses handcrafted grayscale features. The project does not claim deep-learning performance, provenance detection, or forensic certainty.
- The Flask page processes uploaded bytes for the request and is a bounded demo, not a forensic production service.

## Repository map

- [`inference.py`](inference.py) — upload validation, feature extraction, model loading, and prediction formatting.
- [`training_pipeline.py`](training_pipeline.py) — canonical split-first, train-only preprocessing protocol.
- [`scripts/train_and_export_model.py`](scripts/train_and_export_model.py) — supported model/scaler export and lineage-manifest generator.
- [`scripts/evaluate_model.py`](scripts/evaluate_model.py) — deterministic evaluation without exporting binaries.
- [`tests/test_training_protocol.py`](tests/test_training_protocol.py) — regression guard against preprocessing leakage.
- [`outputs/model-evaluation.json`](outputs/model-evaluation.json) — current clean evaluation snapshot.
- [`index.py`](index.py) — Flask page and API route.
- [`app.py`](app.py) — Streamlit control-room interface.
- [`assets/`](assets/) — code-native YOR hero and architecture diagrams.

## Status

`DEMO / REPRODUCIBLE EVALUATION / DEPLOYMENT ARTIFACT LINEAGE PENDING`

This project is an applied-ML and interface experiment. It is not a security, identity, hiring, moderation, or financial decision system.