from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request
from werkzeug.exceptions import RequestEntityTooLarge

from inference import (
    MAX_UPLOAD_MB,
    analyze_image_bytes,
    image_bytes_to_data_url,
    validate_upload,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI vs Real Image Detector</title>
  <style>
    :root {
      --bg: #07111f;
      --panel: rgba(9, 18, 35, 0.82);
      --panel-strong: rgba(5, 12, 24, 0.92);
      --border: rgba(142, 197, 252, 0.22);
      --text: #eff6ff;
      --muted: #9db0c8;
      --accent: #69e2c2;
      --accent-soft: rgba(105, 226, 194, 0.18);
      --danger: #ff8f8f;
      --shadow: 0 30px 90px rgba(0, 0, 0, 0.34);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(102, 126, 234, 0.26), transparent 34%),
        radial-gradient(circle at top right, rgba(105, 226, 194, 0.2), transparent 28%),
        linear-gradient(160deg, #030812 0%, #0b1830 46%, #07111f 100%);
    }
    .shell {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 24px;
      align-items: stretch;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .intro {
      padding: 34px;
      position: relative;
      overflow: hidden;
    }
    .intro::after {
      content: "";
      position: absolute;
      inset: auto -70px -70px auto;
      width: 200px;
      height: 200px;
      background: radial-gradient(circle, rgba(105, 226, 194, 0.26), transparent 70%);
      pointer-events: none;
    }
    .eyebrow {
      display: inline-flex;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(105, 226, 194, 0.24);
      background: rgba(105, 226, 194, 0.08);
      color: #b7ffed;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-size: 12px;
      font-weight: 700;
    }
    h1 {
      margin: 18px 0 14px;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 0.96;
      max-width: 10ch;
    }
    .lede {
      margin: 0;
      max-width: 42ch;
      color: var(--muted);
      line-height: 1.7;
      font-size: 16px;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 26px;
    }
    .stat {
      min-width: 150px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stat strong {
      display: block;
      font-size: 22px;
      margin-bottom: 6px;
    }
    .stat span {
      color: var(--muted);
      font-size: 13px;
    }
    .form-panel {
      padding: 28px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      background: var(--panel-strong);
    }
    .form-panel h2 {
      margin: 0;
      font-size: 24px;
    }
    .form-panel p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }
    .upload-box {
      display: block;
      padding: 22px;
      border-radius: 20px;
      border: 1px dashed rgba(142, 197, 252, 0.35);
      background: rgba(255, 255, 255, 0.03);
      cursor: pointer;
      transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .upload-box:hover {
      border-color: rgba(105, 226, 194, 0.54);
      transform: translateY(-1px);
    }
    input[type="file"] {
      width: 100%;
      color: var(--text);
    }
    .filename {
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }
    button {
      border: 0;
      border-radius: 16px;
      padding: 14px 18px;
      font-size: 15px;
      font-weight: 700;
      color: #04111e;
      background: linear-gradient(135deg, #69e2c2 0%, #8ec5fc 100%);
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      box-shadow: 0 18px 36px rgba(105, 226, 194, 0.24);
    }
    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 24px 46px rgba(105, 226, 194, 0.32);
    }
    .inline-note {
      font-size: 13px;
      color: var(--muted);
    }
    .error {
      margin-top: 24px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(255, 143, 143, 0.34);
      background: rgba(255, 143, 143, 0.08);
      color: #ffd5d5;
    }
    .result-grid {
      display: grid;
      grid-template-columns: 0.92fr 1.08fr;
      gap: 24px;
      margin-top: 24px;
    }
    .preview {
      overflow: hidden;
      padding: 18px;
    }
    .preview img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 18px;
      display: block;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .result {
      padding: 28px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: var(--accent-soft);
      border: 1px solid rgba(105, 226, 194, 0.24);
      color: #d8fff5;
      font-weight: 700;
    }
    .pill.danger {
      background: rgba(255, 143, 143, 0.12);
      border-color: rgba(255, 143, 143, 0.24);
      color: #ffe2e2;
    }
    .score-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 22px;
    }
    .score-card {
      padding: 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .score-card span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }
    .score-card strong {
      font-size: 28px;
      line-height: 1;
    }
    .meta {
      margin-top: 18px;
      padding-top: 18px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--muted);
      line-height: 1.8;
    }
    code {
      color: #d3e9ff;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.95em;
    }
    @media (max-width: 920px) {
      .hero, .result-grid {
        grid-template-columns: 1fr;
      }
      .shell {
        width: min(100% - 20px, 1120px);
        padding-top: 20px;
      }
      .intro, .form-panel, .result {
        padding: 22px;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <article class="panel intro">
        <span class="eyebrow">Classical ML Detector</span>
        <h1>AI or real? Check an image in seconds.</h1>
        <p class="lede">
          This deployment runs the trained SVM model from the repository and inspects texture,
          noise, and grayscale structure to estimate whether an uploaded image is AI-generated.
        </p>
        <div class="stats">
          <div class="stat">
            <strong>224 x 224</strong>
            <span>Resize before feature extraction</span>
          </div>
          <div class="stat">
            <strong>LBP + GLCM</strong>
            <span>Texture features used by the classifier</span>
          </div>
          <div class="stat">
            <strong>{{ max_upload_mb }}MB</strong>
            <span>Upload limit per image</span>
          </div>
        </div>
      </article>

      <form class="panel form-panel" method="post" enctype="multipart/form-data">
        <h2>Upload an image</h2>
        <p>Use a JPG or PNG file and the app will return the estimated AI probability plus confidence.</p>
        <label class="upload-box">
          <input id="image" name="image" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" required>
          <div class="filename" id="filename">No file selected yet.</div>
        </label>
        <button type="submit">Analyze image</button>
        <div class="inline-note">API clients can also POST multipart form data to <code>/api/predict</code>.</div>
      </form>
    </section>

    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}

    {% if result %}
      <section class="result-grid">
        <article class="panel preview">
          <img src="{{ preview_url }}" alt="Uploaded image preview">
        </article>
        <article class="panel result">
          <span class="pill{% if not result.is_real %} danger{% endif %}">{{ result.label }}</span>
          <h2>{{ result.summary }}</h2>
          <div class="score-row">
            <div class="score-card">
              <span>Confidence</span>
              <strong>{{ result.confidence_pct }}%</strong>
            </div>
            <div class="score-card">
              <span>AI probability</span>
              <strong>{{ result.ai_probability_pct }}%</strong>
            </div>
          </div>
          <div class="meta">
            Model: <code>{{ result.model_name }}</code><br>
            Verdict: <code>{{ "real photo" if result.is_real else "AI-generated image" }}</code>
          </div>
        </article>
      </section>
    {% endif %}
  </main>

  <script>
    const input = document.getElementById("image");
    const filename = document.getElementById("filename");
    if (input && filename) {
      input.addEventListener("change", () => {
        filename.textContent = input.files && input.files[0]
          ? input.files[0].name
          : "No file selected yet.";
      });
    }
  </script>
</body>
</html>
"""


def _read_upload():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        raise ValueError("Choose a JPG or PNG image to analyze.")

    file_bytes = upload.read()
    validate_upload(upload.filename, file_bytes)
    return upload, file_bytes


def _render_page(*, error=None, result=None, preview_url=None):
    return render_template_string(
        PAGE_TEMPLATE,
        error=error,
        max_upload_mb=MAX_UPLOAD_MB,
        preview_url=preview_url,
        result=result,
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    return _render_page(error=f"File too large. Max size is {MAX_UPLOAD_MB}MB."), 413


@app.get("/")
def home():
    return _render_page()


@app.post("/")
def analyze_form():
    try:
        upload, file_bytes = _read_upload()
        result = analyze_image_bytes(file_bytes)
        preview_url = image_bytes_to_data_url(
            file_bytes,
            filename=upload.filename,
            mime_type=upload.mimetype,
        )
        return _render_page(result=result, preview_url=preview_url)
    except Exception as exc:
        return _render_page(error=str(exc)), 400


@app.post("/api/predict")
def predict_api():
    try:
        upload, file_bytes = _read_upload()
        result = analyze_image_bytes(file_bytes)
        return jsonify(
            {
                "ok": True,
                "filename": upload.filename,
                "result": result,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
