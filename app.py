
import os
import tempfile
from pathlib import Path
from collections import deque

import numpy as np
import soundfile as sf
import joblib

from flask import Flask, render_template, request


# --------------------------------------------------
# 1. Flask application configuration
# --------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "soundguard_model.joblib"

# Load the trained model and evaluation information
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
classes = [str(label) for label in artifact["classes"]]
expected_features = artifact["feature_count"]

# Keep the latest 10 predictions in memory
history = deque(maxlen=10)


# --------------------------------------------------
# 2. Audio feature extraction
# Must match extract_features.py exactly
# --------------------------------------------------

def extract_features(audio_path):
    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # Pad or trim audio to 5 seconds
    target_length = sample_rate * 5

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    frame_size = 2048
    hop_size = 1024

    if len(audio) < frame_size:
        raise ValueError("Audio is too short.")

    frames = np.lib.stride_tricks.sliding_window_view(
        audio,
        frame_size
    )[::hop_size]

    # Apply Hann window and calculate frequency spectrum
    window = np.hanning(frame_size).astype(np.float32)

    spectrum = np.abs(
        np.fft.rfft(frames * window, axis=1)
    )

    # Root mean square energy
    rms = np.sqrt(
        np.mean(frames ** 2, axis=1)
    )

    # Zero-crossing rate
    zcr = np.mean(
        np.signbit(frames[:, 1:])
        != np.signbit(frames[:, :-1]),
        axis=1
    )

    # Spectral centroid
    spectral_centroid = (
        np.sum(
            spectrum * np.arange(spectrum.shape[1]),
            axis=1
        )
        / (np.sum(spectrum, axis=1) + 1e-8)
    )

    # Construct the same 136 features used in training
    features = np.concatenate([
        [np.mean(rms), np.std(rms)],
        [np.mean(zcr), np.std(zcr)],
        [
            np.mean(spectral_centroid),
            np.std(spectral_centroid)
        ],
        np.mean(spectrum, axis=0)[::16],
        np.std(spectrum, axis=0)[::16],
    ]).astype(np.float32)

    if len(features) != expected_features:
        raise ValueError(
            "Audio features do not match the trained model."
        )

    return features.reshape(1, -1)


# --------------------------------------------------
# 3. Main dashboard
# --------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        history=list(history)
    )


# --------------------------------------------------
# 4. Audio upload and prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("audio")

    if not uploaded or not uploaded.filename:
        return render_template(
            "index.html",
            error="Please select an audio file.",
            history=list(history)
        )

    if Path(uploaded.filename).suffix.lower() != ".wav":
        return render_template(
            "index.html",
            error="Please upload a WAV audio file.",
            history=list(history)
        )

    temp_path = None

    try:
        # Save the upload temporarily
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_path = temp_file.name
            uploaded.save(temp_path)

        # Extract features and predict
        features = extract_features(temp_path)

        prediction = str(model.predict(features)[0])

        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities)) * 100

        result = {
            "label": prediction.replace("_", " ").title(),
            "confidence": round(confidence, 2)
        }

        # Add the result to recent prediction history
        history.appendleft({
            "filename": Path(uploaded.filename).name,
            **result
        })

        return render_template(
            "index.html",
            result=result,
            history=list(history)
        )

    except Exception:
        app.logger.exception("Audio prediction failed")

        return render_template(
            "index.html",
            error=(
                "Unable to analyze this file. "
                "Please check that it is a valid WAV recording."
            ),
            history=list(history)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# 5. Model evaluation dashboard
# --------------------------------------------------

@app.route("/evaluation")
def evaluation():
    report = artifact.get("classification_report", {})
    matrix = artifact.get("confusion_matrix", [])

    metrics = []

    for label in classes:
        values = report.get(str(label), {})

        metrics.append({
            "label": str(label).replace("_", " ").title(),
            "precision": round(
                values.get("precision", 0) * 100, 1
            ),
            "recall": round(
                values.get("recall", 0) * 100, 1
            ),
            "f1": round(
                values.get("f1-score", 0) * 100, 1
            ),
            "support": int(
                values.get("support", 0)
            )
        })

    display_classes = [
        label.replace("_", " ").title()
        for label in classes
    ]

    return render_template(
        "evaluation.html",
        accuracy=round(
            artifact.get("accuracy", 0) * 100, 1
        ),
        train_samples=artifact.get("train_samples", 0),
        test_samples=artifact.get("test_samples", 0),
        classes=display_classes,
        metrics=metrics,
        matrix=matrix
    )


# --------------------------------------------------
# 6. Run the application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
