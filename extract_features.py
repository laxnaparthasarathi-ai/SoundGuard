
import csv
import numpy as np
import soundfile as sf
from pathlib import Path

DATASET_DIR = Path("dataset")
AUDIO_DIR = DATASET_DIR / "audio"
METADATA_FILE = DATASET_DIR / "selected_metadata.csv"
OUTPUT_FILE = DATASET_DIR / "features.npz"

def extract_features(audio_path):
    audio, sample_rate = sf.read(audio_path, dtype="float32")

    # Stereo audio-va mono-va maathuvom
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    # Fixed-length audio segment
    target_length = sample_rate * 5
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # Short-time frames
    frame_size = 2048
    hop_size = 1024
    frames = np.lib.stride_tricks.sliding_window_view(
        audio, frame_size
    )[::hop_size]

    # Window each frame before frequency analysis
    window = np.hanning(frame_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))

    # Numerical audio features
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    zcr = np.mean(
        np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]),
        axis=1
    )
    spectral_centroid = (
        np.sum(spectrum * np.arange(spectrum.shape[1]), axis=1)
        / (np.sum(spectrum, axis=1) + 1e-8)
    )

    return np.concatenate([
        [np.mean(rms), np.std(rms)],
        [np.mean(zcr), np.std(zcr)],
        [
            np.mean(spectral_centroid),
            np.std(spectral_centroid)
        ],
        np.mean(spectrum, axis=0)[::16],
        np.std(spectrum, axis=0)[::16],
    ]).astype(np.float32)

with METADATA_FILE.open(
    "r", newline="", encoding="utf-8"
) as file:
    rows = list(csv.DictReader(file))

features = []
labels = []
failed_files = []

for index, row in enumerate(rows, start=1):
    audio_path = AUDIO_DIR / row["filename"]

    try:
        feature_vector = extract_features(audio_path)
        features.append(feature_vector)
        labels.append(row["category"])
    except Exception as error:
        failed_files.append((row["filename"], str(error)))

    if index % 20 == 0:
        print(f"Processed {index}/{len(rows)} files")

if failed_files:
    print("\nFailed files:")
    for filename, error in failed_files:
        print(filename, "-", error)
    raise SystemExit("Feature extraction failed.")

X = np.vstack(features)
y = np.asarray(labels)

np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    y=y
)

print("\nFeature extraction successful!")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("Saved:", OUTPUT_FILE)
