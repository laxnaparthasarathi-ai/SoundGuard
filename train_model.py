
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_FILE = Path("dataset/features.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load extracted features and sound labels
data = np.load(DATA_FILE, allow_pickle=False)
X = data["X"]
y = data["y"]

# Split data while preserving category proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("SoundGuard - Model Training")
print("---------------------------")
print("Total samples:", len(X))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features per sample:", X.shape[1])

# Train the classifier
model = DecisionTreeClassifier(
    max_depth=8,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# Evaluate on held-out test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nTest accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=sorted(np.unique(y)),
        zero_division=0,
    )
)

classes = sorted(np.unique(y))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=classes))
print("Class order:", classes)

# Save model and evaluation metadata

artifact = {
    "model": model,
    "classes": classes,
    "feature_count": int(X.shape[1]),
    "accuracy": float(accuracy),
    "test_size": 0.25,
    "random_state": 42,
    "classification_report": classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=classes
    ).tolist(),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
}


model_path = MODEL_DIR / "soundguard_model.joblib"
joblib.dump(artifact, model_path)

print("\nSaved model:", model_path)
print("Model training complete!")
