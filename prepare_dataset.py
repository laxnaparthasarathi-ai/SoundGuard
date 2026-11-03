
import csv
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("dataset")
AUDIO_DIR = DATASET_DIR / "audio"
CSV_FILE = DATASET_DIR / "esc50.csv"

SELECTED_CATEGORIES = {
    "rain",
"engine",
"glass_breaking",
"siren",
"dog",
}

with CSV_FILE.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

selected_rows = [
    row for row in rows
    if row["category"] in SELECTED_CATEGORIES
]

valid_rows = []
missing_files = []

for row in selected_rows:
    audio_path = AUDIO_DIR / row["filename"]

    if audio_path.is_file():
        valid_rows.append(row)
    else:
        missing_files.append(row["filename"])

print("SoundGuard - Dataset Preparation")
print("--------------------------------")
print("Selected recordings:", len(selected_rows))
print("Available audio files:", len(valid_rows))
print("Missing audio files:", len(missing_files))

print("\nCategory counts:")
counts = Counter(row["category"] for row in valid_rows)

for category, count in sorted(counts.items()):
    print(f"{category}: {count}")

EXPECTED_TOTAL = len(SELECTED_CATEGORIES) * 40

if len(valid_rows) != EXPECTED_TOTAL:
    raise SystemExit(
        f"Dataset check failed: expected {EXPECTED_TOTAL}, "
        f"found {len(valid_rows)}."
    )

output_csv = DATASET_DIR / "selected_metadata.csv"

with output_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(valid_rows)

print("\nSaved metadata:", output_csv)
print("Dataset preparation successful!")
