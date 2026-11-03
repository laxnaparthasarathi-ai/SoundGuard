import numpy as np
import soundfile as sf
from pathlib import Path

print("SoundGuard - Audio Processing Test")
print("----------------------------------")

sample_rate = 22050
duration = 3

time = np.linspace(
    0, duration, sample_rate * duration, endpoint=False
)

audio = 0.5 * np.sin(2 * np.pi * 440 * time)

output_file = Path("dataset") / "sample_audio.wav"
sf.write(output_file, audio, sample_rate)

# Read the saved audio file
loaded_audio, loaded_sr = sf.read(output_file)

# Calculate basic audio features
mean_amplitude = np.mean(np.abs(loaded_audio))
max_amplitude = np.max(np.abs(loaded_audio))

print("Audio file:", output_file)
print("Sample rate:", loaded_sr)
print("Duration:", len(loaded_audio) / loaded_sr, "seconds")
print("Mean amplitude:", round(mean_amplitude, 4))
print("Maximum amplitude:", round(max_amplitude, 4))
print("Audio processing successful!")