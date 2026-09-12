import numpy as np
from src.features import extract_features


# Create a test audio signal
sample_rate = 16000

duration = 2

time = np.linspace(
    0,
    duration,
    int(sample_rate * duration),
    endpoint=False
)

# Generate a simple test tone
audio = 0.5 * np.sin(
    2 * np.pi * 440 * time
)


# Extract features
features = extract_features(
    audio,
    sample_rate
)


print("Feature extraction successful!")

print("Number of features:", len(features))

print("Feature vector:")

print(features)