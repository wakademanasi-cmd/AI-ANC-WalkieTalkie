import numpy as np

from src.features import extract_features
from src.noise_classifier import classify_noise


# Create a test audio signal
sample_rate = 16000

duration = 3

t = np.arange(
    0,
    duration,
    1 / sample_rate
)

test_audio = (
    0.5 * np.sin(2 * np.pi * 300 * t)
    +
    0.3 * np.sin(2 * np.pi * 600 * t)
)


# Extract the real 36 features
features = extract_features(
    test_audio,
    sample_rate
)


# Classify the features
result = classify_noise(features)


print("----------------------------------")
print("FEATURE → CLASSIFIER TEST")
print("----------------------------------")
print("Sample rate        :", sample_rate)
print("Audio duration     :", duration, "seconds")
print("Number of features :", len(features))
print("Predicted class    :", result)
print("----------------------------------")