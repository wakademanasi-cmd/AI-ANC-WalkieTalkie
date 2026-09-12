import joblib
import librosa

from src.features import extract_features


MODEL_PATH = "models/noise_classifier.joblib"

TEST_FILE = "dataset_normalized/engine/1-18527-A-44.wav"


print()
print("==========================================")
print("       TRAINED ML MODEL TEST")
print("==========================================")


# Load model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# Load test audio
audio, sample_rate = librosa.load(
    TEST_FILE,
    sr=None,
    mono=True
)


print("Test file    :", TEST_FILE)
print("Sample rate  :", sample_rate)
print("Duration     :", len(audio) / sample_rate, "seconds")


# Extract features
features = extract_features(
    audio,
    sample_rate
)


print("Feature size :", len(features))


# Prediction
prediction = model.predict(
    [features]
)[0]


# Probability
probabilities = model.predict_proba(
    [features]
)[0]


classes = model.classes_

confidence = max(probabilities) * 100


print()
print("------------------------------------------")
print("ML PREDICTION")
print("------------------------------------------")

print("Predicted noise :", prediction)
print(f"Confidence      : {confidence:.2f}%")

print()
print("Class probabilities:")

for class_name, probability in zip(
    classes,
    probabilities
):

    print(
        f"{class_name:15s}: {probability * 100:.2f}%"
    )


print()
print("==========================================")
print("       ML MODEL TEST COMPLETE")
print("==========================================")