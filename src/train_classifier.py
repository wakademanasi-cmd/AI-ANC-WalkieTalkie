import os
import joblib
import librosa
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.features import extract_features


DATASET_DIR = "dataset_normalized"

CATEGORIES = [
    "construction",
    "conversation",
    "engine",
    "horn",
    "mixed",
    "traffic",
    "wind"
]


X = []
y = []


print()
print("==========================================")
print("       ML NOISE CLASSIFIER TRAINING")
print("==========================================")


for category in CATEGORIES:

    category_path = os.path.join(
        DATASET_DIR,
        category
    )

    if not os.path.exists(category_path):
        print(f"Missing category: {category}")
        continue

    files = os.listdir(category_path)

    print()
    print(f"Loading: {category}")

    for filename in files:

        if not filename.lower().endswith(".wav"):
            continue

        filepath = os.path.join(
            category_path,
            filename
        )

        try:

            audio, sample_rate = librosa.load(
                filepath,
                sr=None,
                mono=True
            )

            features = extract_features(
                audio,
                sample_rate
            )

            X.append(features)
            y.append(category)

            print(f"  ✓ {filename}")

        except Exception as e:

            print(f"  ✗ {filename}")
            print(f"    Error: {e}")


X = np.array(X)
y = np.array(y)


print()
print("------------------------------------------")
print("Dataset summary")
print("------------------------------------------")

print("Total samples :", len(X))
print("Feature size  :", X.shape[1])


print()
print("Class distribution:")

for category in CATEGORIES:

    count = np.sum(y == category)

    print(
        f"{category:15s}: {count}"
    )


# -------------------------------------------------
# Train / validation split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print()
print("------------------------------------------")
print("Training Random Forest")
print("------------------------------------------")


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


# -------------------------------------------------
# Validation
# -------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("------------------------------------------")
print("MODEL VALIDATION")
print("------------------------------------------")

print(
    f"Validation Accuracy : {accuracy * 100:.2f}%"
)

print()

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# -------------------------------------------------
# Save model
# -------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/noise_classifier.joblib"

joblib.dump(
    model,
    model_path
)


print()
print("==========================================")
print("       ML MODEL TRAINING COMPLETE")
print("==========================================")

print(
    f"Model saved to: {model_path}"
)

print(
    f"Validation accuracy: {accuracy * 100:.2f}%"
)

print("==========================================")