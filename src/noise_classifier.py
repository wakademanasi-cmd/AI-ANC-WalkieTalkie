import numpy as np


NOISE_CLASSES = [
    "IMPULSE",
    "ENGINE",
    "WIND",
    "BABBLE"
]


def classify_noise(features):
    """
    Temporary rule-based classifier.

    The ML model will be added later after the
    project dataset is finalized and trained.
    """

    features = np.asarray(features)

    # Check that we received the expected feature vector
    if len(features) != 36:
        raise ValueError(
            f"Expected 36 features, but received {len(features)}"
        )

    # Temporary prediction
    predicted_class = "ENGINE"

    return predicted_class