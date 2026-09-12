import numpy as np

from src.audio_input import prepare_two_channel_audio


# Simulated microphone signals
reference = np.random.randn(16000).astype(np.float32)
primary = np.random.randn(16000).astype(np.float32)


audio = prepare_two_channel_audio(
    reference,
    primary
)


print("Two-channel input prepared successfully.")
print("Shape:", audio.shape)
print("Reference channel:", audio[0].shape)
print("Primary channel:", audio[1].shape)
print("Data type:", audio.dtype)
print("Maximum amplitude:", np.max(np.abs(audio)))