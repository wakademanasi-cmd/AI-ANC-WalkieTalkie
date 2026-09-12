import numpy as np
import soundfile as sf


# ==================================================
# SETTINGS
# ==================================================

sample_rate = 16000
duration = 5

t = np.arange(
    0,
    duration,
    1 / sample_rate
)


# ==================================================
# SIMULATED SPEECH
# ==================================================

speech = (
    0.5 * np.sin(2 * np.pi * 300 * t)
    +
    0.3 * np.sin(2 * np.pi * 600 * t)
)


# ==================================================
# SIMULATED ENGINE NOISE
# ==================================================

noise = (
    0.35 * np.sin(2 * np.pi * 100 * t)
    +
    0.20 * np.sin(2 * np.pi * 200 * t)
)


# ==================================================
# SIMULATED TWO MICROPHONES
# ==================================================

# Microphone 1:
# Reference microphone mainly hears environmental noise
reference_mic = noise


# Microphone 2:
# Primary microphone hears speech + environmental noise
primary_mic = speech + noise


# ==================================================
# CREATE STEREO SIGNAL
# ==================================================

stereo_signal = np.column_stack(
    (
        reference_mic,
        primary_mic
    )
)


# ==================================================
# SAVE WAV
# ==================================================

output_file = "synthetic_two_mic.wav"

sf.write(
    output_file,
    stereo_signal,
    sample_rate
)


print("----------------------------------")
print("SYNTHETIC TWO-MIC WAV CREATED")
print("----------------------------------")
print(f"File       : {output_file}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Channels   : 2")
print(f"Duration   : {duration} seconds")
print("----------------------------------")
print("Channel 0 : Reference noise")
print("Channel 1 : Speech + noise")
print("----------------------------------")