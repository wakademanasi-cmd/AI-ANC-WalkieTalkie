import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from src.nlms import nlms_filter


# ==================================================
# 1. LOAD STEREO WAV
# ==================================================

input_file = "synthetic_two_mic.wav"

audio, sample_rate = sf.read(input_file)


# ==================================================
# 2. CHECK AUDIO
# ==================================================

print("----------------------------------")
print("TWO-MIC WAV LOADED")
print("----------------------------------")
print(f"Sample rate : {sample_rate} Hz")
print(f"Channels    : {audio.shape[1]}")
print(f"Samples     : {audio.shape[0]}")
print("----------------------------------")


# ==================================================
# 3. SEPARATE MICROPHONE CHANNELS
# ==================================================

reference = audio[:, 0]
primary = audio[:, 1]


print("Channel 0 → Reference microphone")
print("Channel 1 → Primary microphone")


# ==================================================
# 4. APPLY NLMS
# ==================================================

enhanced = nlms_filter(
    primary=primary,
    reference=reference,
    mu=0.005,
    filter_length=256
)


# ==================================================
# 5. SAVE ENHANCED AUDIO
# ==================================================

output_file = "enhanced.wav"

sf.write(
    output_file,
    enhanced,
    sample_rate
)


# ==================================================
# 6. PRINT RESULTS
# ==================================================

print("----------------------------------")
print("NLMS PROCESSING COMPLETE")
print("----------------------------------")
print(f"Output file : {output_file}")
print(f"μ           : 0.005")
print(f"Filter      : 256 taps")
print("----------------------------------")


# ==================================================
# 7. WAVEFORM COMPARISON
# ==================================================

display_time = 0.1

display_samples = int(
    sample_rate * display_time
)

time = np.arange(
    display_samples
) / sample_rate


fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 8)
)


# Reference microphone

axes[0].plot(
    time,
    reference[:display_samples]
)

axes[0].set_title(
    "Reference Microphone — Noise"
)

axes[0].set_ylabel(
    "Amplitude"
)

axes[0].grid(True)


# Primary microphone

axes[1].plot(
    time,
    primary[:display_samples]
)

axes[1].set_title(
    "Primary Microphone — Speech + Noise"
)

axes[1].set_ylabel(
    "Amplitude"
)

axes[1].grid(True)


# Enhanced output

axes[2].plot(
    time,
    enhanced[:display_samples]
)

axes[2].set_title(
    "NLMS Enhanced Output"
)

axes[2].set_xlabel(
    "Time (seconds)"
)

axes[2].set_ylabel(
    "Amplitude"
)

axes[2].grid(True)


plt.tight_layout()

plt.show()