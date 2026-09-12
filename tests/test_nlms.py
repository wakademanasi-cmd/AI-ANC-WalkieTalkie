import numpy as np
import matplotlib.pyplot as plt

from src.nlms import nlms_filter


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

sample_rate = 16000
duration = 5

t = np.linspace(
    0,
    duration,
    int(sample_rate * duration),
    endpoint=False
)


# --------------------------------------------------
# CREATE CLEAN SPEECH-LIKE SIGNAL
# --------------------------------------------------

speech = (
    0.5 * np.sin(2 * np.pi * 300 * t)
    +
    0.3 * np.sin(2 * np.pi * 600 * t)
)


# --------------------------------------------------
# CREATE ENGINE-LIKE NOISE
# --------------------------------------------------

noise = (
    0.35 * np.sin(2 * np.pi * 100 * t)
    +
    0.20 * np.sin(2 * np.pi * 200 * t)
)


# --------------------------------------------------
# PRIMARY MICROPHONE
# Speech + Noise
# --------------------------------------------------

primary = speech + noise


# --------------------------------------------------
# REFERENCE MICROPHONE
# Noise only
# --------------------------------------------------

reference = noise


# --------------------------------------------------
# RUN NLMS
# --------------------------------------------------

enhanced = nlms_filter(
    primary=primary,
    reference=reference,
    mu=0.01,
    filter_length=128
)


# --------------------------------------------------
# RMS CALCULATIONS
# --------------------------------------------------

speech_rms = np.sqrt(
    np.mean(speech ** 2)
)

noise_rms = np.sqrt(
    np.mean(noise ** 2)
)

input_rms = np.sqrt(
    np.mean(primary ** 2)
)

output_rms = np.sqrt(
    np.mean(enhanced ** 2)
)


# --------------------------------------------------
# SNR CALCULATION
# --------------------------------------------------

input_noise_error = primary - speech

output_noise_error = enhanced - speech


input_noise_power = np.mean(
    input_noise_error ** 2
)

output_noise_power = np.mean(
    output_noise_error ** 2
)


input_snr = 10 * np.log10(
    np.mean(speech ** 2)
    /
    (input_noise_power + 1e-10)
)


output_snr = 10 * np.log10(
    np.mean(speech ** 2)
    /
    (output_noise_power + 1e-10)
)


snr_improvement = output_snr - input_snr


# --------------------------------------------------
# NOISE REDUCTION
# --------------------------------------------------

noise_reduction_db = 10 * np.log10(
    (input_noise_power + 1e-10)
    /
    (output_noise_power + 1e-10)
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("----------------------------------")
print("NLMS PERFORMANCE TEST")
print("----------------------------------")

print(
    f"Clean Speech RMS : {speech_rms:.4f}"
)

print(
    f"Noise RMS        : {noise_rms:.4f}"
)

print(
    f"Input RMS        : {input_rms:.4f}"
)

print(
    f"Output RMS       : {output_rms:.4f}"
)

print("----------------------------------")

print(
    f"Input SNR        : {input_snr:.2f} dB"
)

print(
    f"Output SNR       : {output_snr:.2f} dB"
)

print(
    f"SNR Improvement  : {snr_improvement:.2f} dB"
)

print(
    f"Noise Reduction  : {noise_reduction_db:.2f} dB"
)

print("----------------------------------")


# ==================================================
# VISUALIZATION
# ==================================================

# Show only first 0.1 seconds
# so the waveform is easy to see.

# ==================================================
# WAVEFORM COMPARISON
# ==================================================

display_time = 0.1

display_samples = int(
    sample_rate * display_time
)


fig, axes = plt.subplots(
    4,
    1,
    figsize=(12, 10)
)


# --------------------------------------------------
# 1. CLEAN SPEECH
# --------------------------------------------------

axes[0].plot(
    t[:display_samples],
    speech[:display_samples]
)

axes[0].set_title(
    "1. Clean Speech-Like Signal"
)

axes[0].set_ylabel("Amplitude")

axes[0].grid(True)


# --------------------------------------------------
# 2. NOISE
# --------------------------------------------------

axes[1].plot(
    t[:display_samples],
    noise[:display_samples]
)

axes[1].set_title(
    "2. Engine-Like Noise"
)

axes[1].set_ylabel("Amplitude")

axes[1].grid(True)


# --------------------------------------------------
# 3. NOISY PRIMARY SIGNAL
# --------------------------------------------------

axes[2].plot(
    t[:display_samples],
    primary[:display_samples]
)

axes[2].set_title(
    "3. Primary Microphone: Speech + Noise"
)

axes[2].set_ylabel("Amplitude")

axes[2].grid(True)


# --------------------------------------------------
# 4. NLMS ENHANCED SIGNAL
# --------------------------------------------------

axes[3].plot(
    t[:display_samples],
    enhanced[:display_samples]
)

axes[3].set_title(
    "4. NLMS Enhanced Signal"
)

axes[3].set_xlabel("Time (seconds)")

axes[3].set_ylabel("Amplitude")

axes[3].grid(True)


# --------------------------------------------------
# FINAL LAYOUT
# --------------------------------------------------

plt.tight_layout()

plt.show()