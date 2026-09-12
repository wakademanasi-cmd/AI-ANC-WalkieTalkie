import numpy as np
import soundfile as sf


# ==================================================
# 1. LOAD ORIGINAL WAV
# ==================================================

original, sample_rate = sf.read(
    "synthetic_two_mic.wav"
)

# Channel 0 = reference noise
reference = original[:, 0]

# Channel 1 = speech + noise
primary = original[:, 1]


# ==================================================
# 2. LOAD ENHANCED WAV
# ==================================================

enhanced, enhanced_rate = sf.read(
    "enhanced.wav"
)


# ==================================================
# 3. RECREATE KNOWN CLEAN SPEECH
# ==================================================

duration = len(primary) / sample_rate

t = np.arange(
    len(primary)
) / sample_rate

speech = (
    0.5 * np.sin(2 * np.pi * 300 * t)
    +
    0.3 * np.sin(2 * np.pi * 600 * t)
)


# ==================================================
# 4. RMS FUNCTION
# ==================================================

def rms(signal):
    return np.sqrt(
        np.mean(signal ** 2)
    )


# ==================================================
# 5. SNR FUNCTION
# ==================================================

def calculate_snr(signal, clean):
    noise = signal - clean

    signal_power = np.mean(
        clean ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    return 10 * np.log10(
        signal_power /
        (noise_power + 1e-12)
    )


# ==================================================
# 6. CALCULATE PERFORMANCE
# ==================================================

clean_rms = rms(speech)

input_rms = rms(primary)

output_rms = rms(enhanced)

reference_rms = rms(reference)

input_snr = calculate_snr(
    primary,
    speech
)

output_snr = calculate_snr(
    enhanced,
    speech
)

snr_improvement = (
    output_snr - input_snr
)


# ==================================================
# 7. NOISE REDUCTION
# ==================================================

input_noise = primary - speech

output_noise = enhanced - speech

input_noise_power = np.mean(
    input_noise ** 2
)

output_noise_power = np.mean(
    output_noise ** 2
)

noise_reduction = 10 * np.log10(
    input_noise_power /
    (output_noise_power + 1e-12)
)


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print()
print("----------------------------------")
print("REAL WAV NLMS PERFORMANCE")
print("----------------------------------")

print(f"Sample Rate       : {sample_rate} Hz")
print(f"Duration          : {duration:.2f} seconds")

print("----------------------------------")

print(f"Clean Speech RMS  : {clean_rms:.4f}")
print(f"Reference RMS     : {reference_rms:.4f}")
print(f"Input RMS         : {input_rms:.4f}")
print(f"Output RMS        : {output_rms:.4f}")

print("----------------------------------")

print(f"Input SNR         : {input_snr:.2f} dB")
print(f"Output SNR        : {output_snr:.2f} dB")
print(f"SNR Improvement   : {snr_improvement:.2f} dB")
print(f"Noise Reduction   : {noise_reduction:.2f} dB")

print("----------------------------------")