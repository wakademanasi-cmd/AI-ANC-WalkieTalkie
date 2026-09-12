import numpy as np
from src.nlms import nlms_filter


# ==================================================
# 1. CREATE TEST SIGNAL
# ==================================================

sample_rate = 16000
duration = 5

t = np.arange(
    0,
    duration,
    1 / sample_rate
)

# Synthetic speech-like signal
speech = (
    0.5 * np.sin(2 * np.pi * 300 * t)
    +
    0.3 * np.sin(2 * np.pi * 600 * t)
)

# Synthetic engine-like noise
noise = (
    0.35 * np.sin(2 * np.pi * 100 * t)
    +
    0.20 * np.sin(2 * np.pi * 200 * t)
)

# Primary microphone signal
primary = speech + noise


# ==================================================
# 2. SNR FUNCTION
# ==================================================

def calculate_snr(signal, reference):
    signal_power = np.mean(reference ** 2)
    error_power = np.mean((signal - reference) ** 2)

    return 10 * np.log10(
        signal_power / (error_power + 1e-12)
    )


# Input SNR
input_snr = calculate_snr(
    primary,
    speech
)


# ==================================================
# 3. PARAMETERS TO TEST
# ==================================================

mu_values = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05
]

filter_lengths = [
    64,
    128,
    256
]


# ==================================================
# 4. RUN PARAMETER SWEEP
# ==================================================

results = []

print()
print("=" * 75)
print("NLMS PARAMETER TUNING")
print("=" * 75)

print(f"Input SNR: {input_snr:.2f} dB")
print()

print(
    f"{'MU':<10}"
    f"{'FILTER':<10}"
    f"{'OUTPUT SNR':<15}"
    f"{'IMPROVEMENT':<15}"
    f"{'NOISE REDUCTION':<15}"
)

print("-" * 75)


for mu in mu_values:

    for filter_length in filter_lengths:

        enhanced = nlms_filter(
            primary=primary,
            reference=noise,
            mu=mu,
            filter_length=filter_length
        )

        output_snr = calculate_snr(
            enhanced,
            speech
        )

        improvement = output_snr - input_snr

        input_noise_error = primary - speech
        output_noise_error = enhanced - speech

        input_noise_power = np.mean(
            input_noise_error ** 2
        )

        output_noise_power = np.mean(
            output_noise_error ** 2
        )

        noise_reduction = 10 * np.log10(
            input_noise_power /
            (output_noise_power + 1e-12)
        )

        results.append(
            (
                mu,
                filter_length,
                output_snr,
                improvement,
                noise_reduction
            )
        )

        print(
            f"{mu:<10.3f}"
            f"{filter_length:<10}"
            f"{output_snr:<15.2f}"
            f"{improvement:<15.2f}"
            f"{noise_reduction:<15.2f}"
        )


# ==================================================
# 5. FIND BEST RESULT
# ==================================================

best_result = max(
    results,
    key=lambda x: x[2]
)

best_mu = best_result[0]
best_filter = best_result[1]
best_snr = best_result[2]
best_improvement = best_result[3]


print()
print("=" * 75)
print("BEST PARAMETER COMBINATION")
print("=" * 75)

print(f"Best μ              : {best_mu}")
print(f"Best filter length  : {best_filter}")
print(f"Output SNR          : {best_snr:.2f} dB")
print(f"SNR improvement     : {best_improvement:.2f} dB")

print("=" * 75)