import soundfile as sf
import numpy as np

from src.pth_denoiser import load_denoiser, denoise_audio


# -------------------------------------------------
# 1. Load trained PTH model
# -------------------------------------------------

print("Loading PTH noise suppressor...")

model = load_denoiser()

print("PTH MODEL LOADED SUCCESSFULLY")


# -------------------------------------------------
# 2. Load input audio
# -------------------------------------------------

input_file = "test_audio.wav"

print(f"Loading audio: {input_file}")

audio, sample_rate = sf.read(input_file)


# -------------------------------------------------
# 3. Convert stereo → mono if necessary
# -------------------------------------------------

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)


audio = audio.astype(np.float32)


print("Input sample rate:", sample_rate)
print("Input samples:", len(audio))
print("Duration:", round(len(audio) / sample_rate, 2), "seconds")


# -------------------------------------------------
# 4. Run trained CNN
# -------------------------------------------------

print("Running CNN noise suppression...")

enhanced_audio, output_sample_rate = denoise_audio(
    audio,
    sample_rate,
    model
)


# -------------------------------------------------
# 5. Save enhanced audio
# -------------------------------------------------

output_file = "enhanced_pth.wav"

sf.write(
    output_file,
    enhanced_audio,
    output_sample_rate
)


print()
print("====================================")
print("PTH DENOISING COMPLETE")
print("====================================")
print("Output file:", output_file)
print("Output sample rate:", output_sample_rate)
print("Output samples:", len(enhanced_audio))
print("====================================")