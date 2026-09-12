import os
import librosa
import soundfile as sf


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

INPUT_DIR = "data/raw/noise"
OUTPUT_DIR = "dataset_normalized"

TARGET_SAMPLE_RATE = 16000


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# PROCESS EACH NOISE CATEGORY
# --------------------------------------------------

categories = [
    "construction",
    "conversation",
    "engine",
    "horn",
    "mixed",
    "traffic",
    "wind"
]


for category in categories:

    input_category = os.path.join(
        INPUT_DIR,
        category
    )

    output_category = os.path.join(
        OUTPUT_DIR,
        category
    )

    os.makedirs(
        output_category,
        exist_ok=True
    )

    if not os.path.exists(input_category):
        print(f"Skipping missing folder: {category}")
        continue

    files = os.listdir(input_category)

    print()
    print("----------------------------------")
    print(f"Processing: {category}")
    print("----------------------------------")

    for filename in files:

        input_file = os.path.join(
            input_category,
            filename
        )

        # Only process audio files
        if not filename.lower().endswith(
            (".wav", ".mp3", ".m4a")
        ):
            continue

        try:

            # Load audio
            audio, sample_rate = librosa.load(
                input_file,
                sr=TARGET_SAMPLE_RATE,
                mono=True
            )

            # Create output filename
            base_name = os.path.splitext(filename)[0]

            output_file = os.path.join(
                output_category,
                base_name + ".wav"
            )

            # Save as 16 kHz mono WAV
            sf.write(
                output_file,
                audio,
                TARGET_SAMPLE_RATE
            )

            print(
                f"✓ {filename} → "
                f"{base_name}.wav"
            )

        except Exception as e:

            print(
                f"✗ Could not process "
                f"{filename}"
            )

            print(
                f"  Error: {e}"
            )


print()
print("==================================")
print("DATASET PREPROCESSING COMPLETE")
print("==================================")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Sample rate     : {TARGET_SAMPLE_RATE} Hz")
print("Channels        : Mono")
print("Format          : WAV")
print("==================================")