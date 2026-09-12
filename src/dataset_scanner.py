import os
import librosa


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

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

AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".m4a"
)


# --------------------------------------------------
# SCAN DATASET
# --------------------------------------------------

print()
print("==========================================")
print("        DATASET SCANNER")
print("==========================================")


total_files = 0
successful_files = 0
failed_files = 0


for category in CATEGORIES:

    category_path = os.path.join(
        DATASET_DIR,
        category
    )

    print()
    print("------------------------------------------")
    print(f"Category: {category}")
    print("------------------------------------------")

    if not os.path.exists(category_path):
        print("Folder not found!")
        continue

    files = os.listdir(category_path)

    category_count = 0

    for filename in files:

        if not filename.lower().endswith(
            AUDIO_EXTENSIONS
        ):
            continue

        total_files += 1
        category_count += 1

        filepath = os.path.join(
            category_path,
            filename
        )

        try:

            audio, sample_rate = librosa.load(
                filepath,
                sr=None,
                mono=False
            )

            if audio.ndim == 1:
                channels = 1
                samples = len(audio)
            else:
                channels = audio.shape[0]
                samples = audio.shape[1]

            duration = samples / sample_rate

            successful_files += 1

            print(
                f"✓ {filename}"
            )

            print(
                f"   Sample rate : {sample_rate} Hz"
            )

            print(
                f"   Channels    : {channels}"
            )

            print(
                f"   Duration    : {duration:.2f} sec"
            )

        except Exception as e:

            failed_files += 1

            print(
                f"✗ {filename}"
            )

            print(
                f"   ERROR: {e}"
            )

    print(
        f"Files in category: {category_count}"
    )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print("==========================================")
print("             SCAN COMPLETE")
print("==========================================")

print(
    f"Total audio files : {total_files}"
)

print(
    f"Successfully read : {successful_files}"
)

print(
    f"Failed files      : {failed_files}"
)

print("==========================================")