import streamlit as st
import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.nlms import nlms_filter
from src.features import extract_features
from src.noise_classifier import classify_noise


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI ANC Walkie-Talkie",
    page_icon="🎙️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎙️ AI-Powered Adaptive ANC Walkie-Talkie")

st.markdown("### Audio Intelligence Dashboard")

st.divider()


# --------------------------------------------------
# AUDIO INPUT
# --------------------------------------------------

st.header("🎧 Audio Input")

col1, col2 = st.columns(2)


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

with col1:

    st.subheader("📁 Upload Audio")

    uploaded_file = st.file_uploader(
        "Choose a WAV file",
        type=["wav"]
    )


# --------------------------------------------------
# LIVE RECORDING
# --------------------------------------------------

with col2:

    st.subheader("🎙️ Live Recording")

    recorded_audio = st.audio_input(
        "Record your voice",
        sample_rate=16000
    )


# --------------------------------------------------
# SELECT AUDIO SOURCE
# --------------------------------------------------

audio_source = None
source_name = None

if uploaded_file is not None:

    audio_source = uploaded_file
    source_name = uploaded_file.name

elif recorded_audio is not None:

    audio_source = recorded_audio
    source_name = "Live Recording"


# --------------------------------------------------
# PROCESS AUDIO
# --------------------------------------------------

if audio_source is not None:

    st.divider()

    st.header("🔊 Audio Analysis")

    st.write(f"**Selected source:** {source_name}")

    # --------------------------------------------------
    # READ AUDIO
    # --------------------------------------------------

    audio_bytes = audio_source.getvalue()

    st.audio(
        audio_bytes,
        format="audio/wav"
    )

    # --------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------

    try:

        audio, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None,
            mono=False
        )
        # --------------------------------------------------
        # NOISE CLASSIFICATION
        # --------------------------------------------------

        # Use primary microphone for classification
        if audio.ndim == 2:
            classification_audio = audio[1]
        else:
            classification_audio = audio

        # Extract 36 audio features
        features = extract_features(
            classification_audio,
            sample_rate
        )

        # Classify the noise
        detected_noise = classify_noise(features)

        st.subheader("🤖 Noise Classification")

        st.metric(
            "Detected Noise",
            detected_noise
        )
        # --------------------------------------------------
        # NLMS ANC PROCESSING
        # --------------------------------------------------

        if audio.ndim == 2:
            st.subheader("🎧 Two-Microphone ANC")

            # Channel 0 = reference microphone
            # Channel 1 = primary microphone

            reference = audio[0]
            primary = audio[1]

            st.write("Reference microphone → Channel 0")
            st.write("Primary microphone → Channel 1")

            # NLMS parameters
            mu = 0.005
            filter_length = 256

            # Run NLMS
            enhanced_audio = nlms_filter(
                primary=primary,
                reference=reference,
                mu=mu,
                filter_length=filter_length
            )

            st.success("NLMS noise cancellation completed!")

            # Save enhanced audio
            output_buffer = io.BytesIO()

            sf.write(
                output_buffer,
                enhanced_audio,
                sample_rate,
                format="WAV"
            )

            output_buffer.seek(0)

            st.subheader("🔊 Enhanced Voice")

            st.audio(
                output_buffer,
                format="audio/wav"
            )   
            # --------------------------------------------------
            # EVALUATION MODE
            # --------------------------------------------------

            st.subheader("⚙️ Evaluation Mode")

            evaluation_mode = st.radio(
                "Select input type:",
                ["Synthetic Evaluation", "Real Recording"]
            )  
            if evaluation_mode == "Synthetic Evaluation":
                # --------------------------------------------------
                # ANC PERFORMANCE METRICS
                # --------------------------------------------------

                st.subheader("📊 ANC Performance")

                # Create the clean speech signal
                t = np.arange(len(primary)) / sample_rate

                clean_speech = (
                    0.5 * np.sin(2 * np.pi * 300 * t)
                    +
                    0.3 * np.sin(2 * np.pi * 600 * t)
                )

                # Make all signals the same length
                min_length = min(
                    len(clean_speech),
                    len(primary),
                    len(enhanced_audio)
                )

                clean_speech = clean_speech[:min_length]
                primary_metric = primary[:min_length]
                enhanced_metric = enhanced_audio[:min_length]

                # Calculate input noise
                input_noise = primary_metric - clean_speech

                # Calculate output noise
                output_noise = enhanced_metric - clean_speech

                # RMS calculation
                clean_rms = np.sqrt(np.mean(clean_speech ** 2))
                input_noise_rms = np.sqrt(np.mean(input_noise ** 2))
                output_noise_rms = np.sqrt(np.mean(output_noise ** 2))

                # SNR calculation
                input_snr = 20 * np.log10(
                    clean_rms / (input_noise_rms + 1e-10)
                )

                output_snr = 20 * np.log10(
                    clean_rms / (output_noise_rms + 1e-10)
                )

                snr_improvement = output_snr - input_snr

                # Noise reduction
                noise_reduction = 20 * np.log10(
                    (input_noise_rms + 1e-10) /
                    (output_noise_rms + 1e-10)
                )

                # Display metric cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Input SNR",
                        f"{input_snr:.2f} dB"
                    )

                with col2:
                    st.metric(
                        "Output SNR",
                        f"{output_snr:.2f} dB"
                    )   

                with col3:
                    st.metric(
                        "SNR Improvement",
                        f"{snr_improvement:.2f} dB"
                    )

                with col4:
                    st.metric(
                        "Noise Reduction",
                        f"{noise_reduction:.2f} dB"
                    )
            else:
                st.info(
                    "Real Recording mode: SNR metrics are disabled "
                    "because the clean speech reference is unknown."
                )
        # --------------------------------------------------
        # DETERMINE CHANNELS
        # --------------------------------------------------

        if audio.ndim == 1:

            channels = 1
            samples = len(audio)

            # Single channel
            primary = audio

        else:

            channels = audio.shape[0]
            samples = audio.shape[1]

            # First channel = reference
            reference = audio[0]

            # Second channel = primary
            primary = audio[1]

        # --------------------------------------------------
        # BASIC INFORMATION
        # --------------------------------------------------

        duration = samples / sample_rate

        if audio.ndim == 1:

            rms = np.sqrt(np.mean(primary ** 2))

            peak = np.max(np.abs(primary))

        else:

            rms = np.sqrt(np.mean(primary ** 2))

            peak = np.max(np.abs(primary))

        # --------------------------------------------------
        # INFORMATION CARDS
        # --------------------------------------------------

        st.subheader("📊 Audio Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Sample Rate",
                f"{sample_rate} Hz"
            )

        with col2:

            st.metric(
                "Channels",
                channels
            )

        with col3:

            st.metric(
                "Duration",
                f"{duration:.2f} sec"
            )

        with col4:

            st.metric(
                "RMS Level",
                f"{rms:.4f}"
            )

        st.success("Audio loaded and analyzed successfully!")

        # --------------------------------------------------
        # PEAK LEVEL
        # --------------------------------------------------

        st.metric(
            "Peak Amplitude",
            f"{peak:.4f}"
        )

        st.divider()

        # ==================================================
        # WAVEFORM
        # ==================================================

        st.header("📈 Waveform")

        time = np.arange(samples) / sample_rate

        if channels == 1:

            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(time, primary)

            ax.set_title("Input Audio Waveform")

            ax.set_xlabel("Time (seconds)")

            ax.set_ylabel("Amplitude")

            ax.grid(True)

            st.pyplot(fig)

        else:

            # --------------------------------------------------
            # REFERENCE MICROPHONE
            # --------------------------------------------------

            st.subheader("🎙️ Reference Microphone")

            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(time, reference)

            ax.set_title("Reference Noise Signal")

            ax.set_xlabel("Time (seconds)")

            ax.set_ylabel("Amplitude")

            ax.grid(True)

            st.pyplot(fig)

            # --------------------------------------------------
            # PRIMARY MICROPHONE
            # --------------------------------------------------

            st.subheader("🎤 Primary Microphone")

            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(time, primary)

            ax.set_title("Primary / Noisy Speech Signal")

            ax.set_xlabel("Time (seconds)")

            ax.set_ylabel("Amplitude")

            ax.grid(True)

            st.pyplot(fig)

        # ==================================================
        # SPECTROGRAM
        # ==================================================

        st.divider()

        st.header("🌈 Spectrogram")

        # Calculate STFT

        D = librosa.stft(primary)

        magnitude = np.abs(D)

        db = librosa.amplitude_to_db(
            magnitude,
            ref=np.max
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        img = librosa.display.specshow(
            db,
            sr=sample_rate,
            x_axis="time",
            y_axis="hz",
            ax=ax
        )

        ax.set_title(
            "Frequency Spectrum of Input Audio"
        )

        fig.colorbar(
            img,
            ax=ax,
            format="%+2.0f dB"
        )

        st.pyplot(fig)

        # ==================================================
        # INTERPRETATION
        # ==================================================

        st.divider()

        st.header("🧠 Signal Interpretation")

        st.info(
            """
            This stage only analyzes the incoming audio.
            
            In the next stages, the AI model will analyze features
            such as MFCC, spectral characteristics, energy and
            temporal information to identify the type of noise.
            """
        )

    except Exception as e:

        st.error(
            f"Could not process audio: {e}"
        )