import io
import os

import joblib
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from src.features import extract_features
from src.nlms import nlms_filter
from src.adaptive_controller import get_anc_config
import random
from src.audio_input import prepare_two_channel_audio
from src.pth_denoiser import load_denoiser, denoise_audio

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Adaptive ANC Walkie-Talkie",
    page_icon="🎙️",
    layout="wide"
)

# ========================================================
# DASHBOARD DESIGN / CUSTOM CSS
# ========================================================

st.markdown(
"""

<style>
/* ==================================================
   TEAM BRANDING
   ================================================== */

.team-brand {
    position: fixed;
    top: 18px;
    left: 25px;
    z-index: 999999;

    font-size: 20px;
    font-weight: 800;
    letter-spacing: 2px;

    padding: 8px 14px;

    border: 1px solid rgba(100, 150, 255, 0.45);
    border-radius: 10px;

    background: rgba(20, 30, 50, 0.08);

    backdrop-filter: blur(8px);

    text-shadow:
        0 0 6px rgba(80, 150, 255, 0.6),
        0 0 14px rgba(80, 150, 255, 0.35);

    animation: teamGlow 2.5s ease-in-out infinite alternate;
}


/* Subtle glowing animation */

@keyframes teamGlow {

    from {
        box-shadow:
            0 0 4px rgba(80, 150, 255, 0.25);
    }

    to {
        box-shadow:
            0 0 14px rgba(80, 150, 255, 0.55),
            0 0 28px rgba(80, 150, 255, 0.20);
    }

}
/* ==================================================
   GLOBAL FONT
   ================================================== */

html, body, [class*="st-"] {
    font-family: "Segoe UI", Arial, sans-serif;
}


/* ==================================================
   PROJECT HEADER
   ================================================== */

.project-header {
    width: 100%;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 35px;
}

.project-title {
    font-size: 42px !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    margin-bottom: 12px !important;
}

.project-subtitle {
    font-size: 24px !important;
    font-weight: 650 !important;
    margin-bottom: 12px !important;
}

.project-description {
    max-width: 950px;
    margin: 0 auto;
    font-size: 17px !important;
    line-height: 1.6 !important;
    opacity: 0.78;
}


/* ==================================================
   SECTION HEADINGS
   ================================================== */

h2 {
    font-size: 30px !important;
    font-weight: 750 !important;
}

h3 {
    font-size: 24px !important;
    font-weight: 700 !important;
}


/* ==================================================
   PTT CONTROL
   ================================================== */

div[data-testid="stToggle"] {
    transform: scale(1.7);
    transform-origin: left center;
    margin-top: 15px;
    margin-bottom: 30px;
}

div[data-testid="stToggle"] label {
    font-size: 18px !important;
    font-weight: 700 !important;
}


/* ==================================================
   METRIC LABELS
   ================================================== */

div[data-testid="stMetricLabel"] {
    margin-bottom: 4px !important;
}

div[data-testid="stMetricLabel"] p {
    font-size: 19px !important;
    font-weight: 750 !important;
    letter-spacing: 0.3px !important;
}


/* ==================================================
   METRIC VALUES
   ================================================== */

div[data-testid="stMetricValue"] {
    margin-top: 0px !important;
}

div[data-testid="stMetricValue"] div {
    font-size: 25px !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
}


/* ==================================================
   CAPTIONS
   ================================================== */

div[data-testid="stCaptionContainer"] p {
    font-size: 15px !important;
}


/* ==================================================
   ALERT BOXES
   ================================================== */

div[data-testid="stAlert"] {
    border-radius: 10px !important;
}

div[data-testid="stAlert"] p {
    font-size: 16px !important;
}


/* ==================================================
   RADIO BUTTONS
   ================================================== */

div[data-testid="stRadio"] label {
    font-size: 16px !important;
}


/* ==================================================
   SELECTBOX
   ================================================== */

div[data-baseweb="select"] {
    font-size: 16px !important;
}


/* ==================================================
   DIVIDERS
   ================================================== */

hr {
    margin-top: 30px !important;
    margin-bottom: 30px !important;
}

</style>
""",
unsafe_allow_html=True
)

# =========================
# TEAM BRANDING
# =========================

st.markdown(
    '<div class="team-brand">BETACRAFTERS</div>',
    unsafe_allow_html=True
)

# ========================================================
# PROJECT HEADER
# ========================================================

st.markdown(
"""
<div class="project-header">

<div class="project-title">
🎙️ AI-Powered Adaptive Noise Cancellation
</div>

<div class="project-subtitle">
AI ANC Walkie-Talkie
</div>

<div class="project-description">
Clear Speech Communication in Noisy Environments.
This prototype uses AI-based noise classification
followed by adaptive NLMS noise cancellation to
improve speech intelligibility.
</div>

</div>
""",
unsafe_allow_html=True
)
# ============================================================
# LOAD ML MODEL
# ============================================================

MODEL_PATH = "models/noise_classifier.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Noise classifier model not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
@st.cache_resource
def load_pth_model():
    return load_denoiser()

pth_model = load_pth_model()
@st.cache_resource
def load_pth_model():
    return load_denoiser()

pth_model = load_pth_model()

# ------------------------------------------------------------
# WALKIE-TALKIE STATUS
# ------------------------------------------------------------

st.subheader("Walkie-Talkie Status")

ptt_pressed = st.toggle(
    "Push-to-Talk (PTT)",
    value=False
)

if ptt_pressed:
    tx_rx_status = "TRANSMITTING"
else:
    tx_rx_status = "RECEIVING / IDLE"

st.metric(
    "TX/RX Status",
    tx_rx_status
)
st.success("🟢 SYSTEM READY")


# ============================================================
# PROJECT / TEAM
# ============================================================

st.caption("AI ANC Walkie-Talkie • Adaptive Noise Cancellation Prototype")


# ============================================================
# INPUT SOURCE
# ============================================================

st.subheader("🎤 Input Source")

input_mode = st.radio(
    "Select input source:",
    [
        "Upload Audio",
        "Microphone",
        "Two-Microphone Demo"
    ],
    horizontal=True
)


# ============================================================
# AUDIO INPUT
# ============================================================

audio = None
sample_rate = None
audio_bytes = None
is_stereo_anc = False


# ------------------------------------------------------------
# UPLOAD AUDIO
# ------------------------------------------------------------

if input_mode == "Upload Audio":

    uploaded_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"]
    )

    if uploaded_file is not None:

        audio_bytes = uploaded_file.read()

        audio, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None,
            mono=False
        )


# ------------------------------------------------------------
# MICROPHONE
# ------------------------------------------------------------

elif input_mode == "Microphone":

    recorded_audio = st.audio_input(
        "Record your voice",
        sample_rate=16000
    )

    if recorded_audio is not None:

        audio_bytes = recorded_audio.getvalue()

        audio, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None,
            mono=False
        )


# ------------------------------------------------------------
# TWO MICROPHONE DEMO
# ------------------------------------------------------------

elif input_mode == "Two-Microphone Demo":

    st.info(
        "Two-channel ANC demonstration: "
        "Channel 0 = reference microphone, "
        "Channel 1 = primary microphone."
    )

    st.write("### Software Simulation of Final Two-Microphone Architecture")

    st.caption(
        "This software demonstration simulates the final hardware architecture. "
        "The Raspberry Pi will later provide the two INMP441 microphone channels."
    )

    noise_class_demo = st.selectbox(
        "Select background noise",
        [
            "engine",
            "traffic",
            "construction",
            "conversation",
            "horn",
            "mixed",
            "wind"
        ]
    )

    speech_recording = st.audio_input(
        "Record your speech"
    )

    if speech_recording is not None:

        speech_bytes = speech_recording.getvalue()

        speech_audio, speech_sr = librosa.load(
            io.BytesIO(speech_bytes),
            sr=16000,
            mono=True
        )
        noise_folder = os.path.join(
            "demo_noise",
            noise_class_demo
        )
        if not os.path.isdir(noise_folder):
            st.error(
                f"Demo noise folder not found: {noise_folder}. "
                "Please add the demo noise files to the repository."
            )
            st.stop()

        noise_files = [
            os.path.join(noise_folder, f)
            for f in os.listdir(noise_folder)
            if f.lower().endswith(".wav")
        ]

        if not noise_files:
            st.error("No WAV files found for this noise class.")
            st.stop()

        noise_file = random.choice(noise_files)

        noise_audio, noise_sr = librosa.load(
            noise_file,
            sr=16000,
            mono=True
        )

        # Match durations
        duration_samples = min(
            len(speech_audio),
            len(noise_audio)
        )

        speech_audio = speech_audio[:duration_samples]
        noise_audio = noise_audio[:duration_samples]

        # Normalize levels
        speech_audio = speech_audio / (
            np.max(np.abs(speech_audio)) + 1e-8
        )

        noise_audio = noise_audio / (
            np.max(np.abs(noise_audio)) + 1e-8
        )

        # ------------------------------------------------------------
        # CREATE SIMULATED TWO-MICROPHONE SIGNAL
        # ------------------------------------------------------------

        reference = noise_audio

        noise_level = 0.35

        primary = (
            speech_audio +
            noise_level * noise_audio
        )

        audio = prepare_two_channel_audio(
            reference,
            primary
        )

        # Keep the clean speech and injected noise available
        # for quantitative software-demo evaluation.
        clean_speech_demo = speech_audio.copy()
        noise_component_demo = noise_level * noise_audio

        sample_rate = 16000
        is_stereo_anc = True

        st.success(
            f"Simulated two-microphone signal created "
            f"using {noise_class_demo} noise."
        )


# ============================================================
# PROCESS AUDIO
# ============================================================

if audio is not None:

    st.divider()

    # --------------------------------------------------------
    # AUDIO INFORMATION
    # --------------------------------------------------------

    if audio.ndim == 1:

        mono_audio = audio

        channels = 1

    else:

        # librosa returns:
        # channels × samples

        channels = audio.shape[0]

        mono_audio = np.mean(audio, axis=0)


    duration = len(mono_audio) / sample_rate


    # --------------------------------------------------------
    # INPUT INFORMATION
    # --------------------------------------------------------

    st.subheader("📊 Input Signal")

    col1, col2, col3 = st.columns(3)

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
            f"{duration:.2f} s"
        )


    # --------------------------------------------------------
    # ORIGINAL AUDIO
    # --------------------------------------------------------

    st.subheader("🔊 Original / Noisy Audio")

    input_buffer = io.BytesIO()

    sf.write(
        input_buffer,
        mono_audio,
        sample_rate,
        format="WAV"
    )

    input_buffer.seek(0)

    st.audio(
        input_buffer,
        format="audio/wav"
    )


    # ========================================================
    # AI NOISE CLASSIFICATION
    # ========================================================

    st.divider()

    st.subheader("🤖 AI Noise Intelligence")

    try:

        # For two-microphone ANC, classify the reference/noise channel.
        # The classifier was trained on noise recordings.
        if is_stereo_anc:
            classification_audio = audio[0]
        else:
            classification_audio = mono_audio

        features = extract_features(
            classification_audio,
            sample_rate
        )

        prediction = model.predict(
            [features]
        )[0]

        probabilities = model.predict_proba(
            [features]
        )[0]

        classes = model.classes_

        confidence = float(
            np.max(probabilities) * 100
        )

    except Exception as e:

        st.error(
            f"Noise classification failed: {e}"
        )

        st.stop()


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Detected Noise",
            prediction.upper()
        )

    with col2:

        st.metric(
            "AI Confidence",
            f"{confidence:.1f}%"
        )


    # ========================================================
    # ADAPTIVE ANC CONTROLLER
    # ========================================================

    config = get_anc_config(prediction)

    st.subheader("⚙️ Adaptive ANC Controller")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Detected Noise",
            prediction.upper()
        )

    with col2:

        st.metric(
            "ANC Mode",
            config["mode"]
        )

    with col3:

        st.metric(
            "Adaptation Rate (μ)",
            config["mu"]
        )

    with col4:

        st.metric(
            "Filter Length",
            f"{config['filter_length']} taps"
        )
    # ========================================================
    # TWO-MICROPHONE SIGNALS
    # ========================================================

    if is_stereo_anc:

        st.divider()

        st.subheader("🎙️ Two-Microphone Signals")

        st.caption(
            "Reference microphone captures environmental noise. "
            "Primary microphone captures speech mixed with noise."
        )

        mic_col1, mic_col2 = st.columns(2)

        # Display a short section for readability
        mic_display_duration = 0.5

        mic_display_samples = min(
            int(mic_display_duration * sample_rate),
            len(audio[0]),
            len(audio[1])
        )

        mic_time = (
            np.arange(mic_display_samples) / sample_rate
        )

        with mic_col1:

            st.write("**Reference Microphone — Noise**")

            fig_ref, ax_ref = plt.subplots(
                figsize=(5, 3)
            )

            ax_ref.plot(
                mic_time,
                audio[0][:mic_display_samples]
            )

            ax_ref.set_xlabel("Time (seconds)")
            ax_ref.set_ylabel("Amplitude")
            ax_ref.set_title("Reference Channel")

            st.pyplot(fig_ref)
            plt.close(fig_ref)

        with mic_col2:

            st.write("**Primary Microphone — Speech + Noise**")

            fig_primary, ax_primary = plt.subplots(
                figsize=(5, 3)
            )

            ax_primary.plot(
                mic_time,
                audio[1][:mic_display_samples]
            )

            ax_primary.set_xlabel("Time (seconds)")
            ax_primary.set_ylabel("Amplitude")
            ax_primary.set_title("Primary Channel")

            st.pyplot(fig_primary)
            plt.close(fig_primary)
    
    # ========================================================
    # ANC PROCESSING
    # ========================================================

    st.divider()

    st.subheader("🎧 Noise Cancellation")


    if is_stereo_anc:

        reference = audio[0]
        primary = audio[1]

        enhanced_audio = nlms_filter(
            primary=primary,
            reference=reference,
            mu=config["mu"],
            filter_length=config["filter_length"]
        )
        # PTH CNN noise suppression
        enhanced_audio_pth, pth_sample_rate = denoise_audio(
            primary,
            sample_rate,
            pth_model
        )

        # Use PTH CNN output as the final enhanced speech
        enhanced_audio = enhanced_audio_pth
        sample_rate = pth_sample_rate
        st.info(
            f"PTH CNN active • Output samples: {len(enhanced_audio_pth)}"
        )

        # ------------------------------------------------------------
        # ANC PERFORMANCE METRICS
        # ------------------------------------------------------------

        # Known signals in the controlled software simulation
        clean_reference = clean_speech_demo
        input_noise = noise_component_demo

        # Match lengths
        metric_length = min(
            len(clean_reference),
            len(input_noise),
            len(enhanced_audio)
        )

        clean_reference = clean_reference[:metric_length]
        input_noise = input_noise[:metric_length]
        enhanced_metric = enhanced_audio[:metric_length]


        def calculate_snr(signal, noise):
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)

            return 10 * np.log10(
                (signal_power + 1e-12) /
                (noise_power + 1e-12)
            )


        # Input SNR 
        input_snr = calculate_snr(
            clean_reference,
            input_noise
        )

        # Estimate residual noise after ANC
        residual_noise = enhanced_metric - clean_reference

        # Output SNR
        output_snr = calculate_snr(
            clean_reference,
            residual_noise
        )

        # Improvement
        snr_improvement = output_snr - input_snr

        # Noise attenuation
        input_noise_power = np.mean(input_noise ** 2)
        residual_noise_power = np.mean(residual_noise ** 2)

        anc_attenuation = 10 * np.log10(
            (input_noise_power + 1e-12) /
            (residual_noise_power + 1e-12)
        )

        st.success(
             "✅ AI noise suppression using trained PTH CNN completed."
        )
        st.caption(
             "Primary microphone → STFT → trained CNN spectral mask "
            "→ enhanced speech"
        )

    else:

        st.warning(
            "Single-channel input detected. "
            "AI noise classification is available, but true ANC "
            "requires separate reference and primary microphone signals."
        )

        st.caption(
            "For the final hardware prototype, the Raspberry Pi "
            "will provide the two INMP441 microphone channels."
        )
        # For tonight's software demo, preserve the original
        # signal rather than falsely claiming ANC processing.

        enhanced_audio = mono_audio.copy()

    # ------------------------------------------------------------
    # ANC PERFORMANCE
    # ------------------------------------------------------------

    if is_stereo_anc:

        st.subheader("ANC Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("INPUT SNR", f"{input_snr:.2f} dB")

        with col2:
            st.metric("OUTPUT SNR", f"{output_snr:.2f} dB")

        with col3:
            st.metric("SNR IMPROVEMENT", f"{snr_improvement:+.2f} dB")

        with col4:
            st.metric("ANC ATTENUATION", f"{anc_attenuation:.2f} dB")

    else:

        st.subheader("ANC Performance")

        st.info(
            "Quantitative ANC metrics are available only for "
            "the controlled two-microphone demonstration, "
            "where the clean speech and injected noise are known."
        )
    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    st.divider()

    st.subheader("🟢 System Status")

    status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)

    with status_col1:
        st.metric(
            "Input",
            "2-MIC ACTIVE" if is_stereo_anc else "SINGLE MIC"
        )

    with status_col2:
        st.metric(
            "AI",
            "CLASSIFIED"
        )

    with status_col3:
        st.metric(
            "ANC",
            "ACTIVE" if is_stereo_anc else "STANDBY"
        )

    with status_col4:
        st.metric(
            "NLMS",
            "RUNNING" if is_stereo_anc else "STANDBY"
        )

    with status_col5:
        
        st.metric(
            "TX/RX",
            "TRANSMITTING" if ptt_pressed else "RECEIVING"
        )
    # ========================================================
    # ENHANCED AUDIO
    # ========================================================

    st.subheader("🔊 Enhanced Speech")

    output_buffer = io.BytesIO()

    sf.write(
        output_buffer,
        enhanced_audio,
        sample_rate,
        format="WAV"
    )

    output_buffer.seek(0)

    st.audio(
        output_buffer,
        format="audio/wav"
    )


    # ========================================================
    # WAVEFORM COMPARISON
    # ========================================================

    st.divider()

    st.subheader("📈 Signal Comparison")

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    # Display only the first 0.75 seconds so the waveform
    # is visually readable.
    display_duration = 0.75

    display_samples = min(
        int(display_duration * sample_rate),
        len(mono_audio),
        len(enhanced_audio)
    )

    time_original = (
        np.arange(display_samples) / sample_rate
    )

    time_enhanced = (
        np.arange(display_samples) / sample_rate
    )
    # Use the primary microphone signal for the ANC comparison.
    if is_stereo_anc:
        waveform_original = audio[1]
    else:
        waveform_original = mono_audio

    ax.plot(
        time_original,
        waveform_original[:display_samples],
        label="Original / Noisy"
    )

    ax.plot(
        time_enhanced,
        enhanced_audio[:display_samples],
        label="Enhanced"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Original vs Enhanced Waveform")

    ax.legend()

    st.pyplot(fig)


    # ========================================================
    # SPECTROGRAM COMPARISON
    # ========================================================

    st.subheader("🔬 Spectrogram Comparison")

    col1, col2 = st.columns(2)

    with col1:

        st.write("Original / Noisy")

        fig1, ax1 = plt.subplots(
            figsize=(7, 4)
        )

        spectrogram = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(waveform_original)
            ),
            ref=np.max
        )

        librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            x_axis="time",
            y_axis="hz",
            ax=ax1
        )

        ax1.set_ylim(0, 2000)
        ax1.set_title("Primary / Noisy Spectrogram")

        st.pyplot(fig1)


    with col2:

        st.write("Enhanced")

        fig2, ax2 = plt.subplots(
            figsize=(7, 4)
        )

        enhanced_spectrogram = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(enhanced_audio)
            ),
            ref=np.max
        )

        librosa.display.specshow(
            enhanced_spectrogram,
            sr=sample_rate,
            x_axis="time",
            y_axis="hz",
            ax=ax2
        )

        ax2.set_ylim(0, 2000)
        ax2.set_title("Enhanced Spectrogram")

        st.pyplot(fig2)


    # ========================================================
    # SYSTEM PIPELINE
    # ========================================================

    st.divider()

    st.subheader("🔄 AI-ANC Processing Pipeline")

    st.markdown(
        """
        **🎤 Microphones / Audio Input**
        → **🔎 Feature Extraction**
        → **🤖 AI Noise Classification**
        → **⚙️ Adaptive ANC Controller**
        → **🧠 NLMS Adaptive Filter**
        → **🔊 Enhanced Speech**
        """
    )
    st.info(
        f"AI detected **{prediction.upper()}** noise with "
        f"**{confidence:.1f}% confidence**. "
        f"The adaptive controller selected **{config['mode']}** "
        f"with μ = **{config['mu']}** and a "
        f"**{config['filter_length']}-tap NLMS filter**."
    )

    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "AI ANC Walkie-Talkie • Software prototype ready for "
        "Raspberry Pi 4 + dual INMP441 hardware integration"
    )