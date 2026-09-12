import os
import numpy as np
import torch
import torch.nn as nn
import librosa


# ============================================================
# MODEL ARCHITECTURE
# Must exactly match the training code
# ============================================================

class Conv2dNoiseSuppressor(nn.Module):

    def __init__(self):
        super(Conv2dNoiseSuppressor, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.output = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        feat = self.encoder(x)

        mask = self.sigmoid(
            self.output(feat)
        )

        return mask


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "noise_suppressor_from_scratch.pth"
)


def load_denoiser():

    device = torch.device("cpu")

    model = Conv2dNoiseSuppressor()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# AUDIO DENOISING
# ============================================================

def denoise_audio(
    audio,
    sample_rate,
    model
):

    target_sr = 16000

    # --------------------------------------------------------
    # Resample if necessary
    # --------------------------------------------------------

    if sample_rate != target_sr:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=target_sr
        )

        sample_rate = target_sr

    # --------------------------------------------------------
    # STFT
    # Same parameters used during training
    # --------------------------------------------------------

    n_fft = 512
    hop_length = 256

    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann"
    )

    magnitude = np.abs(stft)

    phase = np.angle(stft)

    # --------------------------------------------------------
    # Convert to PyTorch tensor
    #
    # Training shape:
    # [batch, channel, frequency, time]
    # --------------------------------------------------------

    magnitude_tensor = torch.from_numpy(
        magnitude
    ).float()

    magnitude_tensor = magnitude_tensor.unsqueeze(0).unsqueeze(0)

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        predicted_mask = model(
            magnitude_tensor
        )

    predicted_mask = (
        predicted_mask
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Apply predicted mask
    # Same operation used during training
    # --------------------------------------------------------

    enhanced_magnitude = (
        magnitude * predicted_mask
    )

    # --------------------------------------------------------
    # Reconstruct complex STFT
    # using noisy phase
    # --------------------------------------------------------

    enhanced_stft = (
        enhanced_magnitude
        * np.exp(1j * phase)
    )

    # --------------------------------------------------------
    # Inverse STFT
    # --------------------------------------------------------

    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=hop_length,
        window="hann",
        length=len(audio)
    )

    return enhanced_audio.astype(
        np.float32
    ), sample_rate