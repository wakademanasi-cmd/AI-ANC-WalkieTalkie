import numpy as np


def prepare_two_channel_audio(reference, primary):
    """
    Prepare two microphone signals for the ANC pipeline.

    Channel 0 = reference microphone
    Channel 1 = primary microphone

    Both signals are converted to float32 and
    trimmed to the same length.
    """

    reference = np.asarray(reference, dtype=np.float32)
    primary = np.asarray(primary, dtype=np.float32)

    # Make sure both signals have the same length
    n_samples = min(len(reference), len(primary))

    reference = reference[:n_samples]
    primary = primary[:n_samples]

    # Remove DC offset
    reference = reference - np.mean(reference)
    primary = primary - np.mean(primary)

    # Prevent excessively large amplitudes
    max_amplitude = max(
        np.max(np.abs(reference)),
        np.max(np.abs(primary)),
        1e-8
    )

    if max_amplitude > 1.0:
        reference = reference / max_amplitude
        primary = primary / max_amplitude

    # Return exactly the format expected by demo_app.py
    two_channel_audio = np.vstack([
        reference,
        primary
    ])

    return two_channel_audio

def get_hardware_audio():
    """
    Placeholder for Raspberry Pi + dual INMP441 input.

    This function will be connected to the actual
    I2S microphone interface during hardware integration.
    """

    raise NotImplementedError(
        "Hardware microphone input is not connected yet. "
        "Use the software simulation for now."
    )