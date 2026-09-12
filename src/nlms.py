import numpy as np


def nlms_filter(
    primary,
    reference,
    mu=0.01,
    filter_length=128,
    epsilon=1e-8
):
    """
    Normalized Least Mean Squares (NLMS) adaptive filter.

    primary:
        Speech + noise signal

    reference:
        Reference noise signal

    mu:
        Adaptation step size

    filter_length:
        Number of adaptive filter taps

    returns:
        error_signal
        This is the estimated cleaned/enhanced signal.
    """

    # Convert inputs to NumPy arrays
    primary = np.asarray(primary, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    # Make both signals the same length
    n_samples = min(
        len(primary),
        len(reference)
    )

    primary = primary[:n_samples]
    reference = reference[:n_samples]

    # Adaptive filter weights
    weights = np.zeros(
        filter_length,
        dtype=np.float32
    )

    # Output signal
    output = np.zeros(
        n_samples,
        dtype=np.float32
    )

    # Reference signal buffer
    reference_buffer = np.zeros(
        filter_length,
        dtype=np.float32
    )

    # --------------------------------------------------
    # NLMS PROCESSING
    # --------------------------------------------------

    for n in range(n_samples):

        # Shift previous samples
        reference_buffer[1:] = reference_buffer[:-1]

        # Insert newest reference sample
        reference_buffer[0] = reference[n]

        # Estimate noise
        estimated_noise = np.dot(
            weights,
            reference_buffer
        )

        # Calculate error
        error = primary[n] - estimated_noise

        # Normalize adaptation
        normalization = (
            np.dot(
                reference_buffer,
                reference_buffer
            )
            + epsilon
        )

        # Update filter weights
        weights += (
            mu
            * error
            * reference_buffer
            / normalization
        )

        # Store cleaned signal
        output[n] = error

    return output