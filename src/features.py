import numpy as np
import librosa


def extract_features(audio, sample_rate):
    """
    Extract audio features for AI noise classification.
    """

    # Make sure audio is a NumPy array
    audio = np.asarray(audio, dtype=np.float32)

    # --------------------------------------------------
    # 1. MFCC
    # --------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # --------------------------------------------------
    # 2. RMS ENERGY
    # --------------------------------------------------

    rms = librosa.feature.rms(y=audio)

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # --------------------------------------------------
    # 3. ZERO CROSSING RATE
    # --------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(audio)

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # --------------------------------------------------
    # 4. SPECTRAL CENTROID
    # --------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # --------------------------------------------------
    # 5. SPECTRAL BANDWIDTH
    # --------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )

    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    # --------------------------------------------------
    # 6. SPECTRAL ROLLOFF
    # --------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )

    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # --------------------------------------------------
    # COMBINE ALL FEATURES
    # --------------------------------------------------

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [
            rms_mean,
            rms_std,
            zcr_mean,
            zcr_std,
            centroid_mean,
            centroid_std,
            bandwidth_mean,
            bandwidth_std,
            rolloff_mean,
            rolloff_std
        ]
    ])

    return features