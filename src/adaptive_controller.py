ANC_CONFIG = {

    "construction": {
        "mu": 0.005,
        "filter_length": 256,
        "mode": "Aggressive Adaptive"
    },

    "conversation": {
        "mu": 0.002,
        "filter_length": 128,
        "mode": "Speech Preserving"
    },

    "engine": {
        "mu": 0.005,
        "filter_length": 256,
        "mode": "Strong Adaptive"
    },

    "horn": {
        "mu": 0.008,
        "filter_length": 256,
        "mode": "Fast Transient"
    },

    "mixed": {
        "mu": 0.004,
        "filter_length": 256,
        "mode": "General Adaptive"
    },

    "traffic": {
        "mu": 0.005,
        "filter_length": 256,
        "mode": "Strong Adaptive"
    },

    "wind": {
        "mu": 0.003,
        "filter_length": 128,
        "mode": "Low-Frequency Adaptive"
    }
}


def get_anc_config(noise_class):

    noise_class = noise_class.lower()

    if noise_class not in ANC_CONFIG:
        noise_class = "mixed"

    return ANC_CONFIG[noise_class]