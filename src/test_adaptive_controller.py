from src.adaptive_controller import get_anc_config


noise_classes = [
    "construction",
    "conversation",
    "engine",
    "horn",
    "mixed",
    "traffic",
    "wind"
]


print()
print("==========================================")
print("       ADAPTIVE ANC CONTROLLER TEST")
print("==========================================")


for noise_class in noise_classes:

    config = get_anc_config(noise_class)

    print()
    print(f"Noise Class : {noise_class}")
    print(f"ANC Mode    : {config['mode']}")
    print(f"Mu          : {config['mu']}")
    print(f"Filter Size : {config['filter_length']}")


print()
print("==========================================")
print("       CONTROLLER TEST COMPLETE")
print("==========================================")