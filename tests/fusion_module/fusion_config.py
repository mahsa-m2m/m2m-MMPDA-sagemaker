# LABEL MAPPING
CLASS_LABELS = {
    0: "Truthful",
    1: "Deceptive"
}

# FUSION WEIGHTS
# How much trust to place in each model.
FUSION_WEIGHTS = {
    "video": 0.8,
    "audio": 0.2
}