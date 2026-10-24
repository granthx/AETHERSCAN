"""COVID-19 CT-scan classifier (transfer learning, MobileNetV3)."""

__version__ = "0.1.0"

# Class folder names used by the Kaggle dataset "CT Scans for COVID-19 Classification"
CLASS_NAMES = ["NiCT", "nCT", "pCT"]  # alphabetical = ImageFolder order
CLASS_DESCRIPTIONS = {
    "nCT": "Normal CT (no COVID findings)",
    "pCT": "COVID-19 positive CT",
    "NiCT": "Non-informative CT (slice shows no usable lung tissue)",
}
