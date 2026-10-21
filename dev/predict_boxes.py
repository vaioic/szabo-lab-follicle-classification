from pathlib import Path

from shared import follicle_detector

model_path = "../models/2026-10-19_15-02-25/follicle_detector_final.pth"

# Get images
image_path = Path("../data/Dataset01/cropped")
images = list(image_path.glob("*.png"))

print("Predicting boxes")
# Try prediction
follicle_detector.predict_follicles(
    model_path,
    images,
    output_dir="../prediction/2026-10-19_round2",
)

print("DONE")
