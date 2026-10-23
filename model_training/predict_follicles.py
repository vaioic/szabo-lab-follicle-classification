from pathlib import Path

from shared import follicle_detector

image_ids = [
    "276146",
    "276155",
    "2761870",
    "276231",
    "276233",
    "276260",
]

image_folder = "../data/Dataset01/cropped_cleaned/"
model_path = Path(
    "../models_v2/checkpoints/2026-10-22_17-30-18/follicle_detector_final.pth"
)

# Get images
image_folder = Path(image_folder)

img_file_list = []
for id in image_ids:
    img_file_list += list(image_folder.glob(f"{id}*.png"))

print("Predicting boxes")

follicle_detector.predict_follicles(
    model_path,
    img_file_list,
    output_dir="../prediction/2026-10-22_Validation",
    confidence_thresh=0.5,
)

print("DONE")
