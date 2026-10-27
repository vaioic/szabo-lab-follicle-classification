from pathlib import Path

from shared import follicle_detector

image_ids = [
    "276241",
    "276242",
    "276158",
    "276157",
]

image_folder = "../data/Dataset01/cropped_cleaned/"
model_path = Path(
    "../models_v3/checkpoints/2026-10-23_16-01-51round1_16px_augment/follicle_detector_final.pth"
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
    output_dir="../prediction/2026-10-26",
    confidence_thresh=0.5,
)

print("DONE")
