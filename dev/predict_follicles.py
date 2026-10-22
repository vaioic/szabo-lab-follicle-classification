from pathlib import Path

from shared import follicle_detector

image_ids = ["276057", "276058", "276197"]

image_folder = "../data/Dataset01/cropped_cleaned/"
model_path = "../models/2026-10-21_16-29-05/follicle_detector_final.pth"

# Get images
image_folder = Path(image_folder)

img_file_list = []
for id in image_ids:
    img_file_list += list(image_folder.glob(f"{id}*.png"))

print("Predicting boxes")

follicle_detector.predict_follicles(
    model_path,
    img_file_list,
    output_dir="../prediction/2026-10-22_Round2",
    confidence_thresh=0.3,
)

print("DONE")
