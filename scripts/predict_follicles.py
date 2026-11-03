from pathlib import Path

from shared import follicle_detector

image_ids = [
    "276149",
    "276148",
    "276167",
    "276168",
    "276169",
    "276272",
    "276171",
    "276270",
    "276172",
    "276273",
    "276211",
    "276175",
    "276176",
]

image_folder = "../data/dataset_01/images"
model_path = Path(
    "../models/checkpoints/2026-11-02_16-13-337classes_16px_augment/follicle_detector_final.pth"
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
    output_dir="../predictions/2026-11-03",
    confidence_thresh=0.5,
)

print("DONE")
