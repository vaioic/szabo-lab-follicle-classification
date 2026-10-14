from shared import follicle_detector

image_paths = [
    "../data/Dataset01/raw/272940.svs",
    "../data/Dataset01/raw/272941.svs",
]

csv_paths = [
    "../data/Dataset01/qupath/export/272940_bounding_boxes.csv",
    "../data/Dataset01/qupath/export/272941_bounding_boxes.csv",
]

model_path = follicle_detector.train_follicle_detector(
    image_paths, csv_paths, output_dir="../models/2026-10-14_Dev"
)


# Try predicting

validation_image_paths = [
    "../data/Dataset01/raw/272944.svs",
    "../data/Dataset01/raw/272946.svs",
]

follicle_detector.predict_follicles(
    model_path,
    validation_image_paths,
    output_dir="../models/2026-10-14_Dev/Prediction",
    confidence_thresh=0.5,
)
