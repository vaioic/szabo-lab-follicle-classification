from shared import follicle_detector

print("Training model...")
model_path = follicle_detector.train(
    image_folder="../data/Dataset01/cropped_cleaned",
    annotation_folder="../data/Dataset01/training_data/round1/train",
    validation_annotation_folder="../data/Dataset01/training_data/round1/validation",
    base_output_dir="../models_v3",
    anchor_sizes=(
        (16,),
        (32,),
        (128,),
        (500,),
        (900,),
    ),
    num_epochs=25,
    model_name="round1_16px_augment",
    augment=True,
)
print("DONE")
