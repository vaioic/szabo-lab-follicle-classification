from shared import follicle_detector

print("Training model...")
model_path = follicle_detector.train(
    image_folder="../data/Dataset01/cropped_cleaned",
    annotation_folder="../data/Dataset01/training_data/round2/train",
    validation_annotation_folder="../data/Dataset01/training_data/round2/validation",
    base_output_dir="../models_v3",
    checkpoint_path="../models_v3/checkpoints/2026-10-23_16-01-51round1_16px_augment/follicle_detector_final.pth",
    anchor_sizes=(
        (16,),
        (32,),
        (128,),
        (500,),
        (900,),
    ),
    num_epochs=10,
    model_name="round2_16px_augment",
    augment=True,
    learning_rate=0.0005,
)
print("DONE")
