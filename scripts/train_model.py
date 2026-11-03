from shared import follicle_detector

# print("Training model...")

# path = Path("../data/dataset_01/annotations/train")

# files = (p for p in path.glob("*.csv") if p.is_file())
# first_10_files = list(islice(files, 10))

# for f in first_10_files:
#     print(f)


model_path = follicle_detector.train(
    image_folder="../data/dataset_01/images",
    annotation_folder="../data/dataset_01/annotations/train",
    validation_annotation_folder="../data/dataset_01/annotations/validation",
    base_output_dir="../models",
    checkpoint_path=None,
    anchor_sizes=(
        (16,),
        (32,),
        (128,),
        (500,),
        (900,),
    ),
    num_epochs=20,
    model_name="7classes_16px_augment",
    augment=True,
    learning_rate=0.001,
    num_classes=10,
)
print("DONE")
