from shared import follicle_detector

# # Training from scratch
# print("Training model...")
# model_path = follicle_detector.train(
#     "../data/Dataset01/cropped_cleaned",
#     "../data/Dataset01/training_data/round1",
#     "../models",
#     anchor_sizes=(
#         (64,),
#         (96,),
#         (192,),
#         (384,),
#         (786,),
#     ),
#     num_epochs=25,
# )
# print("DONE")

print("Training model...")
model_path = follicle_detector.train(
    "../data/Dataset01/cropped_cleaned",
    "../data/Dataset01/training_data/round1",
    "../models_testing",
    checkpoint_path="../models/2026-10-21_15-41-19/follicle_detector_final.pth",
    anchor_sizes=(
        (64,),
        (96,),
        (192,),
        (384,),
        (786,),
    ),
    num_epochs=3,
    learning_rate=0.005,
)
print("DONE")


# model_path = "../models/2026-10-21_11-07-04/follicle_detector_final.pth"

# print("Predicting boxes")
# # Try prediction
# imgs = [
#     "../data/Dataset01/cropped_cleaned/276222_region0.png",
#     "../data/Dataset01/cropped_cleaned/276260_region0.png",
#     "../data/Dataset01/cropped_cleaned/276172_region0.png",
# ]

# follicle_detector.predict_follicles(
#     model_path,
#     imgs,
#     output_dir="../prediction/2026-10-21_MoreEpochs",
#     confidence_thresh=0.7,
# )

# print("DONE")
