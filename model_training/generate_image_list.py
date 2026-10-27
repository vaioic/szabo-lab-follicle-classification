# Script to generate a list of images

from pathlib import Path

import pandas as pd

image_folder = Path("../data/Dataset01/cropped_cleaned")
output_folder = Path("../data/Dataset01")

reference_file = Path("../data/Dataset01/image_list.csv")

annotation_folder = Path("../data/Dataset01/qupath_cropped/export")

ref_data = pd.read_csv(reference_file)

image_list = list(image_folder.glob("*.png"))

merged_df = pd.DataFrame(
    [
        {
            "Image": img.stem,
            "Image path": str(img),
            "Image ID": int(((img.stem).split("_"))[0]),
        }
        for img in image_folder.glob("*.png")
    ]
)

# Merge the two databases together to get Slide ID and Barcode ID
combined_df = pd.merge(ref_data, merged_df, on="Image ID", how="right")


# Check if an annotation already exists
def check_annotation_exists(image_name):
    chk_path = annotation_folder / f"{image_name}_bounding_boxes.csv"
    return chk_path.is_file()


combined_df["Annotated"] = combined_df["Image"].apply(check_annotation_exists)

combined_df.to_excel((output_folder / "dataset01_image_list_regions.xlsx"), index=False)
