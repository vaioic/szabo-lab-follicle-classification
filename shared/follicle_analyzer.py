from pathlib import Path

import pandas as pd


def count_follicles(annotation_folder, image_csv):

    annotation_folder = Path(annotation_folder)

    csv_files = list(annotation_folder.glob("*.csv"))

    for f in csv_files:
        curr_df = pd.read_csv(f)

        # Extract image info from filename
        matches = (f.name).split("_")
        image_id = matches[0]
        region_id = matches[1].replace("region", "")

        curr_df["Image ID"] = int(image_id)
        curr_df["Region"] = int(region_id)

        break

    # Now read in the image csv file
    df_image = pd.read_csv(image_csv)

    merged_df = pd.merge(curr_df, df_image, on="Image ID", how="inner")

    print(merged_df)

    class_counts = (
        merged_df.groupby(["Region", "Barcode ID", "Class"])
        .size()
        .reset_index(name="Count")
    )

    print(class_counts)


# TODO: Concatenate with Barcode ID
# TODO: Update the master image list with some way to link the tissue slices
