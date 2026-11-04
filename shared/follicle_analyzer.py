from pathlib import Path

import pandas as pd


def count_follicles(annotation_folder, image_csv):

    annotation_folder = Path(annotation_folder)

    csv_files = list(annotation_folder.glob("*.csv"))

    image_df = []
    for f in csv_files:
        curr_df = pd.read_csv(f)

        # Extract image info from filename
        matches = (f.name).split("_")
        image_id = matches[0]
        region_id = matches[1].replace("region", "")

        curr_df["Image ID"] = int(image_id)
        curr_df["Region"] = int(region_id)

        image_df.append(curr_df)

    df_all = pd.concat(image_df)

    # Now read in the image csv file
    df_image = pd.read_csv(image_csv)

    merged_df = pd.merge(df_all, df_image, on="Image ID", how="inner")

    # print(merged_df)

    class_counts = (
        merged_df.groupby(["Slide ID", "Region", "Class"])
        .size()
        .reset_index(name="Count")
    )

    summary_df = class_counts.pivot_table(
        index=["Slide ID", "Region"],
        columns="Class",
        values="Count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    expected_classes = ["Primordial", "Activated", "Primary 3a", "Primary 3b", "Antral"]
    for col in expected_classes:
        if col not in summary_df.columns:
            summary_df[col] = 0

    final_columns = ["Slide ID", "Region"] + expected_classes
    summary_df = summary_df[final_columns]

    summary_df.to_csv(
        Path("../processed") / "2026-11-04 Counts_Pivoted.csv", index=False
    )

    # print(class_counts)


# TODO: Concatenate with Barcode ID
# TODO: Update the master image list with some way to link the tissue slices
