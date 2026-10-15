import csv
import shutil
from pathlib import Path

import numpy as np
import skimage
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm


def copy_images_from_aperio(aperio_csv_path, destination_folder):
    """
    Copy images from the remote folder using information from an Aperio-generated CSV.

    Parameters
    ----------
    aperio_csv_path : str or path-like
        Path to the Aperio-generated CSV file.
    destination_folder : str or path-like
        Path to the output folder.
    """

    aperio_csv_path = Path(aperio_csv_path)

    destination_folder = Path(destination_folder)
    destination_folder.mkdir(exist_ok=True, parents=True)

    with open(aperio_csv_path, mode="r") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            print(row["File Location"])
            shutil.copy(row["File Location"], destination_folder)


def detect_and_crop_tissue(
    input_path, output_path, downsample=16, padding=50, max_link_distance_px=37
):
    """
    Detect tissue in slide and crop images.

    This function identifies tissue slices in the whole slide images, then generates a
    series of images, cropped to just the tissue. To help with tissue slices which may
    be broken or fragmented, the algorithm joins nearby objects together (determined by
    the max_link_distance_px).

    To help conserve memory, the tissue identification is done at the specified
    pyramid level, with an additional 4x downsampling. For example, if the pyramid
    resolution specified was downsampled by 4x, the total downsampling is actually 4 x 4
    = 16x smaller. The smaller resolution is only used for the tissue detection.

    Parameters
    ----------
    input_path : str or path-like
        Path to directory containing slide images
    output_path : str or path-like
        Path to output directory
    ds_level : int, optional
        Pyramid level to read for tissue identification, by default 1
    padding : int, optional
        Number of pixels (full resolution) to pad the detected tissue region, by default 50
    max_link_distance_px : int, optional
        Distance in pixels (downsampled) to link objects, by default 37
    """

    input_path = Path(input_path)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    slide_list = list(input_path.glob("*.svs"))

    # Detect the tissue using one of the lower resolutions

    footprint = skimage.morphology.disk(max_link_distance_px)

    for slide_path in tqdm(slide_list):
        slide = OpenSlide(slide_path)

        slide_width, slide_height = slide.dimensions

        img = (slide.read_region((0, 0), 0, slide.dimensions)).convert("L")

        # Downsample the image for tissue detection
        target_width = slide_width // downsample
        target_height = slide_height // downsample
        mini_img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

        image_array = np.array(mini_img)

        tissue_mask = image_array < 180
        tissue_mask = skimage.morphology.remove_small_objects(
            tissue_mask, max_size=1500
        )

        # Try to join nearby fragments together
        tissue_mask = skimage.morphology.closing(tissue_mask, footprint)

        # Find the individual tissue pieces
        tissue_labels = skimage.measure.label(tissue_mask)
        tissue_data = skimage.measure.regionprops(tissue_labels)

        # Crop and export the images
        base_output_name = slide_path.stem
        for idx, prop in enumerate(tissue_data):
            # bbox = (min_row, min_col, max_row, max_col)
            ymin, xmin, ymax, xmax = prop.bbox

            # Bump the size a little
            x0 = max(0, int(xmin * downsample) - padding)
            y0 = max(0, int(ymin * downsample) - padding)
            x1 = min(int(xmax * downsample) + padding, slide_width)
            y1 = min(int(ymax * downsample) + padding, slide_height)

            # Read in the image region
            cropped = slide.read_region(
                (x0, y0),
                0,
                (x1 - x0, y1 - y0),
            )

            # To fix the white balance, convert back to RGBA, then paste it over a white background
            white_bg = Image.new("RGB", cropped.size, (255, 255, 255))

            white_bg.paste(cropped, (0, 0), cropped)

            white_bg.save(output_path / f"{base_output_name}_region{idx}.png")

        # overlay = skimage.segmentation.mark_boundaries(image_array, tissue_labels)

        # from matplotlib import pyplot as plt

        # plt.imshow(overlay)
        # plt.show()
