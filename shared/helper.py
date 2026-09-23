import csv
from pathlib import Path
import shutil


def copy_images_from_aperio(aperio_csv_path, destination_folder):

    aperio_csv_path = Path(aperio_csv_path)

    destination_folder = Path(destination_folder)
    destination_folder.mkdir(exist_ok=True, parents=True)

    with open(aperio_csv_path, mode="r") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            print(row["File Location"])
            shutil.copy(row["File Location"], destination_folder)
