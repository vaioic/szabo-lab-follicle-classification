from pathlib import Path

import pytest
import torch

from shared.follicle_detector import FollicleImageDataset

ANNOTATION_FOLDER = Path("../data/Dataset01/qupath_cropped/export")
IMAGE_FOLDER = Path("../data/Dataset01/cropped")


@pytest.fixture(scope="module")
def dataset():
    """Initializes the standard dataset once per test module."""
    return FollicleImageDataset(IMAGE_FOLDER, ANNOTATION_FOLDER)


@pytest.fixture(scope="module")
def dataset_binary():
    """Initializes the binary-class dataset once per test module."""
    return FollicleImageDataset(
        IMAGE_FOLDER, ANNOTATION_FOLDER, make_classes_binary=True
    )


def test_FollicleImageDataset(dataset):

    # Get list of CSV-files in the test folder
    files = list(ANNOTATION_FOLDER.glob("*.csv"))

    # Check that the results are as expected
    assert len(dataset) == len(files)
    assert dataset.num_classes == 8


def test_FollicleImageDataset_binary_classes(dataset_binary):

    # Get list of CSV-files in the test folder
    files = list(ANNOTATION_FOLDER.glob("*.csv"))

    # Check that the results are as expected
    assert len(dataset_binary) == len(files)
    assert dataset_binary.num_classes == 1


def test_FollicleImageDataset_getitem(dataset):

    image, target = dataset[0]

    assert torch.is_tensor(target["boxes"])
    assert torch.is_tensor(target["labels"])
    assert torch.is_tensor(target["image_id"])

    assert image.ndim == 3
    assert image.shape[0] == 3

    assert image.dtype == torch.float32
