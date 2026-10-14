from shared.follicle_detector import FollicleImageDataset


def test_FollicleImageDataset():

    image_paths = [
        "../data/Dataset01/raw/272940.svs",
        "../data/Dataset01/raw/272941.svs",
    ]

    csv_paths = [
        "../data/Dataset01/qupath/export/272940_bounding_boxes.csv",
        "../data/Dataset01/qupath/export/272941_bounding_boxes.csv",
    ]

    ds = FollicleImageDataset(image_paths, csv_paths)

    assert len(ds) == 2


def test_FollicleImageDataset_getitem():

    image_paths = [
        "../data/Dataset01/raw/272940.svs",
        "../data/Dataset01/raw/272941.svs",
    ]

    csv_paths = [
        "../data/Dataset01/qupath/export/272940_bounding_boxes.csv",
        "../data/Dataset01/qupath/export/272941_bounding_boxes.csv",
    ]

    ds = FollicleImageDataset(image_paths, csv_paths)

    X = ds[0]

    assert len(ds) == 2
