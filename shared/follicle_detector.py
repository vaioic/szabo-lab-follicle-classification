import json
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from warnings import warn

import pandas as pd
import skimage
import torch
import torchvision
from openslide import OpenSlide
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 2
    momentum: float = 0.9
    weight_decay: float = 0.0005
    num_epochs: int = 15
    num_classes: int = 2
    shuffle: bool = True
    anchor_sizes: tuple = (
        (32,),
        (64,),
        (128,),
        (320,),
        (768,),
    )
    anchor_aspect_ratios: tuple = (0.5, 1.0, 2.0)
    weights: str = "DEFAULT"
    augment_images = False


class FollicleImageDataset(Dataset):
    """
    Loads and formats annotations into a torch-compatible Dataset.

    The input to the function should be the folder containing CSV-files created by
    QuPath. The CSV-files should be named with the following pattern:
    <filename>_bounding_boxes.csv. The function will automatically check that there is a
    corresponding file named <filename>.png in image_folder.

    Note: The Faster-RCNN model requires:
     * images: List of images as a tensor [C, H, W]
     * targets: A list of dicts where each dictionary corresponds to the image in images
       * "boxes": [N, 4]
       * "labels": [N]
       * "image_id": unique integer specifying the image

    Attributes
    ----------
    image_folder : str or Path
        Path to directory containing the images.
    annotation_folder : str or Path
        Path to directory containing the CSV-files with bounding boxes.
    make_classes_binary : bool, optional
        If true, makes all classes = 1, by default False.
    num_classes : int
        Total number of classes (1 for binary, number of labels for multi-class).

    Raises
    ------
    FileNotFoundError
        No CSV-files were found in the annotation_folder.
    FileNotFoundError
        No images matching the annotation files were found in the image_folder.
    """

    def __init__(
        self, image_folder, annotation_folder, make_classes_binary=False, augment=False
    ):

        # Validate the inputs are valid paths
        image_folder = Path(image_folder)

        annotation_folder = Path(annotation_folder)

        list_annotation_files = list(annotation_folder.glob("*.csv"))

        if not list_annotation_files:
            raise FileNotFoundError(
                f"{annotation_folder.stem}: No CSV files were found"
            )

        # Parse the annotations, checking to make sure there is data and that the image
        # exists
        df_list = []
        for f in list_annotation_files:
            df_annotation = pd.read_csv(f)

            if df_annotation.empty:
                warn(f"{f.stem}: No annotations detected. Skipping.")
                continue

            # Check that the image exists and append it to the list
            image_fn = (f.stem).replace("_bounding_boxes", "")

            image_path = image_folder / f"{image_fn}.png"

            if not image_path.exists():
                warn(f"{f.stem}: SVS-file missing. Skipping.")
                continue

            # Checks passed so append annotations and image path to properties
            df_annotation["image_name"] = image_fn
            df_annotation["image_path"] = image_path

            df_list.append(df_annotation)

        # Combine all the validated data as single list
        if not df_list:
            raise FileNotFoundError("No valid annotation or image files were found.")
        else:
            self.df = pd.concat(df_list)

        # Convert classes to an ID
        if not make_classes_binary:
            unique_classes = sorted(self.df["Class"].unique())

            self.class_to_idx = {
                class_name: (idx + 1) for idx, class_name in enumerate(unique_classes)
            }
            self.df["class_id"] = self.df["Class"].map(self.class_to_idx)

        else:
            # This is primarily to train just an ROI proposer
            self.class_to_idx = None
            self.df["class_id"] = 1

        # Get a list of the unique image names
        self.image_paths = list(self.df["image_path"].unique())
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        # Get the annotations for the image
        image_name = Path(image_path).stem

        image_annotations = self.df[self.df["image_name"] == image_name]

        labels = image_annotations["class_id"].values
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        boxes = image_annotations[["XMin", "YMin", "XMax", "YMax"]].values
        boxes = boxes.copy()

        image = skimage.io.imread(image_path).copy()
        height, width = image.shape[:2]

        if self.augment:
            # 1. Random Horizontal Flip (50% chance)
            if torch.rand(1).item() > 0.5:
                image = image[:, ::-1, :]
                x_min = width - boxes[:, 2]
                x_max = width - boxes[:, 0]
                boxes[:, 0] = x_min
                boxes[:, 2] = x_max

            # 2. Random Vertical Flip (50% chance)
            if torch.rand(1).item() > 0.5:
                image = image[::-1, :, :]
                y_min = height - boxes[:, 3]
                y_max = height - boxes[:, 1]
                boxes[:, 1] = y_min
                boxes[:, 3] = y_max

            # 3. Random 90-degree Rotations (0, 90, 180, or 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image = np.rot90(image, k, axes=(0, 1))
                for _ in range(k):
                    # For each 90-degree clockwise rotation, rotate the box coordinates
                    # Width and Height swap during 90-deg rotations
                    old_xmin, old_ymin, old_xmax, old_ymax = (
                        boxes[:, 0],
                        boxes[:, 1],
                        boxes[:, 2],
                        boxes[:, 3],
                    )
                    boxes[:, 0] = height - old_ymax
                    boxes[:, 1] = old_xmin
                    boxes[:, 2] = height - old_ymin
                    boxes[:, 3] = old_xmax
                    # Update height/width reference for subsequent 90-deg turns
                    height, width = width, height

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        # Convert image to tensor (C, H, W)
        img_tensor = torchvision.transforms.functional.to_tensor(image)
        return img_tensor, target

    @property
    def num_classes(self):

        if self.class_to_idx is None:
            nc = 1
        else:
            nc = len(self.class_to_idx)
        return nc


def get_model(config: TrainingConfig, model="fasterrcnn-resnet50"):
    """
    Get torchvision model and update the model head.

    Parameters
    ----------
    num_classes : int
        Number of expected classes
    model : str, optional
        Model name, by default "fasterrcnn-resnet50"

    Returns
    -------
    model : torchvision.model
        The torchvision model.
    """

    match model.lower():
        case "fasterrcnn-resnet50":
            from torchvision.models.detection.rpn import AnchorGenerator

            anchor_generator = AnchorGenerator(
                sizes=config.anchor_sizes,
                aspect_ratios=(config.anchor_aspect_ratios,) * len(config.anchor_sizes),
            )

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights="DEFAULT",
                rpn_anchor_generator=anchor_generator,
            )

            # Update the model head to match the number of classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, config.num_classes
            )
            # model.rpn.head.reset_parameters()

    return model


def train(
    image_folder,
    annotation_folder,
    base_output_dir,
    validation_annotation_folder=None,
    test_annotation_folder=None,
    model_name=None,
    save_every_n_epochs=5,
    checkpoint_path=None,
    **kwargs,
):

    # Create the output directory
    base_output_dir = Path(base_output_dir)

    # Create the base log sub-directory
    base_log_dir = base_output_dir / "runs"

    # Create the checkpoints directory - this should be unique
    strDateTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name is None:
        output_dir = base_output_dir / "checkpoints" / strDateTime
    else:
        output_dir = base_output_dir / "checkpoints" / (strDateTime + model_name)
    output_dir.mkdir(parents=True, exist_ok=False)  # Throw error if it already exists

    # Create the log directory with the same name
    if model_name is None:
        log_dir = base_log_dir / strDateTime
    else:
        log_dir = base_log_dir / (strDateTime + model_name)
    log_dir.mkdir(parents=True, exist_ok=False)

    # Update the hyperparameters dataclass
    config = TrainingConfig()
    valid_fields = {f.name for f in fields(TrainingConfig)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
    config = replace(config, **filtered_kwargs)

    # Save the hyperparameters as a JSON-file
    json_path = output_dir / "config.json"
    with open(json_path, "w") as f:
        json.dump(asdict(config), f, indent=4)
    print(f"Configuration saved to JSON: {json_path}")

    # Initialize the TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # Load the data
    if config.num_classes == 2:
        dataset = FollicleImageDataset(
            image_folder,
            annotation_folder,
            make_classes_binary=True,
            augment=config.augment_images,
        )
    else:
        dataset = FollicleImageDataset(
            image_folder, annotation_folder, augment=config.augment_images
        )

    if validation_annotation_folder is not None:
        if config.num_classes == 2:
            val_dataset = FollicleImageDataset(
                image_folder, validation_annotation_folder, make_classes_binary=True
            )
        else:
            val_dataset = FollicleImageDataset(
                image_folder, validation_annotation_folder
            )

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    # Save class mapping
    if dataset.class_to_idx is not None:
        # Invert {"Name": id} to {id: "Name"} for prediction lookup
        idx_to_class = {str(idx): name for name, idx in dataset.class_to_idx.items()}
    else:
        # Binary case fallback
        idx_to_class = {"1": "Follicle"}

    class_map_path = output_dir / "class_mapping.json"
    with open(class_map_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Class mapping saved to {class_map_path}")

    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Check if GPU is available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = get_model(config)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

        print(f"Loading weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.to(device)

    # Optimizer configuration
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    # Training loop
    model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        training_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] | Loss: {training_loss:.4f}")

        # Log metrics to tensor board
        writer.add_scalar("Loss/Train", training_loss, epoch + 1)

        if validation_annotation_folder is not None:
            val_loss = 0
            with torch.no_grad():
                for images, targets in val_data_loader:
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    val_loss_dict = model(images, targets)
                    val_losses = sum(loss for loss in val_loss_dict.values())
                    val_loss += val_losses.item()

            validation_loss = (
                val_loss / len(val_data_loader) if len(val_data_loader) > 0 else 0.0
            )

            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}] | Train Loss: {training_loss:.4f} | Val Loss: {validation_loss:.4f}"
            )

            writer.add_scalar("Loss/Validation", validation_loss, epoch + 1)

        # Save regular checkpoints
        if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == config.num_epochs:
            ckpt_path = output_dir / f"follicle_detector_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"-> Checkpoint saved: {ckpt_path}")

            log_validation_predictions(
                model=model,
                data_loader=val_data_loader,
                writer=writer,
                epoch=epoch + 1,
                device=device,
                confidence_thresh=0.5,
            )

    # Write hyperparameters after a successful run
    final_metrics = {
        "hparam/train_loss": training_loss,
    }
    if validation_annotation_folder is not None:
        final_metrics["hparam/val_loss"] = validation_loss

    hparams_dict = asdict(config)
    hparams_dict["anchor_sizes"] = str(config.anchor_sizes)
    hparams_dict["anchor_aspect_ratios"] = str(config.anchor_aspect_ratios)

    writer.add_hparams(hparams_dict, final_metrics, run_name=".")
    writer.flush()
    writer.close()

    # Final model path
    final_path = output_dir / "follicle_detector_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final weights saved to {final_path}")
    return final_path


def load_config_from_json(config_path):

    config_path = Path(config_path)

    with open(config_path, "r") as f:
        data = json.load(f)

    # Safety check in case we change the model parameters down the line
    valid_fields = {f.name for f in fields(TrainingConfig)}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    if "anchor_sizes" in filtered_data:
        filtered_data["anchor_sizes"] = tuple(
            tuple(item) for item in filtered_data["anchor_sizes"]
        )
    if "anchor_aspect_ratios" in filtered_data:
        filtered_data["anchor_aspect_ratios"] = tuple(
            filtered_data["anchor_aspect_ratios"]
        )

    return TrainingConfig(**filtered_data)


def predict_follicles(model_path, image_paths, output_dir, confidence_thresh=0.5):

    model_path = Path(model_path)

    image_paths = [Path(p) for p in image_paths]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Look for class_mapping.json in the same directory as the model weights
    class_map_path = model_path.parent / "class_mapping.json"
    if class_map_path.exists():
        with open(class_map_path, "r") as f:
            idx_to_class = json.load(f)
    else:
        idx_to_class = {}
        warn("class_mapping.json not found. Falling back to numeric class IDs.")

    # Load the model to get the number of classes
    checkpoint = torch.load(model_path, map_location=device)

    # Load the model configuration
    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    config = load_config_from_json(config_path)

    # Build the model
    model = get_model(config)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    results = {}
    with torch.no_grad():
        for img_path in image_paths:
            img = skimage.io.imread(img_path)
            img_tensor = (
                torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
            )

            predictions = model(img_tensor)[0]
            scores = predictions["scores"].cpu().numpy()
            keep = scores >= confidence_thresh

            boxes = predictions["boxes"].cpu().numpy()[keep]
            labels = predictions["labels"].cpu().numpy()[keep]
            filtered_scores = scores[keep]

            results[img_path] = {
                "boxes": boxes,
                "labels": labels,
                "scores": filtered_scores,
            }

            all_features = []
            for box, label, score in zip(boxes, labels, filtered_scores):
                xmin, ymin, xmax, ymax = box
                class_name = idx_to_class.get(str(label), f"Class_{label}")

                polygon_coords = [
                    [float(xmin), float(ymin)],
                    [float(xmax), float(ymin)],
                    [float(xmax), float(ymax)],
                    [float(xmin), float(ymax)],
                    [float(xmin), float(ymin)],
                ]

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon_coords],
                    },
                    "properties": {
                        "object_type": "annotation",
                        "classification": class_name,
                        "confidence": float(score),
                    },
                }
                all_features.append(feature)

            # Construct the final GeoJSON FeatureCollection
            geojson_collection = {"type": "FeatureCollection", "features": all_features}

            output_fn = img_path.stem
            output_json_path = output_dir / f"{output_fn}.json"

            with open(output_json_path, "w") as f:
                json.dump(geojson_collection, f, indent=2)

            print(f"GeoJSON successfully exported to {output_json_path}")

    return results


def read_slide_image(image_path):
    # Open the slide
    slide = OpenSlide(image_path)

    img = slide.read_region(location=(0, 0), level=0, size=(slide.dimensions))
    img = img.convert("RGB")

    return img


def log_validation_predictions(
    model, data_loader, writer, epoch, device, confidence_thresh=0.5, max_images=3
):
    """
    Runs model evaluation on a few validation/training samples, draws predicted bounding boxes,
    and logs them to TensorBoard's Images tab.
    """
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            # Run inference
            predictions = model(images)

            logged_count = 0
            for img_tensor, pred in zip(images, predictions):
                if logged_count >= max_images:
                    break

                scores = pred["scores"]
                keep = scores >= confidence_thresh
                pred_boxes = pred["boxes"][keep].cpu()

                # TensorBoard expects images in uint8 format [0, 255] (C, H, W)
                img_uint8 = (img_tensor.cpu() * 255).to(torch.uint8)

                if len(pred_boxes) > 0:
                    drawn_img = torchvision.utils.draw_bounding_boxes(
                        img_uint8, pred_boxes, colors="red", width=2
                    )
                else:
                    drawn_img = img_uint8

                # Send to TensorBoard Images tab
                writer.add_image(
                    f"Val_Predictions/image_{logged_count}", drawn_img, epoch
                )
                logged_count += 1

            break  # Only process the first batch
    model.train()  # Switch back to training mode
