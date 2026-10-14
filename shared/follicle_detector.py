import json
from pathlib import Path

import pandas as pd
import torch
import torchvision
from openslide import OpenSlide
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FollicleImageDataset(Dataset):
    """
    Note: The Faster-RCNN model requires:
     * images: List of images as a tensor [C, H, W]
     * targets: A list of dicts where each dictionary corresponds to the image in images
       * "boxes": [N, 4]
       * "labels": [N]
       * "image_id": unique integer specifying the image
    """

    def __init__(self, image_paths, annotation_paths):

        # Validate the inputs
        if len(image_paths) != len(annotation_paths):
            raise ValueError(
                "The number of images does not match the number of annotation files."
            )

        self.image_paths = [Path(p) for p in image_paths]
        self.image_names = [s.stem for s in self.image_paths]

        # Read in the CSV-files
        annotation_paths = [Path(p) for p in annotation_paths]
        df_list = []
        for idx, f in enumerate(annotation_paths):
            curr_df = pd.read_csv(f)
            curr_df["image_name"] = self.image_names[idx]
            df_list.append(curr_df)

        self.df = pd.concat(df_list)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        img_name = self.image_names[idx]

        img = read_slide_image(img_path)

        # Filter annotations for the current image
        image_annotations = self.df[self.df["image_name"] == img_name]
        boxes = image_annotations[["XMin", "YMin", "XMax", "YMax"]].values

        raw_classes = image_annotations["Class"].values
        int_classes = [2 if c == "No nucleus" else 1 for c in raw_classes]

        # Convert to tensors
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(int_classes, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        # Convert image to tensor (C, H, W)
        img_tensor = torchvision.transforms.functional.to_tensor(img)
        return img_tensor, target


def get_model(num_classes, model="fasterrcnn-resnet50"):
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
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights="DEFAULT"
            )

            # Update the model head to match the number of classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_follicle_detector(
    image_paths,
    annotation_paths,
    output_dir,
    num_epochs=15,
    batch_size=2,
    learning_rate=0.001,
    save_every_n_epochs=5,
    num_classes=2,
):

    # Create the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the log directory
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # Load the data
    dataset = FollicleImageDataset(image_paths, annotation_paths)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Check if GPU is available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = get_model(num_classes)
    model.to(device)

    # Optimizer configuration
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=0.9, weight_decay=0.0005
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_epoch_loss:.4f}")

        # --- Log metrics to TensorBoard ---
        writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch + 1)
        writer.add_scalar(
            "Parameters/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1
        )

        # Save regular checkpoints
        if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == num_epochs:
            ckpt_path = output_dir / f"follicle_detector_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"-> Checkpoint saved: {ckpt_path}")

    # Always close the writer when training completes
    writer.close()

    # Final model path
    final_path = output_dir / "follicle_detector_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final weights saved to {final_path}")
    return final_path


def predict_follicles(model_path, image_paths, output_dir, confidence_thresh=0.5):

    model_path = Path(model_path)

    image_paths = [Path(p) for p in image_paths]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model to get the number of classes
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0]

    # Build the model
    model = get_model(num_classes)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    results = {}
    with torch.no_grad():
        for img_path in image_paths:
            img = read_slide_image(img_path)
            img_tensor = (
                torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
            )

            predictions = model(img_tensor)[0]
            scores = predictions["scores"].cpu().numpy()
            keep = scores >= confidence_thresh

            results[img_path] = {
                "boxes": predictions["boxes"].cpu().numpy()[keep],
                "labels": predictions["labels"].cpu().numpy()[keep],
                "scores": scores[keep],
            }

            all_features = []
            for img_path, data in results.items():
                for box, label, score in zip(
                    data["boxes"], data["labels"], data["scores"]
                ):
                    xmin, ymin, xmax, ymax = box

                    # QuPath expects closed polygon coordinates [x, y] forming a rectangle
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
                            "object_type": "detection",
                            "classification": f"Class_{label}",
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
