from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    dataset = ROOT / "datasets/processed/seat_occupancy_clean"

    # Load project training settings
    settings = yaml.safe_load(
        (ROOT / "configs/seat_training.yaml").read_text()
    )

    # Read the original class names
    original = yaml.safe_load(
        (dataset / "data.yaml").read_text()
    )

    # Create a local YOLO dataset configuration
    data_config = {
        "path": str(dataset.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": original["names"]
    }

    output = ROOT / "datasets/processed/seat_baseline.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data_config))

    # Select available hardware
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    print("Training device:", device)

    # Load pretrained YOLO
    model = YOLO(settings["model"])

    # Train the seat-occupancy detector
    model.train(
        data=str(output),
        epochs=settings["epochs"],
        imgsz=settings["imgsz"],
        batch=settings["batch"],
        device=device,
        project=str(ROOT / "runs/transitsense"),
        name="seats_clean_v1"
    )

if __name__ == "__main__":
    main()