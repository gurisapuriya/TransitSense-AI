
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/raw/bus_occupancy"

# Load dataset configuration
original = yaml.safe_load(
    (DATASET / "data.yaml").read_text()
)

# Create configuration with absolute paths
config = {
    "path": str(DATASET.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "names": original["names"]
}

config_path = ROOT / "datasets/processed/passenger_baseline.yaml"
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(
    yaml.safe_dump(config, sort_keys=False)
)

# Select training device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = 0
else:
    device = "cpu"

print("Training device:", device)

model = YOLO("yolo11n.pt")

model.train(
    data=str(config_path),
    epochs=30,
    imgsz=640,
    batch=8,
    device=device,
    project=str(ROOT / "runs/transitsense"),
    name="passengers_baseline"
)
