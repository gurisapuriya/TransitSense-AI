
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

model_path = (
    ROOT / "runs/transitsense/passengers_baseline/weights/best.pt"
)

model = YOLO(str(model_path))

test_folder = (
    ROOT / "datasets/raw/bus_occupancy/valid/images"
)

model.predict(
    source=str(test_folder),
    conf=0.35,
    imgsz=640,
    save=True,
    project=str(ROOT / "runs/transitsense"),
    name="passenger_predictions"
)

print("Passenger predictions completed.")
