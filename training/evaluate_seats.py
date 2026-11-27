from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

weights = (
    ROOT / "runs/transitsense/seats_clean_v1/weights/best.pt"
)

data = ROOT / "datasets/processed/seat_baseline.yaml"

if not weights.exists():
    raise FileNotFoundError("Train the seat model first.")

model = YOLO(str(weights))

metrics = model.val(
    data=str(data),
    split="test",
    imgsz=640
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)