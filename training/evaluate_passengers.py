
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

model = YOLO(str(
    ROOT / "runs/transitsense/passengers_baseline/weights/best.pt"
))

data = ROOT / "datasets/processed/passenger_baseline.yaml"

metrics = model.val(
    data=str(data),
    split="val",
    imgsz=640
)

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
