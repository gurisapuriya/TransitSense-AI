
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

model_path = (
    ROOT / "runs/transitsense/seats_clean_v1/weights/best.pt"
)

if not model_path.exists():
    raise FileNotFoundError(
        "Cleaned seat model not found. Check training output."
    )

model = YOLO(str(model_path))

test_folder = (
    ROOT / "datasets/processed/"
    "seat_occupancy_clean/test/images"
)

model.predict(
    source=str(test_folder),
    conf=0.35,
    imgsz=640,
    save=True,
    project=str(ROOT / "runs/transitsense"),
    name="seat_predictions"
)

print("Seat predictions generated successfully.")
