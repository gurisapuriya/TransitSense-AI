
from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

passenger_model = YOLO(str(
    ROOT / "runs/transitsense/passengers_baseline/weights/best.pt"
))

seat_model = YOLO(str(
    ROOT / "runs/transitsense/seats_clean_v1/weights/best.pt"
))

def detect(frame):
    counts = {"passenger": 0, "Occupied": 0, "Unoccupied": 0}

    for model in [passenger_model, seat_model]:
        result = model.predict(
            frame, conf=0.4, imgsz=640, verbose=False
        )[0]

        for box in result.boxes:
            cls = int(box.cls.item())
            label = result.names[cls]
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            color = (0, 255, 0) if label == "Unoccupied" else (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {confidence:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
            counts[label] = counts.get(label, 0) + 1

    text = (
        f"Passengers: {counts['passenger']} | "
        f"Occupied seats: {counts['Occupied']} | "
        f"Vacant seats: {counts['Unoccupied']}"
    )

    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 2
    )

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + args.source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    output = ROOT / "results/combined_demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Cannot create output video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = detect(frame)
        writer.write(annotated)

    cap.release()
    writer.release()

    print("Saved:", output)


if __name__ == "__main__":
    main()
