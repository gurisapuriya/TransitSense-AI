"""Track passengers and optionally measure flow across a calibrated doorway."""
import argparse
import csv
import math
import subprocess
from collections import deque
from contextlib import ExitStack
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from ultralytics import YOLO

if __package__:
    from .doorway import Doorway, CrossingCounter
else:
    from doorway import Doorway, CrossingCounter

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "runs/transitsense/passengers_baseline/weights/best.pt"
CSV_FIELDS = ("frame", "time_s", "track_id", "confidence", "x1", "y1", "x2", "y2",
              "anchor_x", "anchor_y", "event", "crossing_s")


def staged_file(destination, stack):
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = stack.enter_context(TemporaryDirectory(dir=destination.parent))
    return Path(directory) / destination.name


def draw_track(frame, track_id, bounds, anchor, trail, now):
    x1, y1, x2, y2 = map(round, bounds)
    center = tuple(map(round, anchor))
    color = (180, 235, 90) if track_id % 2 else (245, 165, 90)
    trail.append((now, center))
    for (_, start), (_, end) in pairwise(trail):
        cv2.line(frame, start, end, color, 1, cv2.LINE_AA)
    cv2.circle(frame, center, 2, color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    origin = (max(1, min(x1, frame.shape[1] - 32)), max(12, y1 + 12))
    for ink, thickness in [((20, 20, 20), 3), (color, 1)]:
        cv2.putText(frame, f"#{track_id}", origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, ink, thickness, cv2.LINE_AA)


def run(source, output, csv_path, calibration, weights, confidence_threshold):
    if not 0 < confidence_threshold <= 1:
        raise ValueError("Detection confidence must be in (0, 1]")
    inputs = [source, weights] + ([calibration] if calibration else [])
    outputs = [output] + ([csv_path] if csv_path else [])
    paths = [path.resolve() for path in inputs + outputs]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, video, calibration, weights and CSV paths must be distinct")
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")
    if any(path.is_dir() for path in outputs):
        raise ValueError("Video and CSV destinations must be files")
    if source.suffix.lower() == ".avi":
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "csv=p=0", str(source.resolve())],
            capture_output=True, text=True,
        )
        codec = probe.stdout.strip()
        if probe.returncode or codec in ("", "unknown", "rawvideo"):
            raise ValueError("Convert raw or unreadable AVI to H.264 MP4 before using OpenCV")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Use an .mp4 video output")
    if csv_path and csv_path.suffix.lower() != ".csv":
        raise ValueError("Use a .csv tracking log")

    with ExitStack() as files, ExitStack() as handles:
        capture = cv2.VideoCapture(str(source))
        handles.callback(capture.release)
        ok, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not ok or frame is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot decode the first frame or read its frame rate: {source}")
        height, width = frame.shape[:2]
        door = Doorway.load(calibration, width, height) if calibration else None
        counter = CrossingCounter(door) if door else None
        model = YOLO(str(weights))
        classes = [key for key, name in model.names.items() if name in ("passenger", "person")]
        if len(classes) != 1:
            raise ValueError("Expected one passenger or person class in the supplied detector")

        video_tmp = staged_file(output, files)
        # A separate footer keeps counters away from people and source annotations.
        footer = 44
        out_width = width + width % 2
        out_height = height + footer + height % 2
        writer = cv2.VideoWriter(str(video_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (out_width, out_height))
        handles.callback(writer.release)
        if not writer.isOpened():
            raise ValueError(f"Cannot create video at {output}")
        log = None
        if csv_path:
            csv_tmp = staged_file(csv_path, files)
            log = csv.writer(handles.enter_context(csv_tmp.open("w", newline="")))
            log.writerow(CSV_FIELDS)

        seen = set()
        trails = {}
        frame_no = 0
        expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        while ok:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Frame size changed at frame {frame_no}")
            now = frame_no / fps
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=classes,
                                 conf=confidence_threshold, imgsz=640, verbose=False)[0]
            if door:
                cv2.line(frame, tuple(map(round, door.start)), tuple(map(round, door.end)),
                         (0, 196, 255), 2)
                counter.expire(now)
            trails = {i: t for i, t in trails.items() if now - t[-1][0] <= 0.24}
            visible = 0
            if result.boxes is not None and result.boxes.id is not None:
                for box in result.boxes:
                    track_id = int(box.id.item())
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    anchor = ((x1 + x2) / 2, (y1 + y2) / 2)
                    event = counter.observe(track_id, anchor, now) if counter else None
                    if log:
                        log.writerow((frame_no, round(now, 4), track_id, round(confidence, 4),
                                      *(round(v, 2) for v in (x1, y1, x2, y2, *anchor)),
                                      event[0] if event else "", round(event[1], 4) if event else ""))
                    seen.add(track_id)
                    visible += 1
                    trail = trails.setdefault(track_id, deque(maxlen=7))
                    draw_track(frame, track_id, (x1, y1, x2, y2), anchor, trail, now)

            if log and not visible:
                log.writerow((frame_no, round(now, 4), *([""] * (len(CSV_FIELDS) - 2))))
            canvas = cv2.copyMakeBorder(frame, 0, out_height - height, 0, out_width - width,
                                        cv2.BORDER_CONSTANT, value=(29, 29, 29))
            status = f"{now:.2f}s   Visible {visible}   IDs seen {len(seen)}"
            flow = (f"Boarding {counter.totals['boarding']}    Alighting {counter.totals['alighting']}"
                    if counter else "Tracking only")
            for offset, message in [(16, status), (35, flow)]:
                cv2.putText(canvas, message, (6, height + offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.43, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(canvas)
            frame_no += 1
            ok, frame = capture.read()
        if expected > 0 and frame_no < round(expected):
            raise ValueError(f"Decoded only {frame_no} of {expected:.0f} frames")
        handles.close()
        video_tmp.replace(output)
        if csv_path:
            csv_tmp.replace(csv_path)
    print(f"Frames: {frame_no} | Unique track IDs observed: {len(seen)}")
    if counter:
        print(f"Boarding: {counter.totals['boarding']} | Alighting: {counter.totals['alighting']}")
    print(f"Video: {output}")
    if csv_path:
        print(f"Track log: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Constant-frame-rate input video")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tracked_passengers.mp4")
    parser.add_argument("--debug-csv", type=Path, help="Optional per-frame track and crossing log")
    parser.add_argument("--calibration", type=Path, help="Doorway JSON for this camera view")
    parser.add_argument("--weights", type=Path, default=MODEL_PATH, help="Local passenger or person detector")
    parser.add_argument("--conf", type=float, default=0.35, help="Minimum detection confidence")
    args = parser.parse_args()
    try:
        run(args.source, args.output, args.debug_csv, args.calibration, args.weights, args.conf)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(1, f"Tracking failed: {error}\n")


if __name__ == "__main__":
    main()
