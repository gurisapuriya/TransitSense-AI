"""Process one calibrated camera or recording and publish anonymous operator measurements."""
import argparse
from contextlib import closing
from collections import deque
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import cv2
import numpy as np
import psutil

from .doorway import CrossingCounter, Doorway
from .spatial import BAGS, Cabin, Detection, SEAT_CLASSES

ROOT = Path(__file__).resolve().parents[1]
SEAT_MODEL = ROOT / "runs/transitsense/seats_clean_v1/weights/best.pt"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(data, stream, allow_nan=False, separators=(",", ":"))
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def file_frames(source):
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise ValueError("Supply a local constant-frame-rate MP4; convert raw AVI with FFmpeg first")
    cap = cv2.VideoCapture(str(source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video stream or frame rate is unreadable")
        count = 0
        expected = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield count / fps, frame
            count += 1
        if count == 0 or (expected > 0 and count < round(expected)):
            raise ValueError(f"Incomplete video stream at frame {count}")
    finally:
        cap.release()


def image_quality(frame):
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
    if np.percentile(gray, 90) < 25:
        return "too dark"
    if np.percentile(gray, 10) > 239:
        return "overexposed"
    if gray.std() < 5 or cv2.Laplacian(gray, cv2.CV_32F).var() < 4:
        return "low detail"
    return "usable"


class Vision:
    def __init__(self, weights, seat_weights, layout, device, luggage_weights):
        from ultralytics import YOLO
        self.device = device
        self.people = YOLO(str(weights))
        labels = set(self.people.names.values())
        if not labels.intersection({"person", "passenger"}):
            raise ValueError("Detector must include person or passenger labels")
        self.luggage = None
        uses_bags = any(layout["passages"].values()) or bool(layout["seats"])
        if uses_bags and not labels.intersection(BAGS):
            if not luggage_weights.is_file():
                raise ValueError(f"Missing luggage detector: {luggage_weights}")
            self.luggage = YOLO(str(luggage_weights))
            self.bag_classes = [i for i, name in self.luggage.names.items() if name in BAGS]
            if not self.bag_classes:
                raise ValueError("Luggage detector needs a recognized bag class")
        self.classes = [i for i, name in self.people.names.items() if name in BAGS | {"person", "passenger"}]
        self.seats = YOLO(str(seat_weights)) if layout["seats"] else None
        if self.seats and not set(SEAT_CLASSES).issubset(self.seats.names.values()):
            raise ValueError("Seat weights must include Occupied and Unoccupied labels")

    @staticmethod
    def boxes(result):
        boxes = result.boxes
        if boxes is None:
            return []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        return [Detection(result.names[int(cls)], tuple(box), float(conf), identity)
                for cls, box, conf, identity in zip(boxes.cls.tolist(), boxes.xyxyn.tolist(),
                                                     boxes.conf.tolist(), ids)]

    def detect(self, frame):
        result = self.people.track(frame, classes=self.classes, persist=True, tracker="bytetrack.yaml",
                                   conf=0.10, imgsz=640, device=self.device, verbose=False)[0]
        detections = self.boxes(result)
        if self.luggage:
            result = self.luggage.track(frame, classes=self.bag_classes, persist=True, tracker="bytetrack.yaml",
                                        conf=0.10, imgsz=640, device=self.device, verbose=False)[0]
            detections.extend(self.boxes(result))
        if self.seats:
            result = self.seats.predict(frame, conf=0.35, imgsz=640, device=self.device, verbose=False)[0]
            detections.extend(self.boxes(result))
        return detections


class Measurements:
    def __init__(self, layout, door, width, height):
        self.events = []
        self.width, self.height = width, height
        self.cabin = Cabin(layout, width, height)
        self.counter = CrossingCounter(door) if door else None

    def sample(self, now, objects, quality):
        self.events.clear()
        people = [o for o in objects if o.label in ("person", "passenger")]
        flow = None
        if self.counter:
            self.counter.expire(now)
            if quality != "usable":
                self.counter.tracks.clear()
            else:
                for obj in people:
                    if obj.track_id is not None:
                        x, y = obj.anchor("center")
                        event = self.counter.observe(obj.track_id, (x*self.width, y*self.height), now)
                        if event:
                            self.events.append({"direction": event[0], "time_s": round(event[1], 4),
                                                "confirmed_s": round(now, 4)})
            flow = self.counter.totals.copy()
        return {"t": round(now, 3), "quality": quality, "flow": flow,
                "visible": sum(o.confidence >= 0.35 for o in people) if quality == "usable" else None,
                **self.cabin.update(now, objects, quality)}


def monitor(args):
    layout = json.loads(args.layout.read_text())
    if args.oak and args.report and args.seconds is None:
        raise ValueError("A recorded OAK replay needs a finite --seconds duration")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError("Duration must be positive")
    inputs = [p.resolve() for p in (args.layout, args.door, args.source, args.weights, args.seat_weights, args.luggage_weights) if p]
    outs = [p.resolve() for p in (args.state, args.report) if p]
    if len(outs) != len(set(outs)) or any(o in inputs or o.is_dir() or o.suffix != ".json" for o in outs):
        raise ValueError("Use distinct JSON output files separate from inputs")
    for path in [args.weights] + ([args.seat_weights] if layout["seats"] else []):
        if not path.is_file():
            raise ValueError(f"Missing model: {path}")
    if args.oak:
        from .oak_input import frames_oak
        frames = frames_oak()
    else:
        frames = file_frames(args.source)
    info = {"schema": 1, "name": args.name or layout["name"], "source": "OAK camera" if args.oak else "Recorded video",
            "layout": layout, "door": json.loads(args.door.read_text()) if args.door else None}
    if args.gap is not None:
        if not info["door"] or not math.isfinite(args.gap) or args.gap <= 0:
            raise ValueError("--gap needs a doorway calibration and a positive finite duration")
        info["door"]["max_gap_s"] = args.gap
    samples = []
    events = []
    # Limit runtime measurement storage even for an indefinitely running camera.
    timings = deque(maxlen=10000)
    process = psutil.Process()
    peak_rss = 0
    count = 0
    measure = None
    latest = None
    next_publish = 0
    started = None

    def publish(status):
        if args.state:
            write_json(args.state, {**info, "status": status, "updated": time.time(), "sample": latest})

    publish("starting")
    try:
        vision = Vision(args.weights, args.seat_weights, layout, args.device, args.luggage_weights)
        with closing(frames):
            for now, frame in frames:
                if args.seconds and now >= args.seconds:
                    break
                if measure is None:
                    h, w = frame.shape[:2]
                    door = Doorway.load(args.door, w, h) if args.door else None
                    if door and args.gap is not None:
                        door = replace(door, max_gap_s=args.gap)
                    measure = Measurements(layout, door, w, h)
                    started = time.perf_counter()
                if frame.shape[:2] != (measure.height, measure.width):
                    raise ValueError("Frame dimensions changed; recalibrate the camera")
                before = time.perf_counter()
                quality = image_quality(frame)
                # Track every frame, including unusable frames, to advance ByteTrack's clock.
                objects = vision.detect(frame)
                latest = measure.sample(now, objects, quality)
                if args.report:
                    events.extend(measure.events)
                timings.append((time.perf_counter() - before) * 1000)
                peak_rss = max(peak_rss, process.memory_info().rss)
                count += 1
                if now + 1e-6 >= next_publish:
                    if args.state:
                        latest["performance"] = {"processing_fps": round(count/(time.perf_counter()-started), 2),
                                                 "frames": count, "p95_ms": round(float(np.percentile(timings,95)), 2),
                                                 "peak_rss_mib": round(peak_rss/2**20,1)}
                    if args.report:
                        samples.append(latest)
                    publish("running")
                    next_publish = now + 0.25
        if not count:
            raise ValueError("No frames were processed")
        seconds = time.perf_counter() - started
        stats = {"frames": count, "processing_fps": round(count/seconds, 2),
                 "median_ms": round(float(np.median(timings)), 2),
                 "p95_ms": round(float(np.percentile(timings, 95)), 2),
                 "peak_rss_mib": round(peak_rss / 2**20, 1), "device": args.device}
        if args.report:
            if not samples or samples[-1]["t"] != latest["t"]:
                samples.append(latest)
            write_json(args.report, {**info, "samples": samples, "events": events, "benchmark": stats})
        info["benchmark"] = stats
        publish("ended")
        return stats
    except BaseException:
        publish("interrupted")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="Constant-frame-rate MP4")
    source.add_argument("--oak", action="store_true")
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--door", type=Path, help="Optional calibrated door-line JSON")
    p.add_argument("--name")
    p.add_argument("--gap", type=float, help="Override the doorway track-memory duration in seconds")
    p.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt")
    p.add_argument("--luggage-weights", type=Path, default=ROOT / "yolo11n.pt")
    p.add_argument("--seat-weights", type=Path, default=SEAT_MODEL)
    p.add_argument("--state", type=Path, default=ROOT / "results/monitor/live.json")
    p.add_argument("--report", type=Path, help="Save an aggregate replay")
    p.add_argument("--seconds", type=float)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    try:
        print(json.dumps(monitor(args), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        p.exit(1, f"Monitor failed: {error}\n")


if __name__ == "__main__":
    main()
