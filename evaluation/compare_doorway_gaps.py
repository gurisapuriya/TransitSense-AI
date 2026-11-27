"""Replay identical PCDS track observations with two prespecified memory durations.

Run from the repository root with python -m evaluation.compare_doorway_gaps.
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from itertools import groupby

import cv2

from evaluation.evaluate_doorway import DIRECTIONS, match_events
from inference.doorway import Doorway, CrossingCounter


ROOT = Path(__file__).resolve().parents[1]


def replay(config, video, tracks, gaps):
    """Use the original frame clock and check the log covers the entire recording."""
    capture = cv2.VideoCapture(str(video))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if min(width, height, fps, count) <= 0:
        raise ValueError(f"Converted video missing or unreadable: {video}")

    door = Doorway.load(config, width, height)
    counters = [CrossingCounter(replace(door, max_gap_s=gap)) for gap in gaps]
    events = [{direction: [] for direction in DIRECTIONS} for _ in gaps]
    frames = 0
    with tracks.open(newline="") as source:
        for frame, rows in groupby(csv.DictReader(source), lambda row: int(row["frame"])):
            if frame != frames:
                raise ValueError(f"Expected frame {frames} in {tracks}, found {frame}")
            now = frame / fps
            for counter in counters:
                counter.expire(now)
            for row in rows:
                if abs(float(row["time_s"]) - now) > 0.0001:
                    raise ValueError(f"Frame clock in {tracks} disagrees with {video}")
                if not row["track_id"]:
                    continue
                anchor = float(row["anchor_x"]), float(row["anchor_y"])
                for counter, results in zip(counters, events):
                    event = counter.observe(int(row["track_id"]), anchor, now)
                    if event:
                        direction, when = event
                        results[direction].append(when)
            frames += 1
    if frames != count:
        raise ValueError(f"Expected {count} frames, found {frames} in {tracks}")
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-dir", type=Path, default=ROOT / "results/doorway")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "results/converted_videos")
    args = parser.parse_args()
    evaluation = Path(__file__).resolve().parent
    with (evaluation / "doorway_fresh_reference.csv").open(newline="") as source:
        refs = list(csv.DictReader(source))
    with (evaluation / "doorway_quiet_intervals.csv").open(newline="") as source:
        quiet = list(csv.DictReader(source))

    gaps = (0.24, 0.80)
    print("clip,gap_s,direction,reference,predicted,matched,missed,extra,quiet_events,times_s")
    for clip in sorted({row["clip"] for row in refs + quiet}):
        events = replay(ROOT / f"configs/pcds_{clip}_door.json",
                        args.videos_dir / f"{clip}.mp4",
                        args.tracks_dir / f"{clip}_person_low.csv", gaps)
        for gap, predictions in zip(gaps, events):
            for direction, times in predictions.items():
                reference = [float(row["time_s"]) for row in refs
                             if row["clip"] == clip and row["direction"] == direction]
                hits, misses, extras = match_events(reference, times, 0.5)
                quiet_events = sum(any(float(row["start_s"]) <= t < float(row["end_s"])
                                       for row in quiet if row["clip"] == clip) for t in times)
                print(f"{clip},{gap:.2f},{direction},{len(reference)},{len(times)},{len(hits)},"
                      f"{len(misses)},{len(extras)},{quiet_events},"
                      + " ".join(f"{time:.3f}" for time in times))


if __name__ == "__main__":
    main()
