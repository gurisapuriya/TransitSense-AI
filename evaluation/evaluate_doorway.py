"""Compare one clip's tracking log with reviewed, approximate crossing times."""
import argparse
import csv
import json
import math
from pathlib import Path

DIRECTIONS = ("boarding", "alighting")


def match_events(reference, predicted, tolerance):
    """Match in time order within one direction, using each event at most once."""
    reference, predicted = sorted(reference), sorted(predicted)
    matched, missed, extras = [], [], []
    i = j = 0
    while i < len(reference) and j < len(predicted):
        delta = predicted[j] - reference[i]
        if delta < -tolerance:
            extras.append(predicted[j])
            j += 1
        elif delta > tolerance:
            missed.append(reference[i])
            i += 1
        else:
            matched.append((reference[i], predicted[j]))
            i += 1
            j += 1
    return matched, missed + reference[i:], extras + predicted[j:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, help="Clip ID in the reference CSV")
    predictions_input = parser.add_mutually_exclusive_group(required=True)
    predictions_input.add_argument("--tracks", type=Path, help="Per-frame tracking debug CSV")
    predictions_input.add_argument("--report", type=Path, help="Aggregate monitor JSON with crossing times")
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("doorway_reference.csv"))
    parser.add_argument("--tolerance", type=float, default=0.5, help="Time tolerance in seconds")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")
    with args.reference.open(newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["clip"] == args.clip]
    if not rows:
        parser.error(f"no reference entries for {args.clip}")
    if args.report:
        report = json.loads(args.report.read_text())
        if report.get("door") is None or "events" not in report:
            parser.error("Run a calibrated monitor trial with event logging")
        predictions = [{"event": event["direction"], "crossing_s": event["time_s"]} for event in report["events"]]
    else:
        with args.tracks.open(newline="") as source:
            predictions = [row for row in csv.DictReader(source) if row["event"]]
    for direction in DIRECTIONS:
        ref = [float(r["time_s"]) for r in rows if r["direction"] == direction]
        pred = [float(r["crossing_s"]) for r in predictions if r["event"] == direction]
        hits, misses, extras = match_events(ref, pred, args.tolerance)
        print(f"{direction}: reference={len(ref)} predicted={len(pred)} "
              f"absolute_error={abs(len(ref) - len(pred))} matched={len(hits)} "
              f"missed={len(misses)} extra={len(extras)}")
        print(f"  matched (ref, pred): {hits} | missed times: {misses} | extra times: {extras}")


if __name__ == "__main__":
    main()
