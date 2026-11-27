"""Score unique real cabin stills once each, after the seat confirmation interval."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def score(directory, phase=None):
    refs = json.loads((ROOT / "evaluation/cabin_reference.json").read_text())
    if phase:
        refs = [r for r in refs if r["phase"] == phase]
    matches = comparisons = unknown = false_available = aisle_extras = 0
    errors = {k: [] for k in ("front", "middle", "rear")}
    for ref in refs:
        report = json.loads((directory / (ref["id"] + ".json")).read_text())
        last = report["samples"][-1]
        for zone, expected in ref["visible_people"].items():
            predicted = last["zones"][zone]["people"]
            errors[zone].append(None if predicted is None else abs(predicted-expected))
        for seat, expected in ref["seat_states"].items():
            state = last["seats"][seat]
            matches += state == expected
            comparisons += 1
            unknown += state == "uncertain"
            false_available += state == "available" and expected != "available"
        if not ref["aisle_object_present"]:
            aisle_extras += any(s["passages"]["aisle"]["status"] == "review" for s in report["samples"])
    return {"scenes": len(refs), "correct_seat_states": matches, "seat_positions_reviewed": comparisons,
            "predicted_uncertain": unknown, "false_available": false_available,
            "negative_scenes_with_aisle_alert": aisle_extras, "zone_absolute_errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=ROOT / "dashboard/replays")
    parser.add_argument("--phase", choices=("calibration", "additional"))
    args = parser.parse_args()
    print(json.dumps(score(args.replays, args.phase), indent=2))
