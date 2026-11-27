"""Score alert episodes against the controlled pixel-level aisle placement trial."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def score(path):
    samples = json.loads(path.read_text())["samples"]
    if not samples or samples[0]["t"] != 0 or samples[-1]["t"] < 7.9:
        raise ValueError("Expected the full eight-second synthetic bag trial")
    if any(s["quality"] != "usable" or s["passages"]["aisle"]["status"] == "unobserved"
           for s in samples):
        raise ValueError("Aisle measurements missing during the trial")
    episodes = []
    start = None
    for sample in samples:
        t = sample["t"]
        alert = sample["passages"]["aisle"]["status"] == "review"
        if alert and start is None:
            start = t
        if not alert and start is not None:
            episodes.append((start,t))
            start = None
    if start is not None:
        episodes.append((start,8))
    # Sampling at ~4 Hz permits up to one output interval at either edge.
    valid = [p for p in episodes if 5 <= p[0] < 7 and p[1] <= 7.31]
    matched = int(bool(valid))
    return {"reference_episodes":1,"matched":matched,"missed":1-matched,
            "extra":len(episodes)-matched,"alert_intervals_s":episodes}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",type=Path,default=ROOT/"dashboard/replays/bag.json")
    print(json.dumps(score(parser.parse_args().replay),indent=2))
