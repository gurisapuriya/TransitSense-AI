"""Build a pixel-level aisle challenge from an existing Inside Bus View test image.

A polygon cut from the red shopping bag at seat 1B is copied onto the aisle.
This is a synthetic positive control for the complete video inference pathway.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make_video(output):
    ref = json.loads((ROOT / "evaluation/cabin_reference.json").read_text())
    source = next(row["source"] for row in ref if row["id"] == "quiet")
    image = cv2.imread(str(ROOT / source))
    if image is None:
        raise ValueError(f"Missing source still: {source}")
    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    outline = np.array([(229,430),(239,400),(251,384),(268,384),(278,390),(316,393),
                        (348,395),(366,410),(369,443),(355,495),(353,523),(339,536),
                        (293,533),(254,516),(237,498)], dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [outline], 255)
    x, y, w, h = cv2.boundingRect(outline)
    patch = image[y:y+h, x:x+w]
    alpha = mask[y:y+h, x:x+w].astype(bool)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        path = Path(folder) / "challenge.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 540))
        if not writer.isOpened():
            raise ValueError("Could not create challenge video")
        try:
            for i in range(80):
                frame = image.copy()
                if 10 <= i < 17 or 30 <= i < 70:
                    target = frame[355:355+h, 417:417+w]
                    target[alpha] = patch[alpha]
                writer.write(frame)
        finally:
            writer.release()
        path.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cabin/bag_challenge.mp4")
    make_video(parser.parse_args().output)
