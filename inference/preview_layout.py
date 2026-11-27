"""Render camera calibration over one local frame, before recording measurements."""
import argparse
from contextlib import closing
import json
from pathlib import Path

import cv2
import numpy as np

from .doorway import Doorway
from .monitor import file_frames
from .spatial import Cabin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="A still image or calibrated-view MP4")
    parser.add_argument("--layout", required=True, type=Path)
    parser.add_argument("--door", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Local review PNG")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png" or args.output.resolve() in {
            p.resolve() for p in (args.source,args.layout,args.door) if p}:
        parser.error("Choose a separate PNG output")
    if args.source.suffix.lower() == ".mp4":
        with closing(file_frames(args.source)) as frames:
            _, frame = next(frames)
    else:
        frame = cv2.imread(str(args.source))
    if frame is None:
        parser.error("Cannot read the source image")
    h,w = frame.shape[:2]
    cabin = Cabin(json.loads(args.layout.read_text()),w,h)
    def point(p):
        return tuple(np.rint(np.array(p) * (w-1,h-1)).astype(int))
    def label(text, pos):
        pos = (max(pos[0],2), max(pos[1],16))
        for color, thickness in [((12,12,12),3),((245,245,245),1)]:
            cv2.putText(frame,text,pos,cv2.FONT_HERSHEY_SIMPLEX,.5,color,thickness,cv2.LINE_AA)
    for group, color in [(cabin.zones,(30,215,170)), (cabin.passages,(35,130,255))]:
        for name, region in group.items():
            if region is not None:
                poly = np.array([point(p) for p in region], dtype=np.int32)
                cv2.polylines(frame,[poly],True,color,2)
                label(name,point(region.mean(axis=0)))
    for name, box in cabin.seats.items():
        cv2.rectangle(frame,point(box[:2]),point(box[2:]),(230,190,70),2)
        label(name,point(box[:2]))
    if args.door:
        door=Doorway.load(args.door,w,h)
        cv2.line(frame,tuple(map(round,door.start)),tuple(map(round,door.end)),(0,140,255),3)
        label("inside",point(json.loads(args.door.read_text())["inside"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        parser.error("Could not save the preview")


if __name__ == "__main__":
    main()
