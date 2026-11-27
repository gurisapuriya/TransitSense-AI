"""Feed a labelled synthetic detection sequence through the same spatial and flow engine."""
import argparse
from pathlib import Path

from inference.doorway import Doorway
from inference.monitor import Measurements, ROOT, write_json
from inference.spatial import Detection


def rectangle(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def fixture():
    seats = {}
    for i, x in enumerate((0.13, 0.32, 0.51, 0.70, 0.84), 1):
        for side, y in (("A", 0.20), ("B", 0.62)):
            seats[f"{i}{side}"] = [x, y, x + 0.06, y + 0.13]
    return {"name": "Controlled stop", "aspect_ratio": 2, "anchor": "center",
            "zones": {k: rectangle(x1, 0.12, x2, 0.83)
                      for k, x1, x2 in (("front", 0.06, 0.37), ("middle", 0.37, 0.65), ("rear", 0.65, 0.94))},
            "zone_capacities": {"front": 4, "middle": 4, "rear": 4},
            "seats": seats, "passages": {"aisle": rectangle(.06,.39,.94,.57),
                                         "door": rectangle(.12,.76,.26,.95)},
            "dwell_s": 2, "seat_confirm_s": .3}


def person(x, y, identity):
    return Detection("person", (x-.025, y-.04, x+.025, y+.04), .91, identity)


def interpolate(t, t1, t2, start, end):
    fraction = min(1, max(0, (t-t1)/(t2-t1)))
    return start + fraction*(end-start)


def detections(t, layout):
    passengers = [person(.87,.27,1)]
    if t >= 1:
        x = interpolate(t,3,5,.19,.535)
        y = interpolate(t,1,3,.96,.49)
        passengers.append(person(x, interpolate(t,5,6,y,.26),2))
    if t >= 5:
        passengers.append(person(interpolate(t,7,9,.19,.33),interpolate(t,5,7,.96,.48),3))
    if t < 16:
        passengers.append(person(interpolate(t,12,14,.73,.19),interpolate(t,14,16,.49,.98),4))
    objects = list(passengers)
    for name, box in layout["seats"].items():
        label = "Occupied" if name == "5A" or (name == "3A" and t >= 6) else "Unoccupied"
        objects.append(Detection(label, tuple(box), .92))
    if 7 <= t < 13:
        objects.append(Detection("suitcase", (.45,.39,.51,.53), .88, 20))
    if 2 <= t < 3:
        objects.append(Detection("backpack", (.16,.74,.21,.88), .77, 21))
    return objects


def simulate():
    layout = fixture()
    door = {"start": [.12,.84], "end": [.26,.84], "inside": [.19,.5],
            "margin_height_fraction": .035, "max_gap_s": .8, "confirm_frames": 2}
    counter = Doorway((.12*1000,.84*500),(.26*1000,.84*500),-1,.035*500,.8,2)
    monitor = Measurements(layout, counter, 1000, 500)
    samples = []
    events = []
    for frame in range(181):
        t = frame / 10
        quality = "low detail" if 16.5 <= t < 17.5 else "usable"
        samples.append(monitor.sample(t, detections(t,layout), quality))
        events.extend(monitor.events)
    return {"schema": 1, "name": "Controlled stop", "source": "Simulated detections",
            "layout": layout, "door": door, "samples": samples, "events": events}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dashboard/replays/controlled.json")
    args = parser.parse_args()
    write_json(args.output, simulate())
