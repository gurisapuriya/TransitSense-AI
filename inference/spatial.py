"""Calibrated, single-view cabin measurements. Geometry uses normalized camera coordinates."""
from dataclasses import dataclass
from math import dist, isfinite

import cv2
import numpy as np

ZONES = ("front", "middle", "rear")
BAGS = {"suitcase", "backpack", "handbag"}
SEAT_CLASSES = {"Occupied": "occupied", "Unoccupied": "available"}


@dataclass(frozen=True)
class Detection:
    label: str
    box: tuple
    confidence: float
    track_id: int | None = None

    def anchor(self, kind="foot"):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, y2 if kind == "foot" else (y1 + y2) / 2)


def unit(value):
    return isinstance(value, (int, float)) and isfinite(value) and 0 <= value <= 1


def polygon(points):
    if points is None:
        return None
    if len(points) < 3 or any(len(p) != 2 or not all(map(unit, p)) for p in points):
        raise ValueError("Regions need at least three normalized points")
    contour = np.asarray(points, dtype=np.float32)
    if cv2.contourArea(contour) < 0.00001:
        raise ValueError("Region has no measurable area")
    return contour


def contains(contour, point):
    return contour is not None and cv2.pointPolygonTest(contour, point, False) >= 0


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - w*h
    return w*h / union if union > 0 else 0


@dataclass
class Cargo:
    since: float
    origin: tuple
    last_seen: float


@dataclass
class Seat:
    candidate: str = "uncertain"
    since: float = 0


class Cabin:
    def __init__(self, config, width, height):
        self.config = config
        ratio = config["aspect_ratio"]
        if not isfinite(ratio) or ratio <= 0 or abs((width / height) / ratio - 1) > 0.02:
            raise ValueError("Camera aspect ratio differs from calibration")
        self.zones = {k: polygon(config["zones"].get(k)) for k in ZONES}
        self.limited_zones = set(config.get("limited_zones", []))
        if not self.limited_zones.issubset(ZONES):
            raise ValueError("Unknown limited-visibility zone")
        self.capacities = config.get("zone_capacities", {})
        if any(k not in ZONES or type(v) is not int or v < 1 for k, v in self.capacities.items()):
            raise ValueError("Zone capacities must be positive integers for named zones")
        self.passages = {k: polygon(v) for k, v in config["passages"].items()}
        self.seats = config["seats"]
        self.availability_occluded = set(config.get("availability_occluded", []))
        if not self.availability_occluded.issubset(self.seats):
            raise ValueError("An occluded seat must have a mapped rectangle")
        for rect in self.seats.values():
            if len(rect) != 4 or not all(map(unit, rect)) or rect[0] >= rect[2] or rect[1] >= rect[3]:
                raise ValueError("Mapped seats need normalized rectangles of positive area")
        self.anchor = config.get("anchor", "foot")
        if self.anchor not in ("foot", "center"):
            raise ValueError("Person anchor must be foot or center")
        self.confirm = config.get("seat_confirm_s", 0.3)
        self.dwell = config.get("dwell_s", 2.0)
        if any(not isfinite(v) or v <= 0 for v in (self.dwell, self.confirm)):
            raise ValueError("Persistence durations must be finite and positive")
        self.seat_history = {name: Seat() for name in self.seats}
        self.cargo = {}
        self.last_time = None

    def seat_observations(self, objects):
        # A box may vote for only one physical seat. Competing or weak matches abstain.
        votes = {k: set() for k in self.seats}
        for obj in objects:
            if obj.label not in SEAT_CLASSES or obj.confidence < 0.4:
                continue
            scores = sorted(((iou(obj.box, rect), k) for k, rect in self.seats.items()), reverse=True)
            if scores and scores[0][0] >= 0.30:
                best, name = scores[0]
                if len(scores) > 1 and best - scores[1][0] < 0.07:
                    for score, tied in scores:
                        if best - score < 0.07:
                            votes[tied].add("uncertain")
                else:
                    state = SEAT_CLASSES[obj.label]
                    if state == "available" and name in self.availability_occluded:
                        state = "uncertain"
                    votes[name].add(state)
        for obj in objects:
            if obj.label not in BAGS or obj.confidence < 0.35:
                continue
            x, y = obj.anchor("center")
            for name, rect in self.seats.items():
                if (rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]
                        and iou(obj.box, rect) >= 0.10 and "available" in votes[name]):
                    votes[name].add("uncertain")
        return {k: next(iter(v)) if len(v) == 1 else "uncertain" for k, v in votes.items()}

    def update(self, now, objects, quality="usable"):
        if self.last_time is not None and now < self.last_time:
            raise ValueError("Source timestamps must be monotonic")
        if self.last_time is None or now - self.last_time > 0.75 or quality != "usable":
            self.cargo.clear()
            self.seat_history = {k: Seat() for k in self.seats}
        self.last_time = now
        usable = quality == "usable"
        counts = {k: 0 if usable and p is not None else None for k, p in self.zones.items()}
        ambiguous = set()
        if usable:
            for obj in objects:
                if obj.label not in ("person", "passenger"):
                    continue
                memberships = [k for k, region in self.zones.items() if contains(region, obj.anchor(self.anchor))]
                if len(memberships) == 1 and obj.confidence >= 0.35:
                    counts[memberships[0]] += 1
                elif memberships:
                    ambiguous.update(memberships)
        zones = {}
        for k, count in counts.items():
            status = "observed" if count is not None else "unobserved"
            if count is not None and (k in ambiguous or k in self.limited_zones):
                status = "uncertain"
            free = max(0, self.capacities[k] - count) if status == "observed" and k in self.capacities else None
            zones[k] = {"people": count, "status": status, "remaining": free}

        seats = {}
        for k, observed in self.seat_observations(objects if usable else []).items():
            history = self.seat_history[k]
            if history.candidate != observed:
                history.candidate, history.since = observed, now
            seats[k] = observed if now - history.since >= self.confirm else "uncertain"

        passages = {}
        active = {}
        for name, region in self.passages.items():
            duration = None
            if usable and region is not None:
                for obj in objects:
                    if obj.label not in BAGS or obj.track_id is None or obj.confidence < 0.35:
                        continue
                    anchor = obj.anchor()
                    key = (name, obj.track_id)
                    if not contains(region, anchor):
                        self.cargo.pop(key, None)
                        continue
                    history = self.cargo.get(key)
                    if history is None or now - history.last_seen > 0.35 or dist(anchor, history.origin) > 0.055:
                        history = Cargo(now, anchor, now)
                    history.last_seen = now
                    active[key] = history
                    duration = max(duration or 0, now - history.since)
            status = "unobserved"
            if usable and region is not None:
                status = "no candidate" if duration is None else ("review" if duration >= self.dwell else "monitoring")
            passages[name] = {"status": status, "dwell_s": round(duration or 0, 2)}
        # Allow a short missed detection, but publish alerts only when an object is observed.
        self.cargo = {k: v for k, v in self.cargo.items() if now - v.last_seen <= 0.35}
        self.cargo.update(active)
        return {"zones": zones, "seats": seats, "passages": passages}
