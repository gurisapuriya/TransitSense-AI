"""Directional crossing of a finite doorway segment in a fixed camera view."""
import json
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path


@dataclass(frozen=True)
class Doorway:
    start: tuple[float, float]
    end: tuple[float, float]
    inside_sign: int
    margin: float
    max_gap_s: float
    confirm_frames: int

    @classmethod
    def load(cls, path: Path, width: int, height: int):
        config = json.loads(path.read_text())

        def pixel_point(name):
            x, y = config[name]
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"{name} must be a normalized point within the image")
            return x * (width - 1), y * (height - 1)

        start, end, inside = map(pixel_point, ("start", "end", "inside"))
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = hypot(dx, dy)
        cross = dx * (inside[1] - start[1]) - dy * (inside[0] - start[0])
        margin = config["margin_height_fraction"] * height
        gap, frames = config["max_gap_s"], config["confirm_frames"]
        if not length or not 0 < margin < abs(cross) / length:
            raise ValueError("Use distinct line endpoints and an inside point clear of the margin")
        if not isfinite(gap) or gap <= 0 or type(frames) is not int or frames < 1:
            raise ValueError("max_gap_s and confirm_frames must be positive")
        return cls(start, end, 1 if cross > 0 else -1, margin, gap, frames)

    def distance(self, point):
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        x, y = point[0] - self.start[0], point[1] - self.start[1]
        return self.inside_sign * (dx * y - dy * x) / hypot(dx, dy)

    def on_segment(self, point):
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        projection = (point[0] - self.start[0]) * dx + (point[1] - self.start[1]) * dy
        return 0 <= projection <= dx * dx + dy * dy


@dataclass
class Track:
    point: tuple[float, float]
    time: float
    distance: float
    stable_side: int = 0
    candidate: int = 0
    samples: int = 0
    crossing_s: float | None = None


class CrossingCounter:
    def __init__(self, door):
        self.door = door
        self.tracks = {}
        self.totals = {"boarding": 0, "alighting": 0}

    def expire(self, now):
        self.tracks = {i: t for i, t in self.tracks.items() if now - t.time <= self.door.max_gap_s}

    def observe(self, track_id, point, now):
        """Return (direction, interpolated crossing time) after confirming the new side."""
        door = self.door
        distance = door.distance(point)
        side = 1 if distance > door.margin else -1 if distance < -door.margin else 0
        track = self.tracks.get(track_id)
        if track is None or now - track.time > door.max_gap_s:
            track = self.tracks[track_id] = Track(point, now, distance)

        # Intersect the actual path, including observations inside the margin.
        if (distance > 0) != (track.distance > 0):
            fraction = track.distance / (track.distance - distance)
            intersection = tuple(a + fraction * (b - a) for a, b in zip(track.point, point))
            track.crossing_s = None
            if door.on_segment(intersection):
                track.crossing_s = track.time + fraction * (now - track.time)
        track.point, track.time, track.distance = point, now, distance

        track.samples = track.samples + 1 if side == track.candidate else 1
        track.candidate = side
        if not side or track.samples < door.confirm_frames:
            return None
        previous = track.stable_side
        track.stable_side = side
        crossing = track.crossing_s
        track.crossing_s = None
        if previous and side != previous and crossing is not None:
            direction = "boarding" if side == 1 else "alighting"
            self.totals[direction] += 1
            return direction, crossing
        return None
