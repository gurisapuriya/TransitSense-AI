import unittest

from inference.doorway import CrossingCounter, Doorway
from evaluation.evaluate_doorway import match_events


class DoorwayTests(unittest.TestCase):
    def setUp(self):
        self.counter = CrossingCounter(Doorway((10, 50), (90, 50), 1, 4, 0.24, 2))

    def walk(self, ys, x=50, track_id=1, start=0):
        events = []
        for i, y in enumerate(ys):
            event = self.counter.observe(track_id, (x, y), start + i / 25)
            if event:
                events.append(event[0])
        return events

    def test_crossing_and_later_reverse(self):
        self.assertEqual(self.walk([30, 32, 49, 53, 68, 70, 65, 50, 35, 30]),
                         ["boarding", "alighting"])

    def test_hover_and_tentative_crossing(self):
        self.assertEqual(self.walk([30, 32, 49, 52, 48, 55, 48, 40, 36]), [])

    def test_start_on_line(self):
        self.assertEqual(self.walk([50, 52, 60, 62]), [])

    def test_no_crossing_beyond_endpoint(self):
        self.assertEqual(self.walk([30, 32, 60, 65], x=95), [])

    def test_observations_inside_margin_follow_actual_route_around_endpoint(self):
        points = [(80, 30), (80, 35), (95, 48), (95, 52), (80, 60), (80, 62)]
        self.assertFalse(any(self.counter.observe(1, p, i / 25) for i, p in enumerate(points)))

    def test_lost_track_does_not_bridge_long_gap(self):
        self.walk([30, 32])
        self.assertEqual(self.walk([60, 62], start=1), [])

    def test_short_gap_allows_continuous_track(self):
        self.walk([30, 32])
        self.assertEqual(self.walk([60, 62], start=0.16), ["boarding"])

    def test_reassigned_id_does_not_inherit_other_persons_path(self):
        self.walk([30, 32])
        self.assertEqual(self.walk([60, 62], track_id=2, start=0.12), [])

    def test_opposite_interior(self):
        self.counter = CrossingCounter(Doorway((10, 50), (90, 50), -1, 4, 0.24, 2))
        self.assertEqual(self.walk([30, 32, 60, 62]), ["alighting"])

    def test_diagonal_door_with_horizontal_motion(self):
        counter = CrossingCounter(Doorway((10, 10), (90, 90), 1, 4, 0.24, 2))
        events = []
        for i, x in enumerate([70, 60, 40, 30]):
            event = counter.observe(4, (x, 50), i / 25)
            if event:
                events.append(event)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "boarding")
        self.assertAlmostEqual(events[0][1], 0.06)

    def test_one_to_one_scoring(self):
        hits, missed, extra = match_events([1, 2, 5], [0, 1, 1.25, 2.4, 6], 0.5)
        self.assertEqual(hits, [(1, 1), (2, 2.4)])
        self.assertEqual(missed, [5])
        self.assertEqual(extra, [0, 1.25, 6])


if __name__ == "__main__":
    unittest.main()
