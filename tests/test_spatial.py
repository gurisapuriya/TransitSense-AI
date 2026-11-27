from dataclasses import replace
import json
from unittest import TestCase

import numpy as np

from evaluation.simulated_stop import fixture, person, simulate
from inference.monitor import image_quality
from inference.spatial import Cabin, Detection


class SpatialTests(TestCase):
    def setUp(self):
        self.layout = fixture()
        self.cabin = Cabin(self.layout, 1000, 500)

    def test_zone_visibility_and_uncertain_capacity(self):
        self.layout["zones"]["rear"] = None
        self.layout["limited_zones"] = ["middle"]
        c = Cabin(self.layout, 1000, 500)
        data = c.update(0, [person(.2,.5,4), replace(person(.2,.5,5), confidence=.2)])
        self.assertEqual(data["zones"]["front"], {"people": 1, "status": "uncertain", "remaining": None})
        self.assertEqual(data["zones"]["rear"]["people"], None)
        self.assertEqual(data["zones"]["middle"]["remaining"], None)

    def test_overlapping_regions_do_not_double_count(self):
        self.layout["zones"]["middle"] = self.layout["zones"]["front"]
        data = Cabin(self.layout, 1000, 500).update(0, [person(.2,.5,4)])
        self.assertEqual([v["people"] for v in data["zones"].values()], [0,0,0])
        self.assertEqual(data["zones"]["middle"]["status"], "uncertain")

    def test_seat_confirmation_and_loss(self):
        box = tuple(self.layout["seats"]["1A"])
        vacant = Detection("Unoccupied", box, .92)
        self.assertEqual(self.cabin.update(0,[vacant])["seats"]["1A"], "uncertain")
        self.assertEqual(self.cabin.update(.4,[vacant])["seats"]["1A"], "available")
        self.assertEqual(self.cabin.update(.5,[])["seats"]["1A"], "uncertain")
        self.assertEqual(self.cabin.update(.6,[vacant])["seats"]["1A"], "uncertain")

    def test_bag_or_conflict_vetoes_availability(self):
        box = tuple(self.layout["seats"]["1A"])
        vacant = Detection("Unoccupied", box, .95)
        for second in [Detection("backpack", box, .7), Detection("Occupied", box, .8)]:
            result = Cabin(self.layout,1000,500)
            result.update(0,[vacant,second])
            self.assertEqual(result.update(.4,[vacant,second])["seats"]["1A"], "uncertain")

    def test_dwell_resets_on_exit_and_id_change(self):
        bag = Detection("suitcase", (.45,.39,.51,.53), .9, 22)
        for i in range(20):
            self.cabin.update(i/10, [bag])
        out = replace(bag, box=(.45,.59,.51,.73))
        self.cabin.update(2,[out])
        self.assertEqual(self.cabin.update(2.1,[bag])["passages"]["aisle"]["dwell_s"],0)
        self.assertEqual(self.cabin.update(2.2,[replace(bag,track_id=23)])["passages"]["aisle"]["dwell_s"],0)

    def test_moving_bag_and_standing_passenger(self):
        for i in range(30):
            x = .1+i*.021
            bag=Detection("handbag", (x,.45,x+.05,.5), .9, 87)
            s=self.cabin.update(i/10,[bag,person(.55,.47,5)])
            self.assertNotEqual(s["passages"]["aisle"]["status"],"review")

    def test_full_stop_reference(self):
        replay=simulate()
        samples=replay["samples"]
        self.assertEqual([e["direction"] for e in replay["events"]], ["boarding","boarding","alighting"])
        self.assertTrue(all(e["time_s"] <= e["confirmed_s"] for e in replay["events"]))
        self.assertEqual(samples[-1]["flow"], {"boarding":2,"alighting":1})
        self.assertEqual([s["t"] for s in samples if s["passages"]["aisle"]["status"]=="review"],
                         [i/10 for i in range(90,130)])
        self.assertFalse(any(s["passages"]["door"]["status"]=="review" for s in samples))
        dark=samples[169]
        self.assertIsNone(dark["visible"])
        self.assertTrue(all(z["people"] is None for z in dark["zones"].values()))
        self.assertTrue(all(v=="uncertain" for v in dark["seats"].values()))
        self.assertNotIn("track_id", json.dumps(replay))

    def test_invalid_aspect(self):
        with self.assertRaisesRegex(ValueError,"aspect ratio"):
            Cabin(self.layout,320,240)

    def test_black_and_white_frames_withhold_counts(self):
        image=np.zeros((240,320,3),np.uint8)
        self.assertEqual(image_quality(image), "too dark")
        self.assertEqual(image_quality(image+255), "overexposed")
        self.assertEqual(image_quality(image+60), "low detail")
