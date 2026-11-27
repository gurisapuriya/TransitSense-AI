from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from unittest import TestCase
from unittest.mock import Mock

from dashboard.serve import handler_for


class OperatorServiceTests(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name) / "state.json"

    def get(self, path):
        request = Mock()
        request.makefile.return_value = BytesIO(f"GET {path} HTTP/1.0\r\n\r\n".encode())
        output = BytesIO()
        request.sendall.side_effect = output.write
        handler_for(self.state)(request, ("127.0.0.1", 0), Mock())
        headers, _, body = output.getvalue().partition(b"\r\n\r\n")
        return headers, body

    def test_missing_live_source_and_stale_frames(self):
        self.assertEqual(json.loads(self.get("/api/live")[1])["status"],"offline")
        self.state.write_text(json.dumps({"status":"running","updated":time.time()-60,"sample":{"visible":8}}))
        stale=json.loads(self.get("/api/live")[1])
        self.assertEqual(stale["status"],"stale")
        self.assertIsNone(stale["sample"])

    def test_current_and_final_observations(self):
        for status in ["running","ended"]:
            data={"status":status,"updated":time.time(),"sample":{"visible":2}}
            self.state.write_text(json.dumps(data))
            self.assertEqual(json.loads(self.get("/api/live")[1]),data)

    def test_listed_replays_are_playable_aggregate_time_series(self):
        for row in json.loads(self.get("/api/runs")[1]):
            headers,body=self.get("/api/runs/"+row["id"])
            self.assertIn(b"200 OK",headers)
            samples=json.loads(body)["samples"]
            self.assertGreater(len(samples),1)
            times=[s["t"] for s in samples]
            self.assertEqual(times,sorted(set(times)))
            self.assertNotIn(b"track_id",body)
