import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

from dashboard.data import BASE, replays

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@skipUnless(AppTest, "Install dashboard/requirements.txt for Streamlit tests")
class StreamlitConsoleTests(TestCase):
    def open_console(self):
        return AppTest.from_file(str(BASE / "console.py"), default_timeout=15).run()

    def test_sources_and_replay_controls(self):
        app = self.open_console()
        self.assertEqual([m.value for m in app.metric], ["4", "2", "0", "8"])
        for key in replays():
            app.selectbox[0].set_value(key).run()
            self.assertFalse(app.exception, key)
            self.assertEqual(len(app.metric), 4, key)
        app.selectbox[0].set_value("controlled").run()
        app.slider[0].set_value(0.0).run()
        self.assertEqual(app.metric[0].value, "2")
        app.button[0].click().run()
        self.assertEqual(app.button[0].label, "Pause")
        app.button[0].click().run()
        self.assertEqual(app.button[0].label, "Play")
        self.assertFalse(app.exception)

    def test_live_disconnect_and_local_recording(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "live.json"
            demo = replays()["controlled"][1]
            with patch.dict(os.environ, TRANSITSENSE_STATE=str(state), TRANSITSENSE_REPLAY=str(demo)):
                app = self.open_console()
                app.selectbox[0].set_value("local").run()
                self.assertEqual(app.metric[0].value, "2")
                app.selectbox[0].set_value("live").run()
                self.assertIn("No monitor connected", app.info[0].value)
                report = json.loads(demo.read_text())
                state.write_text(json.dumps({**report, "sample": report["samples"][0], "status": "ended"}))
                app.run()
                self.assertEqual(app.metric[0].value, "2")
                state.write_text(json.dumps({"status": "running", "updated": 0, "sample": report["samples"][0]}))
                app.run()
                self.assertIn("Connection stale", app.info[0].value)
                self.assertFalse(app.metric)
