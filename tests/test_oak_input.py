from contextlib import closing
from datetime import timedelta
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from inference.oak_input import _packets, frames_oak, probe


class OakInputTests(TestCase):
    def setUp(self):
        self.dai = MagicMock(__version__="3.0.0")
        self.pipeline = self.dai.Pipeline.return_value.__enter__.return_value
        self.pipeline.isRunning.return_value = True
        self.camera = self.pipeline.create.return_value.build.return_value
        self.queue = self.camera.requestOutput.return_value.createOutputQueue.return_value
        self.dai.Clock.now.return_value = timedelta(seconds=3)
        self.sdk = patch.dict("sys.modules", depthai=self.dai)
        self.sdk.start()
        self.addCleanup(self.sdk.stop)

    def packet(self, ts_s, seq=1):
        packet = MagicMock()
        packet.getTimestampDevice.return_value = timedelta(seconds=ts_s)
        packet.getTimestamp.return_value = timedelta(seconds=ts_s)
        packet.getSequenceNum.return_value = seq
        packet.getCvFrame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        return packet

    def test_relative_device_time_and_early_cleanup(self):
        self.queue.tryGet.side_effect = [self.packet(100), self.packet(100.4)]
        with closing(frames_oak()) as frames:
            self.assertEqual(next(frames)[0], 0)
            later, image = next(frames)
            self.assertAlmostEqual(later, 0.4)
            self.assertEqual(image.shape, (480, 640, 3))
        self.dai.Pipeline.return_value.__exit__.assert_called_once()

    def test_backward_timestamp_releases_pipeline(self):
        self.queue.tryGet.side_effect = [self.packet(100), self.packet(95)]
        frames = frames_oak()
        next(frames)
        with self.assertRaisesRegex(RuntimeError, "timestamps"):
            next(frames)
        self.dai.Pipeline.return_value.__exit__.assert_called_once()

    def test_stream_timeout_and_disconnect_release_pipeline(self):
        self.queue.tryGet.return_value = None
        with self.assertRaises(TimeoutError):
            next(_packets(timeout_s=0))
        self.pipeline.isRunning.return_value = False
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            next(frames_oak())
        self.assertEqual(self.dai.Pipeline.return_value.__exit__.call_count, 2)

    def test_depth_probe_counts_nonzero_pixels_and_sequence_gaps(self):
        node = self.pipeline.create.return_value
        queue = node.out.createOutputQueue.return_value
        depth = MagicMock()
        depth_image = np.zeros((480, 640), dtype=np.uint16)
        depth_image[:240] = 1250
        depth.getFrame.return_value = depth_image
        queue.tryGet.side_effect = [{"rgb": self.packet(2.8, 10), "depth": depth},
                                    {"rgb": self.packet(2.9, 12), "depth": depth}]
        report = probe(2, depth=True)
        self.assertEqual(report["mean_valid_depth_fraction"], 0.5)
        self.assertEqual(report["skipped_sequence_numbers"], 1)
        self.assertAlmostEqual(report["capture_to_host_p50_ms"], 150)

    def test_missing_optional_dependency(self):
        with patch.dict("sys.modules", depthai=None), self.assertRaisesRegex(RuntimeError, "optional"):
            next(frames_oak())
