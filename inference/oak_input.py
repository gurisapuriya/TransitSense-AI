"""Optional DepthAI v3 capture for a USB OAK-D Pro (RVC2)."""
import argparse
from contextlib import closing
from datetime import timedelta
import json
from statistics import median
import time

SIZE = (640, 480)
FPS = 15


def _packets(*, depth=False, timeout_s=5.0):
    try:
        import depthai as dai
    except ImportError as exc:
        raise RuntimeError("OAK input requires the optional depthai 3.x package. See docs/edge.md.") from exc
    if dai.__version__.split(".")[0] != "3":
        raise RuntimeError(f"OAK input requires DepthAI 3.x, detected {dai.__version__}")
    with dai.Pipeline() as pipeline:
        rgb_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb = rgb_camera.requestOutput(SIZE, type=dai.ImgFrame.Type.BGR888p, fps=FPS,
                                       enableUndistortion=depth)
        output = rgb
        if depth:
            # RVC2 aligns stereo depth in the StereoDepth node.
            stereo = pipeline.create(dai.node.StereoDepth)
            for socket, target in [(dai.CameraBoardSocket.CAM_B, stereo.left),
                                   (dai.CameraBoardSocket.CAM_C, stereo.right)]:
                camera = pipeline.create(dai.node.Camera).build(socket)
                camera.requestOutput((640, 400), fps=FPS).link(target)
            rgb.link(stereo.inputAlignTo)
            sync = pipeline.create(dai.node.Sync)
            sync.setSyncThreshold(timedelta(seconds=0.5 / FPS))
            rgb.link(sync.inputs["rgb"])
            stereo.depth.link(sync.inputs["depth"])
            output = sync.out
        # Bound queue latency when the host detector runs slower than capture.
        queue = output.createOutputQueue(maxSize=1, blocking=False)
        pipeline.start()
        while pipeline.isRunning():
            deadline = time.monotonic() + timeout_s
            packet = queue.tryGet()
            while packet is None:
                if not pipeline.isRunning():
                    raise RuntimeError("OAK camera disconnected or pipeline stopped")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"No OAK frames received for {timeout_s:g} seconds")
                time.sleep(0.005)
                packet = queue.tryGet()
            rgb_packet = packet["rgb"] if depth else packet
            age_ms = (dai.Clock.now() - rgb_packet.getTimestamp()).total_seconds() * 1000
            frame = rgb_packet.getCvFrame()
            if frame.shape != (SIZE[1], SIZE[0], 3):
                raise RuntimeError(f"Unexpected OAK color frame dimensions: {frame.shape}")
            depth_map = packet["depth"].getFrame() if depth else None
            if depth and depth_map.shape != frame.shape[:2]:
                raise RuntimeError(f"Aligned depth dimensions differ from RGB: {depth_map.shape}")
            yield rgb_packet, frame, depth_map, age_ms
        raise RuntimeError("OAK camera disconnected or pipeline stopped")


def frames_oak():
    """Yield (seconds since the first device frame, BGR image). Close on early exit."""
    origin = None
    previous = None
    with closing(_packets()) as packets:
        for packet, image, _, _ in packets:
            stamp = packet.getTimestampDevice()
            if previous is not None and stamp <= previous:
                raise RuntimeError("OAK capture timestamps stopped increasing")
            if origin is None:
                origin = stamp
            previous = stamp
            yield (stamp - origin).total_seconds(), image


def probe(frame_count, depth):
    """Measure capture/transport and RGB-D validity without running a detector."""
    import numpy as np
    import psutil

    if frame_count < 2:
        raise ValueError("Use at least two frames")
    process = psutil.Process()
    ages, valid_depth = [], []
    peak_rss = skipped = 0
    prev_seq = first_arrival = None
    with closing(_packets(depth=depth)) as packets:
        for packet, _, depth_map, age_ms in packets:
            now = time.perf_counter()
            if first_arrival is None:
                first_arrival = now
            seq = packet.getSequenceNum()
            if prev_seq is not None:
                if seq <= prev_seq:
                    raise RuntimeError("OAK RGB sequence number stopped increasing")
                skipped += seq - prev_seq - 1
            prev_seq = seq
            ages.append(age_ms)
            if depth_map is not None:
                valid_depth.append(np.count_nonzero(depth_map) / depth_map.size)
            peak_rss = max(peak_rss, process.memory_info().rss)
            if len(ages) == frame_count:
                break
    return {"requested_fps": FPS, "rgb_size": list(SIZE), "received_frames": len(ages),
            "received_fps": round((frame_count - 1) / (now - first_arrival), 2),
            "skipped_sequence_numbers": skipped, "capture_to_host_p50_ms": round(median(ages), 2),
            "capture_to_host_p95_ms": round(float(np.percentile(ages, 95)), 2),
            "sampled_peak_rss_mib": round(peak_rss / 1024**2, 2),
            "mean_valid_depth_fraction": round(float(np.mean(valid_depth)), 4) if depth else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure local OAK-D Pro capture and transport")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--depth", action="store_true", help="Also acquire aligned stereo depth")
    args = parser.parse_args()
    try:
        print(json.dumps(probe(args.frames, args.depth), indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"OAK capture failed: {exc}\n")
