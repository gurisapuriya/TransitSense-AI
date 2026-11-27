# OAK-D Pro and Raspberry Pi bench procedure

The camera adapter requests 640 × 480 BGR at 15 FPS from an OAK-D Pro. The detector and tracker run on the host connected to the camera. The monitor receives one image and its device capture time relative to the first frame. A one-frame queue drops old images when the host falls behind. Calibrate the exact camera output and mounting position used for the demonstration. [Luxonis camera node](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/camera/) · [Luxonis node queues](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes)

Development checks ran on an Apple M4 MacBook Pro with 16 GB of memory. The project interpreter had no DepthAI installation and the visible USB inventory listed no OAK. Automated tests cover frame timing, loss of connection, timeout, cleanup, depth validity, and skipped sequence numbers with simulated packets. Physical USB capture, practical inference speed on the Pi, and RGB-D alignment accuracy remain open measurements.

## Camera and host

Attach the OAK to the machine doing inference. On a Pi, use Raspberry Pi OS with a Python version that has a DepthAI v3 wheel for its architecture, and set up the Linux USB permissions described by Luxonis. If camera resets occur on a Pi-powered connection, verify the power supply and test a powered USB hub. [DepthAI installation](https://docs.luxonis.com/software-v3/depthai) · [Luxonis Raspberry Pi deployment](https://docs.luxonis.com/hardware/platform/deploy/to-rpi)

From the repository root on that host, use a dedicated camera environment. The following commands install the small capture probe and measure 150 frames per mode. Each probe prints a JSON summary. Neither probe writes images to disk.

```sh
python3 -m venv ~/venvs/transitsense-camera
. ~/venvs/transitsense-camera/bin/activate
python -m pip install 'depthai>=3,<4' numpy psutil opencv-python
python -m inference.oak_input --frames 150
python -m inference.oak_input --frames 150 --depth
```

The RGB-D probe pairs color with stereo depth from the left and right sockets using the RVC2 depth-alignment path and a synchronization tolerance of half a frame period. It checks matching dimensions and reports the mean fraction of pixels with nonzero depth. To verify physical alignment, place an object with a clear edge in the scene and compare its depth boundary with its RGB edge using the vendor's viewing example. The operator monitor currently uses RGB for its estimates. [Luxonis depth alignment example](https://docs.luxonis.com/software-v3/depthai/examples/image_align/depth_align)

Capture-to-host latency uses the synchronized message timestamp at host receipt. Capture-probe FPS measures delivered frames over wall time after the first arrival. Sequence gaps expose dropped frames. Peak RSS is sampled for the host probe process. Run the application benchmark separately to include model inference, tracking and spatial logic. [Luxonis latency measurement](https://docs.luxonis.com/software-v3/depthai/tutorials/optimizing)

## Run the monitor with live capture

Use a single host environment with both the project dependencies and DepthAI installed, and make the trained checkpoints available at the paths in the project README. In that environment, set the layout to your camera's calibration and run a one-minute trial. The following commands use the active environment's Python. Add `--door path/to/door.json` when the doorway is in view and separately calibrated.

```sh
python -m pip install -r requirements.txt
echo 'Enter the path to this camera layout JSON'
read -r camera_layout
python -m inference.monitor --oak --layout "$camera_layout" --device cpu \
  --seconds 60 --report results/monitor/oak.json
```

During capture, open a second terminal in the repository. On the project Mac, run `python3.13 -m dashboard.serve` for the Streamlit console. On another host, use the dashboard environment from the README or run `python3 -m dashboard.serve --basic` for the lightweight display. The service reads `results/monitor/live.json` and serves the dashboard at `http://127.0.0.1:8765`.

## Operator display on a Pi

A first hardware configuration pairs the OAK with the development host for USB capture and inference. The Pi runs a browser for aggregate results. Keep the operator service bound to loopback on the inference host. From the Pi, use an existing SSH account on that host to forward its dashboard to the Pi's loopback interface. Use the team's own SSH destination. This example uses the default dashboard port, 8765.

```sh
echo 'Enter the SSH user@inference-host'
read -r inference_host
ssh -N -o ExitOnForwardFailure=yes -L "127.0.0.1:8091:127.0.0.1:8765" "$inference_host"
```

Open `http://127.0.0.1:8091` in the Pi's browser while the SSH connection runs. The browser receives the operator dashboard and its aggregate updates. The video stays on the inference host.

To compare host placements, run the monitor on each machine with the same weights, profile, and recording before a live-camera trial. Record host model, OS, Python and package versions, scene/profile, number of frames, processing FPS, p50/p95 processing latency, peak process RSS, missed events and extras. Run the OAK capture probe separately on each USB host and retain the JSON summaries. Real-time operation needs both enough processed frames per second for the event timings and acceptable count accuracy on staged physical crossings.

## Staged acceptance trial

Use a fixed, measured doorway and a consented bus-like setup. Align the source clock with an observer at the start. Keep the calibration session and a later scored session separate. If one view cannot cover both the doorway and the rear, test each camera placement as a separate run.

1. At the door, include individual entries, exits, a reversal, two people close together, and an interval with nobody crossing. Before inspecting predictions, record each crossing in a CSV with columns `clip,direction,time_s`. Use `boarding` or `alighting`. A reviewed zero-event clip can be represented by a single row with its clip name and empty direction and time fields.
2. In the cabin, mark physical seat IDs, zone boundaries and visibility on the preview image. At agreed timestamps, have an observer count visible people per zone and label each mapped seat occupied, available or uncertain. Include an obscured seat and an object on a seat. Score each observation once; avoid scoring every repeated video frame as an independent case.
3. Place a bag in the aisle and remove it, carry it through, leave it outside the aisle, and include a standing passenger. Record the start and end of each placement independently. Compare alert episodes, missed placements, extra alerts, and time from placement to alert. These trials establish whether the proxy corresponds to an actual impediment.
4. Run a finite monitor report per session. It stores aggregate observations, anonymous crossing times and runtime measurements. With the corresponding door geometry, the following command scores the exported events against the observer's CSV.

```sh
python -m evaluation.evaluate_doorway --clip trial_01 \
  --reference results/monitor/manual-doorway.csv --report results/monitor/oak.json
```

Report the denominator and per-direction errors, including failed cases. A low count error can conceal a missed crossing and a false crossing in the same direction. Record whether each hardware trial used Pi or Mac inference and the tested camera placement.
