# TransitSense mock-bus field guide

**First milestone:** obtain one stable RGB stream and one depth-probe result from the Raspberry Pi with the OAK looking at an empty room. **Final milestone:** demonstrate and score the claim that an operator can see crowding near the entrance while usable space remains farther inside. Keep simple doorway counts and the cabin distribution experiment as distinct results.

This guide assumes one **USB OAK-D with RGB and stereo cameras**, a Raspberry Pi 4 with 8 GB RAM, and the existing Mac checkout. Read the model label and record the exact variant. The repository adapter was written for a USB RVC2 OAK-D Pro; compatibility with your exact unit is a bench check. A PoE model needs its own discovery/power procedure. All run names and suggested trial sizes below are examples. The project's camera commands request 640 × 480 at 15 FPS. The live monitor saves aggregate JSON; recording identifiable footage is a separate, deliberate step.

## 1. Bring up the Pi and transfer the prototype

Use a Pi 4-compatible USB-C supply (5 V, 3 A), microSD with room for packages and short recordings, a short USB 3 data cable, a stable camera mount, masking tape, chairs, and a backpack or suitcase. If the Pi already boots a supported 64-bit Raspberry Pi OS, use it. For a fresh card, Raspberry Pi Imager can set the hostname, username, Wi-Fi and SSH. Imaging overwrites the selected card. Example hostname below: `transitsense`. Connect the camera to a blue USB 3 port on the Pi. [Raspberry Pi Imager](https://www.raspberrypi.com/documentation/computers/imager.html) · [Pi 4 power specification](https://pip-assets.raspberrypi.com/categories/545-raspberry-pi-4-model-b/documents/RP-008344-DS-7-raspberry-pi-4-product-brief.pdf) · [Luxonis USB guide](https://docs.luxonis.com/hardware/platform/deploy/usb-deployment-guide)

**On the Mac**, use the username you configured in Imager. Use the Pi's IP address if its hostname does not resolve. This transfer includes the local trained weights, which are stored under the Git-ignored `runs/` directory. Use a new destination for the trial to preserve any previous Pi checkout.

```sh
cd /Users/HP/TransitSense-AI
pi_host='YOUR_PI_USERNAME@transitsense.local'
ssh "$pi_host" 'mkdir -p ~/transitsense-trial'
# If rsync is missing on the Pi: ssh -t "$pi_host" 'sudo apt update && sudo apt install -y rsync'
rsync -avR --exclude '__pycache__' requirements.txt configs/ inference/ dashboard/ evaluation/ tests/ \
  yolo11n.pt runs/transitsense/passengers_baseline/weights/best.pt \
  runs/transitsense/seats_clean_v1/weights/best.pt "$pi_host:transitsense-trial/"
ssh "$pi_host"
```

**On the Pi**, install the capture environment first. Start in a new shell in the project directory for each later command block. Activate the same environment each time.

```sh
cd ~/transitsense-trial
uname -m
getconf LONG_BIT
python3 --version
sudo apt update
sudo apt install -y python3-venv ffmpeg usbutils rsync
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: 'depthai>=3,<4' numpy psutil opencv-python
mkdir -p results/monitor/field datasets/raw/staged_mock_bus
lsusb
```

Look for a 64-bit userspace and `aarch64`. For first-time Linux USB access, install the OAK udev rule following the **Linux udev rules** section of the vendor guide, reload the rules and reconnect the OAK. Run the application as your normal account. If pip cannot find an ARM64 wheel, check the Python version against the available DepthAI v3 wheels and use a compatible interpreter in a fresh environment. Keep the existing working Mac environment. [Luxonis USB setup and power troubleshooting](https://docs.luxonis.com/hardware/platform/deploy/usb-deployment-guide) · [DepthAI v3 installation](https://docs.luxonis.com/software-v3/depthai)

```sh
python -m inference.oak_input --frames 150 > results/monitor/field/pi-rgb.json
python -m inference.oak_input --frames 150 --depth > results/monitor/field/pi-depth.json
cat results/monitor/field/pi-rgb.json results/monitor/field/pi-depth.json
```

**Check before setting up actors.** Both commands should finish successfully and report 150 received frames. Read delivered FPS, sequence gaps, capture-to-host p95 and valid-depth fraction. Repeat depth capture with a textured box at two measured distances. In the vendor viewer/example, inspect whether RGB and depth edges line up and whether the farther position measures farther away. Save a non-identifying screenshot or notes. A nonzero depth fraction alone cannot establish alignment. If the device resets, check the USB data cable, USB 3 port and adequate power; a powered USB 3 hub is a practical diagnostic. Close the viewer/probe before starting another camera process. [Luxonis RGB-D alignment example](https://docs.luxonis.com/software-v3/depthai/examples/image_align/depth_align)

Then install inference dependencies and verify all models are readable on the Pi.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python -m pip freeze > results/monitor/field/pi-packages.txt
python - <<'PY'
from ultralytics import YOLO
for path in ("yolo11n.pt", "runs/transitsense/passengers_baseline/weights/best.pt",
             "runs/transitsense/seats_clean_v1/weights/best.pt"):
    print(path, YOLO(path).names)
PY
```

If PyTorch or Ultralytics cannot be installed on the Pi, preserve the capture environment. Pi capture plus Mac replay inference will still let you collect and score the experiment; separately report the limits of live inference on the Pi.

## 2. Build and measure the mock bus

Use a private room with a background and entry route you control. Agree with volunteers on recording, who can review the footage, any classroom display, and a deletion date. Use volunteers who agree to each intended use, keep bystanders out of view, and record without audio. Keep raw video locally in the ignored staged-data folder. A diagram, aggregate replay and numerical scores can be shared independently of permission to show identifiable video. Follow your course's participant procedures.

1. Tape a doorway at one end, a center aisle, and three consecutive floor zones. Put two chairs in each zone if the room allows it. Give each chair an ID, such as F1/F2, M1/M2 and R1/R2. Leave room for normal movement. Measure the mock-up, draw a floor plan, and mark fixed camera positions with tape.
2. Define experimental capacity from the physical layout. For example, two chair positions and one marked standing position in a zone give three designated places. Record the definition before testing. This is a classroom occupancy proxy. Keep measured marked places distinct from a claim about safe vehicle capacity. Record seat availability separately.
3. Try one fixed wide view. If the threshold or rear seats are hidden, run **door view** and **cabin view** as two independent experiments with the same OAK. In the door view show both sides of the line. In the cabin view show the zone marks, seat surfaces and an aisle patch for luggage. Record mount height, angle, orientation, lighting and blind spots for each view. Results from successive takes cannot be combined into a simultaneous whole-bus observation.
4. Assign an operator and an independent observer. With three team members, the operator can start a timed take and then join the second actor while the observer labels. Freeze the working configuration after **session A** for calibration. Record **session B** later as the scored session. Rotate roles, order or clothing where feasible. Keep whole sessions together when splitting data. Use a new session C for confirmation after any B-informed tuning.

## 3. Capture calibration frames and recorded trials

The built-in `--oak --report` command saves measurements and events; a video file for later independent annotation needs a recorder. Below is a small optional **RGB-only** recipe using the project's actual OAK frame generator. Run it on the capture host, in the activated environment, only for agreed takes. It writes an MP4 and a frame-to-device-time CSV and refuses to replace an existing take. An interrupted take may leave partial files; use a new take ID. It does not start the dashboard or inference. Stop other camera clients first.

```sh
record_take () {
  python - "$1" "$2" <<'PY'
from contextlib import closing
import csv, re, sys
from pathlib import Path
import cv2
from inference.oak_input import FPS, SIZE, frames_oak

name, duration = sys.argv[1], int(sys.argv[2])
if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or duration <= 0:
    raise SystemExit("Use a simple take ID and a positive duration in seconds")
folder = Path("datasets/raw/staged_mock_bus") / name
folder.mkdir(parents=True, exist_ok=False)
video = folder / "rgb.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
if not writer.isOpened():
    raise SystemExit("Cannot open the MP4 writer")
worst_drift = max_gap = 0
last = None
try:
    with closing(frames_oak()) as stream, (folder/"clock.csv").open("x", newline="") as out:
        log = csv.writer(out)
        log.writerow(("frame", "device_s", "replay_s"))
        for i, (t, frame) in enumerate(stream):
            writer.write(frame)
            log.writerow((i, round(t, 6), round(i/FPS, 6)))
            worst_drift = max(worst_drift, abs(t-i/FPS))
            if last is not None:
                max_gap = max(max_gap, t-last)
            last = t
            if t >= duration:
                break
finally:
    writer.release()
print(video, "frames:", i+1, "max clock drift (s):", round(worst_drift, 3),
      "max capture gap (s):", round(max_gap, 3))
PY
}
record_take A_door_empty 5
ffmpeg -v error -n -i datasets/raw/staged_mock_bus/A_door_empty/rgb.mp4 \
  -frames:v 1 results/monitor/field/door-empty.png
```

With the OAK in the cabin position, run:

```sh
record_take A_cabin_empty 5
ffmpeg -v error -n -i datasets/raw/staged_mock_bus/A_cabin_empty/rgb.mp4 \
  -frames:v 1 results/monitor/field/cabin-empty.png
```
 Inspect each MP4 and verify the expected scene, duration and orientation. For scoring a replay, observer labels and predictions must use the **same MP4 clock**; the sidecar preserves true capture times. At 15 FPS one frame is about 0.067 s. If the clock drifts by more than about 0.1 s or has sizeable gaps through an event, flag the take and repeat it for timing comparisons. Keep failed captures in the run log. For an independent **live** trial, an observer can film a visible start cue and the action with a second consented device; align its clock with the monitor's source time before comparing event times. Do not open two clients on the same OAK at once.

## 4. Calibrate the two views

Make working copies on the capture host. The old camera coordinates belong to the example buses. Use the copies for the room.

```sh
cp -n configs/doorway_view.json configs/mock-door-view.json
cp -n configs/pcds_sample_0006_door.json configs/mock-door-line.json
cp -n configs/cabin_real.json configs/mock-cabin.json
```

Open the empty-room PNG at its original 640 × 480 resolution in an image editor with pixel coordinates. Record each clicked point. Convert pixels to normalized coordinates with `x/639` and `y/479`, using the top-left origin.

- **Door.** In `mock-door-line.json`, replace `start` and `end` with the two ends of the taped threshold. Set `inside` to a point clearly inside the room beyond the margin. The short doorway run can start with the existing 0.035 height margin, two confirmations, and the experimental `--gap 0.80`. Verify one entry, one exit and one approach-and-retreat in session A.
- **Cabin.** Set `aspect_ratio` to 1.3333333333 for the requested OAK output. Replace all original zone and passage polygons, seat IDs and rectangles. Each seat rectangle is `[left, top, right, bottom]`. Clear old `limited_zones` and `availability_occluded` entries and enter only limitations of your room. Set unseen regions to JSON `null`. Use `zone_capacities` only for zones whose places you measured. Use `center` or `foot` consistently for the camera perspective; check visually during A. Put a small stable bag **on the aisle floor** to verify its lower box center falls inside the monitored polygon. The starter dwell is two seconds.
- **Preview and freeze.** Extract the cabin still as above, then inspect both overlays. Save the calibrated JSONs with the session record. If you move the camera, record the new placement and make a new calibration before the corresponding scored take.

```sh
python -m inference.preview_layout --source results/monitor/field/door-empty.png \
  --layout configs/mock-door-view.json --door configs/mock-door-line.json \
  --output results/monitor/field/door-overlay.png
python -m inference.preview_layout --source results/monitor/field/cabin-empty.png \
  --layout configs/mock-cabin.json --output results/monitor/field/cabin-overlay.png
```

## 5. Perform the scored scenarios

A practical target is 30–60 seconds per take, with a few seconds of baseline before and after each action. Repeat important cases at least twice if time permits. These are suggested sample sizes, not course requirements. Use unique IDs, for example `record_take B_door_01 45`. Write down what actually happened, even if it diverged from the script.

| Take | Action | Independent reference |
| --- | --- | --- |
| Door baseline | Empty doorway; someone waits inside; approach the line and retreat | Zero crossings |
| Door flow | Single entries and exits, one person leaves and comes back, then two close together | Direction and threshold-crossing time for each person |
| Distribution | Actors gather at the entrance while middle/rear places are open. Pause, then move one person to the rear while total occupancy stays constant | Front/middle/rear counts, visible seat states, designated places remaining at each pause |
| Seats | Sit down, stand up, partly obscure one chair, then put a bag on an empty chair | Occupied, available or uncertain by seat ID; bag-covered seat cannot be called available |
| Passage | Carry a bag through briefly, place it in the aisle for 5 s, remove it, place it outside the aisle, then have a person stand there alone. Repeat at the marked doorway if visible | Region, object placement/removal times, whether a review is expected, and earliest eligible time (placement + dwell) |
| Robustness | Repeat the same sequence with lower illumination, overlap or a second calibrated angle. Change one factor per comparison | Same labels plus condition, camera placement and any blind spots |

For each take note session, take ID, view, light, operator, configuration version, and whether inference was live or replayed. Label the recording before inspecting predictions. At agreed source times, e.g. each scripted pause, record both the actual physical counts and visibility/occlusion. This lets you explain an unseen passenger without silently changing the reference count. When usable space means a genuinely open standing place or seat, have the observer mark it directly too. Capacity minus detected people alone cannot tell whether luggage occupies a place.

## 6. Run live and reproduce the result offline

**Pi, first terminal:** activate the environment in `~/transitsense-trial` and run `python -m dashboard.serve`. **Mac, separate terminal:** run `ssh -N -L 8091:127.0.0.1:8765 "$pi_host"` with the same SSH destination set in step 1. Open [the Pi's forwarded dashboard](http://127.0.0.1:8091/) on the Mac and choose **Live monitor**.

**Pi, second terminal:** activate the same environment and run one view at a time. Use the corresponding camera placement. Keep session A/B take names distinct from these live demo names.

```sh
cd ~/transitsense-trial
. .venv/bin/activate
python -m inference.monitor --oak --seconds 60 --device cpu \
  --layout configs/mock-door-view.json --door configs/mock-door-line.json --gap 0.80 \
  --name live-door-01 --report results/monitor/field/live-door-01.json

python -m inference.monitor --oak --seconds 60 --device cpu \
  --layout configs/mock-cabin.json --weights runs/transitsense/passengers_baseline/weights/best.pt \
  --name live-cabin-01 --report results/monitor/field/live-cabin-01.json
```

**For the scored video**, use `--source` on the exact MP4 your observer reviewed. Repeat the second command with each cabin take and the first with each doorway take. Replace the IDs for each run.

```sh
python -m inference.monitor --source datasets/raw/staged_mock_bus/B_door_01/rgb.mp4 \
  --layout configs/mock-door-view.json --door configs/mock-door-line.json --gap 0.80 \
  --name B_door_01 --report results/monitor/field/B_door_01.json
python -m inference.monitor --source datasets/raw/staged_mock_bus/B_cabin_01/rgb.mp4 \
  --layout configs/mock-cabin.json --weights runs/transitsense/passengers_baseline/weights/best.pt \
  --name B_cabin_01 --report results/monitor/field/B_cabin_01.json
```

Benchmark the **same MP4s and calibrations** on the Mac with the same weights, CPU setting and model versions where possible. Transfer the agreed clips and configs over SSH to the Mac if participant permissions cover that use. Use the Mac project's `.venv/bin/python`. Compare doorway and cabin workloads separately. Keep original reports in host-specific folders. If the Pi drops crossings or has insufficient frame continuity for seat confirmation or bag dwell, use Pi capture with Mac offline inference for the recorded study, or plug the OAK into the Mac with a compatible DepthAI environment for a live fallback. State exactly what each device did. No Pi-to-Mac live video transport is implemented by the existing dashboard tunnel.

## 7. Score and package the course evidence

1. **Flow.** Create `results/monitor/field/door-reference.csv` with header `clip,direction,time_s`. Example row: `B_door_01,boarding,3.24`. Use `alighting` for exits. Use `B_door_quiet,,` for a reviewed empty clip. Run the command below. Report reviewed, matched, missed, extra and absolute count error separately by direction and condition. The default matching tolerance is ±0.5 s.
2. **Zones and seats.** From the report's `samples`, use the nearest sample to each preselected observation time and record the time difference. For each zone average the absolute difference between estimated and observed counts. Report exact seat-state matches, uncertain estimates and false-available calls, including the number of seats assessed. Report unobserved zones separately. For the central claim, show one row before and one after the rider moves rearward.
3. **Obstructions.** Compare alert **episodes** against reference placements. Count missed and extra episodes and measure delay from placement to first review and from removal to clearing. The bundled `score_cabin` and `score_obstruction` commands score the supplied example fixtures. For new room trials, tabulate your own labels against `samples[].zones`, `samples[].seats` and `samples[].passages` in a spreadsheet. Do not reuse the example fixture labels.
4. **Runtime and recovery.** Save frame count, processing FPS, median and p95 processing time, sampled peak memory, camera delivery FPS, latency and skips. Test unplug/replug in a separate unscored run and record the dashboard state and manual restart. The implementation exits on disconnect. To measure end-to-end operator latency, observe a visible action and the corresponding dashboard change on the same external clock (for example a consented recording of both). Inference p95 alone excludes display polling and, for an obstruction, the intentional dwell.
5. **Depth.** Include the depth bench evidence and a comparison during partial overlap if feasible. The monitor uses RGB for inference; the depth probe tests synchronized acquisition. A claim that depth improves counts requires a depth-assisted decision method and an RGB-only vs RGB-plus-depth comparison on matched events. Identify that comparison as outstanding if you only demonstrate alignment.
6. **Final package.** Include the mock-bus floor plan with approximate dimensions, camera poses and blind spots, non-identifying calibration overlays, frozen configs, reference labels, aggregate reports, per-condition results, Pi/Mac benchmarks, and a brief operator demo. Keep consent records and restricted footage separate from the repository. Retain/delete footage according to what participants agreed. Give the presentation one clear failure case as well as the successful example.

```sh
python -m evaluation.evaluate_doorway --clip B_door_01 \
  --reference results/monitor/field/door-reference.csv \
  --report results/monitor/field/B_door_01.json
```
