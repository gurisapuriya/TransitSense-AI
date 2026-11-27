# TransitSense AI

TransitSense is a spatial-analysis course prototype for a transit operator. The intended display answers three questions during a stop or trip. How many people entered and left? Where are visible seats or usable standing space? Has an object blocked a doorway or aisle long enough to warrant a look? The design uses temporary tracks and aggregate measurements. Seats, zones and obstructions each need their own evaluation. Doorway flow alone cannot establish onboard occupancy without a starting count and coverage of every active door.

## Open the operator console

On the project Mac, start the Streamlit dashboard from the repository root using the installed Python 3.13, then open [the operator console](http://127.0.0.1:8765/).

```sh
python3.13 -m dashboard.serve
```

On another host, install the dashboard packages in a dedicated environment. The inference process can keep using its own Python environment and write to the same aggregate state file.

```sh
python3.13 -m venv .venv-dashboard
.venv-dashboard/bin/python -m pip install -r dashboard/requirements.txt
.venv-dashboard/bin/python -m dashboard.serve
```

For a lightweight Pi installation, `python3 -m dashboard.serve --basic` serves the original console with the Python standard library. The launcher also uses that console when Streamlit is absent.

A [console screenshot](docs/operator-console.jpg) shows the controlled-stop replay. Select **Doorway · boarding video** for measurements from actual PCDS footage, **Cabin · busy still** for estimated distribution and mapped seats, or **Aisle · composite bag trial** for a detector-driven obstruction example. The **Controlled stop** replay shows every feature on a labelled synthetic detection sequence. Play, pause or scrub the source clock. The live option polls local aggregate measurements. Camera frames and passenger track IDs stay outside the operator API.

**Measured results.** The doorway counter matched 8 of 11 events on additional labelled footage, including 1 of 3 in the crowded queue, with zero extra events. The cabin pipeline matched 18 of 24 seat states on three calibration stills and 4 of 4 front-seat states in a separate image. The full detector pipeline flagged the sustained bag placement in the composite aisle trial. These experiments and their limitations are documented in the [evaluation report](docs/evaluation.md).

## Process a recording

Use the local Python environment with the dependencies in `requirements.txt`. Video inference needs OpenCV, Ultralytics and the local weights. The camera adapter adds an optional DepthAI 3.x dependency. Source data setup and provenance are in [the dataset guide](datasets/README.md) and [source registry](datasets/sources.csv).

```sh
.venv/bin/python -m inference.monitor \
  --source results/converted_videos/sample_0006.mp4 \
  --layout configs/doorway_view.json --door configs/pcds_sample_0006_door.json \
  --gap 0.80 --name "Doorway 06" --report results/monitor/doorway-run.json
```

Select **Live monitor** during the run. A file is processed as quickly as the host allows. Its source timestamps control tracking and dwell measurements. At the end, the console labels the final observation. A disconnected or stale producer loses its live status. To load the full timeline later, start the dashboard with `--replay results/monitor/doorway-run.json` and select **Local recording**.

For a cabin trial, create the local image-based fixtures and run all three detectors. The video in this example contains one real still repeated for three seconds.

```sh
.venv/bin/python -m evaluation.cabin_replays
.venv/bin/python -m inference.monitor \
  --source results/cabin/busy.mp4 --layout configs/cabin_real.json \
  --weights runs/transitsense/passengers_baseline/weights/best.pt \
  --name "Busy cabin still" --report results/monitor/cabin-run.json
```

The general weights are `yolo11n.pt`; they supply person and luggage classes. For cabin runs, the monitored passenger model above replaces the general person detector, the general model supplies luggage detections, and `runs/transitsense/seats_clean_v1/weights/best.pt` supplies seat detections. For doorway-only views, one model is sufficient. The legacy two-model annotated-video demo remains available in `inference/combined_inference.py`.

## Calibrate a new fixed view

Start from the closest layout in `configs/`. On a representative frame, trace physical front, middle and rear regions and the visible aisle and door. Normalize x by image width and y by image height, with the origin at top left. Give each visible physical seat a stable ID and rectangle. Use `null` for an unseen region. Pick a consistent person anchor for the perspective. Mark partially visible regions in `limited_zones` and seats with a permanently obscured sitting surface in `availability_occluded`. A measured zone capacity is optional. The simulated capacity is illustrative.

Use a second door calibration file for direction. Its two endpoints bound the actual threshold, and its third point is on the bus interior side. Review the overlay against the source before recording results. A moved camera, even at the same resolution, needs another calibration.

```sh
.venv/bin/python -m inference.preview_layout --source results/cabin/quiet.mp4 \
  --layout configs/cabin_real.json --output results/monitor/cabin-calibration.png
```

The monitor rejects a mismatched aspect ratio. It withholds spatial estimates in dark, overexposed or low-detail frames. Fixed-view occlusion and weak observations can yield uncertain zones or seats. Bag-like objects trigger a review only after remaining in the configured passage for its dwell interval. Unknown objects, inference errors and changes to the camera pose still require human review.

## Edge configuration

The proposed OAK-D Pro supplies RGB to the connected inference host. An optional bench mode inspects aligned depth. The local service can deliver aggregate results to a Raspberry Pi browser through an SSH tunnel. The Mac CPU benchmarks and the physical Pi/OAK validation procedure are in the [edge guide](docs/edge.md). OAK capture and Pi inference are pending a hardware trial.

## Doorway flow

The tracker uses a local YOLO detector and ByteTrack. For each camera view, a JSON file specifies two points on the doorway and a point inside the bus, in image coordinates normalized from 0 to 1. The anchor for these overhead clips is the center of the detected box. Moving across the finite segment toward the interior counts as boarding. Moving away counts as alighting.

The calibration files use a margin of 3.5% of image height on each side. A track must establish its side with two observations beyond the margin and then establish the opposite side. A confirmed return crossing counts separately. A new ID or a gap over 0.24 seconds starts fresh. The first person in a clip can already be partway through a crossing, leaving too little history to count them. A new camera angle, crop or camera position requires a new calibration.

### Reproduce a run

Run from the repository root using the project virtual environment. The example below uses the general person detector and the darker PCDS sample. Use the corresponding clip ID for samples 0001 and 0002. The original AVIs are local research data. Convert them with FFmpeg once. The initial conversion can be skipped when the MP4 already exists.

```sh
clip=sample_0004
mkdir -p results/converted_videos
ffmpeg -n -i "datasets/raw/people_counting/sample/videos/${clip}.avi" \
  -c:v libx264 -pix_fmt yuv420p -an "results/converted_videos/${clip}.mp4"

.venv/bin/python inference/track_passengers.py \
  --source "results/converted_videos/${clip}.mp4" \
  --calibration "configs/pcds_${clip}_door.json" \
  --weights yolo11n.pt --conf 0.10 \
  --output "results/doorway/${clip}_person_low.mp4" \
  --debug-csv "results/doorway/${clip}_person_low.csv"

.venv/bin/python evaluation/evaluate_doorway.py \
  --clip "$clip" --tracks "results/doorway/${clip}_person_low.csv"
.venv/bin/python -m unittest discover -s tests -v
```

Omit `--weights` and `--conf` to run the fine-tuned passenger detector with the original 0.35 threshold. Use distinct output names for comparison. The checkpoint is expected at `runs/transitsense/passengers_baseline/weights/best.pt`. Omitting `--calibration` gives a tracking-only run. Omitting `--debug-csv` skips writing individual track coordinates.

The video footer shows visible tracks, IDs seen over the clip and, when calibrated, the two flow counts. Each CSV row records a zero-based frame and time for a constant-frame-rate video. Tracked boxes include confidence, pixel coordinates and the anchor. An empty ID marks a frame with no active track. A confirmed event records its direction and the estimated line-crossing time. Confirmation can occur several frames later.

### What the sample runs showed

The reference CSV contains approximate times from visual review of each person at the doorway. The video's embedded cumulative counters provide a second check on totals. The evaluator matches predictions once each, in the same direction, within 0.5 seconds of a reference time. The runtime counter uses only detector tracks and the geometry file. Sample 0004 was checked after selecting the general detector and confidence on the first two clips. All three are small PCDS examples, with one reviewer and a separately placed line per view.

| Clip | Reviewed board / alight | Passenger detector, 0.35: matched / extra | General person detector, 0.10: matched / extra |
| --- | --- | --- | --- |
| sample_0001 | 2 / 3 | 0 / 1 | 1 / 0 |
| sample_0002 | 1 / 1 | 1 / 1 | 1 / 0 |
| sample_0004 | 0 / 3 | 2 / 0 | 2 / 0 |

With the general detector at 0.10, four of seven exits and zero of three entries matched. On sample 0002 the original passenger detector returned the correct totals in both directions, yet its boarding event occurred around 0.29 s and the actual boarding was near 7.24 s. Scoring the timestamps exposes that error. On sample 0001 the general detector at 0.35 missed all five events. At 0.10 its track across the doorway at 1.31 s recovered one exit. Around 3.3 s the track ID changes as another person moves across the threshold, leaving the counter without a continuous trajectory.

Later experiments, the fresh doorway comparison, cabin and obstruction metrics, and split-audit results are in the [evaluation report](docs/evaluation.md).
