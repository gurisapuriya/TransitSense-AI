"""Read the aggregate files used by the console and its local HTTP API."""
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_STATE = BASE.parent / "results/monitor/live.json"
RUNS = {
    "controlled": "Controlled stop · simulation", "boarding": "Doorway · boarding video",
    "doorway": "Doorway · mixed flow video", "bag": "Aisle · composite bag trial",
    "front_row": "Cabin · February still", "quiet": "Cabin · quiet still",
    "moderate": "Cabin · moderate still", "busy": "Cabin · busy still",
}


def replays(extra=None):
    sources = {key: (label, BASE / "replays" / f"{key}.json") for key, label in RUNS.items()}
    if extra:
        sources["local"] = ("Local recording", extra)
    return sources


def live_state(path):
    if not path.is_file():
        return {"status": "offline", "sample": None}
    state = json.loads(path.read_bytes())
    if state["status"] in ("running", "starting") and time.time() - state["updated"] > 5:
        state.update(status="stale", sample=None)
    return state


def api_data(path, live, sources):
    if path == "/api/runs":
        return json.dumps([{"id": k, "name": v[0]} for k, v in sources.items()]).encode()
    if path == "/api/live":
        return json.dumps(live_state(live), allow_nan=False).encode()
    key = path.removeprefix("/api/runs/")
    if key != path and key in sources:
        return sources[key][1].read_bytes()
    raise LookupError(path)
