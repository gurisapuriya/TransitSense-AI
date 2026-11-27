"""Create independent, three-second still-image replays for the real cabin fixture."""
import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "evaluation/cabin_reference.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/cabin")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as staging:
        for record in json.loads(REFERENCE.read_text()):
            source = ROOT / record["source"]
            if not source.is_file():
                parser.error(f"Missing original test image: {source}")
            name = record["id"] + ".mp4"
            temp = Path(staging) / name
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-loop", "1",
                "-framerate", "10", "-i", str(source), "-vf", "scale=960:540",
                "-frames:v", "30", "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an", str(temp),
            ], check=True)
            dest = args.output_dir / name
            temp.replace(dest)
            print(f"{name}: 30 repeated frames from {source.name} -> {dest}")


if __name__ == "__main__":
    main()
