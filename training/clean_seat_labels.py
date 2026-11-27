
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/raw/seat_occupancy"
DEST = ROOT / "datasets/processed/seat_occupancy_clean"

names = yaml.safe_load(
    (SOURCE / "data.yaml").read_text()
)["names"]

converted = 0

for source_split, target_split in [
    ("train", "train"),
    ("valid", "val"),
    ("test", "test"),
]:
    src = SOURCE / source_split
    dst = DEST / target_split

    shutil.copytree(
        src / "images",
        dst / "images",
        dirs_exist_ok=True
    )
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    for label in (src / "labels").glob("*.txt"):
        output_lines = []

        for line in label.read_text().splitlines():
            if not line.strip():
                continue

            values = line.split()
            class_id = int(values[0])
            coords = list(map(float, values[1:]))

            if not 0 <= class_id < len(names):
                raise ValueError(f"Invalid class: {label}")

            if any(v < 0 or v > 1 for v in coords):
                raise ValueError(f"Invalid coordinates: {label}")

            if len(values) == 5:
                # Existing bounding box: preserve it
                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError(f"Invalid box: {label}")
                output_lines.append(line)

            elif len(coords) >= 6 and len(coords) % 2 == 0:
                # Polygon vertices -> enclosing bounding box
                xs = coords[0::2]
                ys = coords[1::2]

                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Invalid polygon: {label}")

                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                output_lines.append(
                    f"{class_id} {xc:.6f} {yc:.6f} "
                    f"{w:.6f} {h:.6f}"
                )
                converted += 1

            else:
                raise ValueError(f"Unknown label format: {label}")

        (dst / "labels" / label.name).write_text(
            "\n".join(output_lines) + "\n"
            if output_lines else ""
        )

config = {
    "path": str(DEST.resolve()),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": names
}

(DEST / "data.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False)
)

print("Conversion completed!")
print("Polygon annotations converted:", converted)
print("Clean dataset:", DEST)
