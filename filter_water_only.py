#!/usr/bin/env python3
"""Create a YOLO water-only detection export without changing original files.

Rules:
  * Include an image only if it has >= 1 `water` box and 0 `undefined` boxes.
  * Images with both classes are excluded entirely (safe default).
  * Remap `water` to class ID 0; preserve xywh coordinates as supplied.
  * Preserve original train/valid/test splits; never infer missing test partitions.
  * Skip invalid or missing labels, preserve source untouched.
  * This filters by label name, NOT the semantic accuracy of the labels.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_names(raw):
    names = raw["names"]
    if isinstance(names, dict):
        return {int(k): str(v).strip().lower() for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v).strip().lower() for i, v in enumerate(names)}
    raise ValueError("Expected 'names' list or mapping in data.yaml")


def normalized_annotation(txt, water_id, undefined_id):
    boxes = []
    has_undefined = False
    for row in txt.splitlines():
        if not row.strip():
            continue
        parts = row.split()
        if len(parts) != 5:
            return None, "malformed_label"
        try:
            label_id = int(parts[0])
            numbers = [float(value) for value in parts[1:]]
        except (ValueError, OverflowError):
            return None, "malformed_label"
        if any(not (0 <= x <= 1) for x in numbers) or numbers[2] <= 0 or numbers[3] <= 0:
            return None, "invalid_box_coordinates"
        x, y, w, h = numbers
        if x - w / 2 < -0.00001 or x + w / 2 > 1.00001 or y - h / 2 < -0.00001 or y + h / 2 > 1.00001:
            return None, "out_of_bounds_box"
        if label_id == undefined_id:
            has_undefined = True
        elif label_id == water_id:
            boxes.append("0 " + " ".join(parts[1:]))
        else:
            return None, f"unknown_class_id_{label_id}"
    if has_undefined:
        return None, "contains_undefined"  # Exclude whole image, even mixed cases.
    if not boxes:
        return None, "no_water_boxes"
    return boxes, "kept"


def make_previews(output, count):
    if count <= 0:
        return
    try:
        import cv2
    except ImportError:
        print("Review previews skipped (opencv-python not installed).")
        return
    pairs = []
    for split in ("train", "val", "test"):
        folder = output / "images" / split
        if not folder.is_dir():
            continue
        for img in folder.iterdir():
            if img.suffix.lower() in IMAGE_SUFFIXES:
                pairs.append((split, img, output / "labels" / split / (img.stem + ".txt")))
    picks = random.Random(42).sample(pairs, min(count, len(pairs)))
    previews = output / "review_previews"
    previews.mkdir(parents=True, exist_ok=True)
    saved = 0
    for split, img, lbl in picks:
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        for line in lbl.read_text().splitlines():
            fields = line.split()
            if len(fields) != 5: continue
            cx, cy, bw, bh = map(float, fields[1:])
            left = round((cx - bw / 2) * width)
            right = round((cx + bw / 2) * width)
            top = round((cy - bh / 2) * height)
            bottom = round((cy + bh / 2) * height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 0), 2)
            cv2.putText(frame, "water (unverified)", (max(0, left), max(20, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        cv2.imwrite(str(previews / f"{split}_{img.stem}.jpg"), frame)
        saved += 1
    print(f"Created {saved} random review previews in {previews}")


def main():
    parser = argparse.ArgumentParser(description="Export water-only YOLO images and boxes, excluding all images with undefined annotations")
    parser.add_argument("--source", type=Path, default=Path("datasets/raw/water_spills"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/water_only"))
    parser.add_argument("--preview-count", type=int, default=30, help="Random annotated images to generate for visual review (requires opencv-python)")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "data.yaml").exists():
        parser.error(f"Source data.yaml not found: {source / 'data.yaml'}")
    if output == source or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate directories")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite existing nonempty output: {output}")

    with (source / "data.yaml").open() as f:
        cfg = yaml.safe_load(f)
    names = get_names(cfg)
    if sorted(names.values()) != ["undefined", "water"]:
        parser.error(f"Expected only water and undefined labels. Got: {names}")
    water_id = next(k for k, v in names.items() if v == "water")
    undefined_id = next(k for k, v in names.items() if v == "undefined")

    counts = Counter()
    manifest = []
    for target_split, candidates in (("train", ("train",)), ("val", ("valid", "val")), ("test", ("test",))):
        split_root = next((source / s for s in candidates if (source / s / "images").is_dir()), None)
        if split_root is None:
            print(f"No {target_split} split found; skipping")
            continue
        in_imgs = split_root / "images"
        in_labels = split_root / "labels"
        if not in_labels.is_dir():
            raise FileNotFoundError(f"Labels directory missing: {in_labels}")
        dest_imgs = output / "images" / target_split
        dest_labels = output / "labels" / target_split
        dest_imgs.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)

        images = [f for f in in_imgs.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
        for img in sorted(images):
            label = in_labels / (img.stem + ".txt")
            if not label.exists():
                boxes, reason = None, "missing_label_file"
            else:
                boxes, reason = normalized_annotation(label.read_text(), water_id, undefined_id)
            if boxes is not None:
                dest_img = dest_imgs / img.name
                dest_label = dest_labels / (img.stem + ".txt")
                shutil.copy2(img, dest_img)
                dest_label.write_text("\n".join(boxes) + "\n")
                counts[(target_split, "kept_images")] += 1
                counts[(target_split, "kept_boxes")] += len(boxes)
                manifest.append((target_split, str(img.relative_to(source)), "kept", len(boxes)))
            else:
                counts[(target_split, reason)] += 1
                manifest.append((target_split, str(img.relative_to(source)), reason, 0))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "filter_manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "source_image", "status", "water_boxes"])
        writer.writerows(manifest)
    config = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "water"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "README.txt").write_text(
        "TRANSITSENSE WATER-ONLY FILTERED DATASET\n"
        "Source: datasets/raw/water_spills. Original images and labels were not modified.\n"
        "Images with one or more undefined labels were excluded, even if water was also present.\n"
        "Images with no water annotations, missing or invalid labels were excluded.\n"
        "Water boxes were remapped from original class ID to class 0; coordinates preserved.\n"
        "IMPORTANT: This is a mechanical label filter, not verified spill imagery.\n"
        "Earlier samples showed people labeled 'water'; inspect all relevant examples before training.\n"
        "See filter_manifest.csv and review_previews/ for review.\n"
        "Original train/val/test groupings are retained but may contain near-duplicate video frames.\n"
        "Check licenses, correct semantic labels, and create a truly independent test set for final evaluation.\n"
    )
    print("\nWater-only filter complete. Original source unchanged.")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counts[(split, 'kept_images')]} images, {counts[(split, 'kept_boxes')]} boxes")
    print("Excluded:")
    for (split, reason), amount in sorted(counts.items()):
        if reason not in ("kept_images", "kept_boxes"):
            print(f"  {split}: {reason} = {amount}")
    print("Saved to:", output)
    make_previews(output, args.preview_count)
    print("WARNING: Filtering cannot distinguish mislabeled people from real water spills. Visual audit required.")


if __name__ == "__main__":
    main()
