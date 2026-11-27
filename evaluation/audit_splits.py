"""Flag exact and visually similar cross-split images for manual leakage review."""
from hashlib import sha256
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def fingerprints(folder):
    records = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in (".png",".jpeg",".jpg"):
            continue
        # Reduced JPEG decoding keeps the difference-hash screen inexpensive.
        gray=cv2.imread(str(p),cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if gray is None:
            raise ValueError(f"Unreadable image: {p}")
        thumb=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
        bits=0
        for bit in (thumb[:,1:] > thumb[:,:-1]).flat:
            bits=(bits << 1) | int(bit)
        records.append((p.name.split(".rf.")[0],sha256(p.read_bytes()).digest(),bits))
    return records


def audit(train, evaluation):
    known=fingerprints(train)
    held=fingerprints(evaluation)
    names={r[0] for r in known}
    hashes={r[1] for r in known}
    if not known or not held:
        raise ValueError("Empty evaluation or training split")
    near=sum(min((a[2]^b[2]).bit_count() for b in known) <= 4 for a in held)
    return {"train_images":len(known),"evaluation_images":len(held),
            "shared_export_stem":sum(a[0] in names for a in held),
            "identical_file":sum(a[1] in hashes for a in held),"dhash_within_four_bits":near}


if __name__ == "__main__":
    for dataset,split in [("processed/seat_occupancy_clean","test"),("raw/bus_occupancy","valid")]:
        base=ROOT/"datasets"/dataset
        print(dataset,split,audit(base/"train/images",base/split/"images"))
