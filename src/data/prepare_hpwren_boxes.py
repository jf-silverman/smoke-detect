"""Prepare the AI-For-Mankind HPWREN box dataset for YOLO training (Phase C).

Converts the Pascal-VOC "day_time_wildfire_v2" set (2191 HPWREN images with smoke
bounding boxes, CC BY-NC-SA 4.0, aiformankind/wildfire-smoke-dataset) into YOLO format
under data/hpwren_boxes/, and writes hpwren_boxes.yaml.

LEAK-SAFETY (the point of this script). Phase C trains here and is EVALUATED on the 24
held-out FIgLib TTD fires. Those two sets both derive from HPWREN, so the same frame can
appear in both -- which would leak the test set into training. The AI-For-Mankind files are
hash-renamed (the original fire/camera id is gone), so we cannot dedup by name. Instead we
dedup by CONTENT: perceptual-hash (pHash, resize-invariant) every AI-For-Mankind image and
every FIgLib TTD frame, and drop any training image whose nearest FIgLib frame is within
--phash-thresh Hamming distance. Dropped ids are logged so the exclusion is auditable.

    python src/data/prepare_hpwren_boxes.py            # convert + dedup + split + yaml
    python src/data/prepare_hpwren_boxes.py --phash-thresh 6 --val-frac 0.1

No GPU. Fast (pHash over ~4300 images is seconds).
"""

from __future__ import annotations

import argparse
import glob
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "hpwren_boxes" / "day_time_wildfire_v2"
OUT = ROOT / "data" / "hpwren_boxes"
FIGLIB = ROOT / "data" / "figlib" / "images"


def phash_set(paths: list[Path]) -> dict[Path, imagehash.ImageHash]:
    out = {}
    for p in paths:
        try:
            out[p] = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception as e:  # noqa: BLE001 -- skip unreadable frames, report count
            print(f"  skip unreadable {p.name}: {e}")
    return out


def voc_to_yolo(xml_path: Path) -> list[tuple[float, float, float, float]]:
    """Return YOLO-normalized (cx, cy, w, h) boxes for the single 'smoke' class."""
    root = ET.parse(xml_path).getroot()
    W = float(root.findtext("size/width"))
    H = float(root.findtext("size/height"))
    boxes = []
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip().lower() != "smoke":
            continue
        b = obj.find("bndbox")
        xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
        xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
        # clip to frame, guard against annotation overflow (some xmax == width+eps)
        xmin, xmax = max(0.0, xmin), min(W, xmax)
        ymin, ymax = max(0.0, ymin), min(H, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append(((xmin + xmax) / 2 / W, (ymin + ymax) / 2 / H,
                      (xmax - xmin) / W, (ymax - ymin) / H))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phash-thresh", type=int, default=6,
                    help="drop a training image if its nearest FIgLib frame is within this "
                         "Hamming distance (0-64). Lower = stricter match; 6 is conservative.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"missing {SRC} -- extract day_time_wildfire_v2.tar.gz first")

    imgs = sorted((SRC / "images").glob("*.jpeg"))
    print(f"AI-For-Mankind images: {len(imgs)}")

    # 1) pHash the FIgLib TTD test frames (the set we must NOT leak).
    figlib_frames = [Path(p) for p in glob.glob(str(FIGLIB / "*" / "*.jpg"))]
    print(f"FIgLib TTD frames to protect: {len(figlib_frames)}")
    fig_hashes = list(phash_set(figlib_frames).values())

    # 2) pHash the training images, find nearest-FIgLib distance for each.
    train_hashes = phash_set(imgs)
    fig_arr = np.array([h.hash.flatten() for h in fig_hashes], dtype=bool)  # (Nf, 64)

    kept, dropped = [], []
    min_dists = []
    for p, h in train_hashes.items():
        hv = h.hash.flatten()[None, :]                       # (1, 64)
        d = (fig_arr != hv).sum(axis=1).min() if len(fig_arr) else 64
        min_dists.append(int(d))
        (dropped if d <= args.phash_thresh else kept).append(p)

    md = np.array(min_dists)
    print(f"\nnearest-FIgLib Hamming distance over training imgs: "
          f"min={md.min()} p5={np.percentile(md,5):.0f} median={np.median(md):.0f}")
    print(f"LEAK-GUARD (thresh {args.phash_thresh}): kept {len(kept)}, "
          f"dropped {len(dropped)} near-duplicates of FIgLib test frames")

    # log the dropped ids so the exclusion is auditable
    (OUT / "leak_dropped.txt").write_text("\n".join(sorted(p.name for p in dropped)))

    # 3) split + write YOLO images/labels.
    rng = np.random.default_rng(args.seed)
    kept = sorted(kept)
    rng.shuffle(kept)
    n_val = max(1, round(len(kept) * args.val_frac))
    split = {p: ("val" if i < n_val else "train") for i, p in enumerate(kept)}

    n_boxes = {"train": 0, "val": 0}
    for sub in ("train", "val"):
        (OUT / "images" / sub).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sub).mkdir(parents=True, exist_ok=True)
    for p, sub in split.items():
        boxes = voc_to_yolo(SRC / "annotations" / "xmls" / (p.stem + ".xml"))
        if not boxes:
            continue
        shutil.copy2(p, OUT / "images" / sub / p.name)
        lbl = OUT / "labels" / sub / (p.stem + ".txt")
        lbl.write_text("\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes))
        n_boxes[sub] += len(boxes)

    n_tr = len(list((OUT / "images" / "train").glob("*.jpeg")))
    n_va = len(list((OUT / "images" / "val").glob("*.jpeg")))

    # 4) YOLO data yaml. Absolute path is rewritten on the VM by the startup script (same as
    #    the pyro-sdis splits), so keep the repo-local absolute path here.
    yaml = OUT / "hpwren_boxes.yaml"
    yaml.write_text(
        f"# AI-For-Mankind HPWREN smoke boxes (CC BY-NC-SA 4.0). Phase C training set.\n"
        f"# Leak-guarded against the 24 FIgLib TTD test fires by pHash (thresh "
        f"{args.phash_thresh}); {len(dropped)} near-dupes dropped (see leak_dropped.txt).\n"
        f"path: {OUT}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: [smoke]\n"
    )
    print(f"\nwrote {n_tr} train / {n_va} val images "
          f"({n_boxes['train']}/{n_boxes['val']} boxes) -> {OUT}")
    print(f"yaml -> {yaml}")


if __name__ == "__main__":
    main()
