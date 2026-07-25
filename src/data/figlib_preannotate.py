"""Generate CVAT pre-annotations for Phase C pseudo-labeling of local FIgLib fires.

Phase C needs an in-distribution smoke detector, which needs bounding boxes. Canonical
FIgLib ships only onset labels (the filename offset), not boxes. Rather than draw every box
from scratch, we let the existing pyro-sdis detector do a first pass and then hand-correct in
CVAT -- semi-supervised pseudo-labeling. This script produces the machine's first pass.

WHAT IT DOES
  * Enumerates FIgLib frames (reusing figlib_temporal.scan_frames), splits fires into a
    TRAIN set (pseudo-label these) and a held-out EVAL set (the recent 2024-2025 CA fires we
    must NOT train on -- Palisades/Coches/Tenaja/etc.). The eval fires are excluded here by
    whole-fire holdout so the machine never touches them.
  * ONSET GATE: only positive frames (offset >= 0, i.e. at/after ignition) are pre-annotated,
    so the detector is never asked to invent a box on a pre-ignition frame. This uses the free
    onset labels as a guardrail against the detector's own false alarms.
  * Runs the detector on native-resolution TILES (the resolution it was trained at; whole-frame
    downscaling pooled the small plumes away -- see figlib_tiled.py), maps each tile's boxes back
    to full-frame coordinates, merges seam duplicates with NMS, and keeps boxes above --conf.
  * Writes an "Ultralytics YOLO Detection" bundle -- the format CVAT imports directly and that
    `yolo train` consumes with no conversion (chosen over legacy "YOLO 1.1" for exactly that
    round-trip). Frames are copied under flattened <fire>__<stem>.jpg names so CVAT can match
    annotations to images by filename across fires.

    python src/data/figlib_preannotate.py                 # defaults: conf 0.15, 30 frames/fire
    python src/data/figlib_preannotate.py --conf 0.10 --max-per-fire 40

Then in CVAT: create a task from data/figlib_preannot/images/train, Upload annotations ->
"Ultralytics YOLO Detection" -> the produced zip, and correct. No GPU, no Nuclio, no Docker.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
FIGLIB = ROOT / "data" / "figlib"
OUT = ROOT / "data" / "figlib_preannot"
WEIGHTS = ROOT / "runs" / "gcp_grouped_1280" / "weights" / "best.pt"
sys.path.insert(0, str(ROOT / "src"))
from models.figlib_temporal import scan_frames  # noqa: E402

# The 6 recent (2024-2025) CA fires: held out for eval, never pseudo-labeled. These are exactly
# the fires the zero-shot detector missed in the TTD study -- the target Phase C must beat. They
# share no tower with the 19 training fires, so the holdout is leak-clean.
EVAL_FIRES = {
    "20240708_VistaFire_wilson-e-mobo-c",
    "20240825_TenajaFire_buff-n-mobo-c",
    "20241009_BahrmanFire_hp-n-mobo-c",
    "20250107_PalisadesFire_69bravo-e-mobo-c",
    "20250612_HighwayFire_hp-e-mobo-c",
    "20250908_CochesFire_sm-n-mobo-c",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tiles_with_offset(W: int, H: int, tile: int, stride: int):
    """Yield (x, y) top-left corners of native-res tiles covering the frame (edge-aligned last)."""
    xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
    ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))
    for y in ys:
        for x in xs:
            yield x, y


def frame_boxes(model, img: Image.Image, tile: int, stride: int, conf: float,
                device: str, batch: int) -> np.ndarray:
    """Return full-frame normalized (cx, cy, w, h) boxes for one frame, NMS-merged across tiles."""
    W, H = img.size
    corners = list(tiles_with_offset(W, H, tile, stride))
    crops = [img.crop((x, y, min(x + tile, W), min(y + tile, H))) for x, y in corners]
    xyxy, scores = [], []
    for j in range(0, len(crops), batch):
        chunk = crops[j:j + batch]
        res = model.predict(chunk, device=device, imgsz=tile, conf=conf, verbose=False)
        for (x, y), r in zip(corners[j:j + batch], res):
            if not len(r.boxes):
                continue
            b = r.boxes.xyxy.cpu().numpy()          # tile-local pixels
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y                        # -> full-frame pixels
            xyxy.append(b)
            scores.append(r.boxes.conf.cpu().numpy())
    if not xyxy:
        return np.empty((0, 4), dtype=np.float32)
    xyxy = np.concatenate(xyxy).astype(np.float32)
    scores = np.concatenate(scores).astype(np.float32)
    keep = nms(torch.from_numpy(xyxy), torch.from_numpy(scores), iou_threshold=0.5).numpy()
    xyxy = xyxy[keep]
    # -> normalized cx, cy, w, h, clipped to frame
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    x1, x2 = np.clip([x1, x2], 0, W)
    y1, y2 = np.clip([y1, y2], 0, H)
    return np.stack([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15,
                    help="keep pre-annotation boxes above this confidence (default 0.15)")
    ap.add_argument("--max-per-fire", type=int, default=30,
                    help="evenly sample at most this many positive frames per fire (default 30)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if not WEIGHTS.exists():
        raise SystemExit(f"missing detector weights at {WEIGHTS}")

    device = pick_device()
    model = YOLO(str(WEIGHTS))

    df = scan_frames()
    df = df[df["smoke"]]                              # ONSET GATE: positive (offset>=0) frames only
    df = df[~df["seq"].isin(EVAL_FIRES)]             # whole-fire holdout of the recent eval fires
    train_fires = sorted(df["seq"].unique())
    print(f"pseudo-labeling {len(train_fires)} train fires on {device} "
          f"(holding out {len(EVAL_FIRES)} recent eval fires)")

    # even-sample up to max-per-fire positive frames per fire (spread across the onset window)
    picked = []
    for seq, g in df.groupby("seq"):
        g = g.sort_values("offset")
        if len(g) > args.max_per_fire:
            idx = np.linspace(0, len(g) - 1, args.max_per_fire).round().astype(int)
            g = g.iloc[idx]
        picked.append(g)
    df = pd.concat(picked).reset_index(drop=True)

    img_dir = OUT / "images" / "train"
    lbl_dir = OUT / "labels" / "train"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rel_paths, n_boxes, n_empty = [], 0, 0
    for i, row in df.iterrows():
        name = f"{row['seq']}__{row['stem']}"        # flattened, unique across fires
        img = Image.open(row["path"]).convert("RGB")
        boxes = frame_boxes(model, img, args.tile, args.stride, args.conf, device, args.batch)
        shutil.copy2(row["path"], img_dir / f"{name}.jpg")
        (lbl_dir / f"{name}.txt").write_text(
            "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes))
        rel_paths.append(f"images/train/{name}.jpg")
        n_boxes += len(boxes)
        n_empty += len(boxes) == 0
        if i % 50 == 0:
            print(f"  {i}/{len(df)} frames  ({n_boxes} boxes so far)", flush=True)

    (OUT / "train.txt").write_text("\n".join(rel_paths))
    (OUT / "data.yaml").write_text(
        "# FIgLib pseudo-label pre-annotations (Phase C). Correct in CVAT, then train.\n"
        f"path: {OUT}\ntrain: images/train\nval: images/train\nnames:\n  0: smoke\n")

    # zip the bundle for a one-click CVAT 'Upload annotations' (Ultralytics YOLO Detection)
    zpath = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("data.yaml", "train.txt"):
            z.write(OUT / f, f)
        for sub in ("labels/train", "images/train"):
            for p in sorted((OUT / sub).glob("*")):
                z.write(p, f"{sub}/{p.name}")

    print(f"\n{len(df)} frames pre-annotated: {n_boxes} boxes, "
          f"{n_empty} frames with no detection (draw those from scratch).")
    print(f"bundle -> {OUT}\nzip (import into CVAT) -> {zpath}")


if __name__ == "__main__":
    main()
