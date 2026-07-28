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
  * VIGNETTE FILTER: drops the phantom boxes the detector fires on the black rounded fisheye
    corners -- only where a box center is in a corner AND its region is near-black, so a real plume
    near a (bright) edge is kept. It filters the scaffold you correct, not the final labels; you are
    the label authority in CVAT, so this only cuts deletion clicks, it does not bias ground truth.
  * Writes an "Ultralytics YOLO Detection" bundle -- the format CVAT imports directly and that
    `yolo train` consumes with no conversion (chosen over legacy "YOLO 1.1" for exactly that
    round-trip). Frames are copied under flattened <fire>__<stem>.jpg names so CVAT can match
    annotations to images by filename across fires.

    python src/data/figlib_preannotate.py                 # defaults: conf 0.20, 30 frames/fire
    python src/data/figlib_preannotate.py --conf 0.15 --max-per-fire 40

Then in CVAT: create a task from data/figlib_preannot/images/train, Upload annotations ->
"Ultralytics YOLO Detection" -> the produced *_labels.zip (labels only, no images -- packing the
frames makes the browser upload stall before the import ever registers), append, and correct.
No GPU, no Nuclio, no Docker.
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

# NIGHT EXCLUSION: optical smoke detection collapses to flame/glow detection at night (plumes are
# not self-luminous), so the whole field -- and the canonical FIgLib benchmark -- is day-only. We
# match that scope: any sequence whose name marks it nocturnal is dropped from the Phase C bundle.
# Name-based because FIgLib tags these in the folder name (e.g. 20201202_BondFire-nightime_...).
NIGHT_PATTERN = r"night|nocturnal"


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


def drop_contained(xyxy: np.ndarray, thresh: float) -> np.ndarray:
    """Return a boolean keep-mask that drops any box almost entirely inside a LARGER box.

    NMS (iou 0.5) does not remove a small box nested in a big one: their IoU is
    intersection/union = small_area/big_area, which is well below 0.5 when the inner box is much
    smaller, so both survive as a nested pair (the same plume detected at two scales across tiles).
    Here a box is dropped when >= `thresh` of ITS OWN area lies inside some strictly larger box, so
    only the outer box that bounds the whole plume is kept."""
    n = len(xyxy)
    if n < 2:
        return np.ones(n, dtype=bool)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        ix1 = np.maximum(x1[i], x1); iy1 = np.maximum(y1[i], y1)
        ix2 = np.minimum(x2[i], x2); iy2 = np.minimum(y2[i], y2)
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        frac_i = inter / max(area[i], 1e-6)          # fraction of box i covered by each other box
        contained = (frac_i >= thresh) & (area > area[i])   # strictly larger container -> no ties
        if contained.any():
            keep[i] = False
    return keep


def frame_boxes(model, img: Image.Image, tile: int, stride: int, conf: float,
                device: str, batch: int, vr: float, vluma: float, contain: float) -> np.ndarray:
    """Return full-frame normalized (cx, cy, w, h) boxes for one frame, NMS-merged across tiles.

    Boxes in the dark fisheye vignette corners are dropped (VIGNETTE FILTER): a box is discarded
    only if BOTH its center is in a corner (elliptical radius > vr; edge midpoints are ~1.0, true
    corners ~1.4) AND its pixel region is near-black (mean luma < vluma). Requiring both keeps a
    real plume that happens to sit near an edge (which is bright and not in a corner) while removing
    the phantom boxes on the black rounded corners where there is no scene.

    CONTAINMENT FILTER (`contain`): after NMS, a box nested >= `contain` inside a larger box is
    dropped -- the nested duplicates NMS's IoU rule cannot see (see drop_contained)."""
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
    xyxy = xyxy[drop_contained(xyxy, contain)]      # remove nested duplicates NMS's IoU misses
    # clip to frame (pixels)
    x1 = np.clip(xyxy[:, 0], 0, W); x2 = np.clip(xyxy[:, 2], 0, W)
    y1 = np.clip(xyxy[:, 1], 0, H); y2 = np.clip(xyxy[:, 3], 0, H)
    cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W; bh = (y2 - y1) / H
    # VIGNETTE FILTER: drop a box only if its center is in a corner AND its region is near-black.
    gray = np.asarray(img.convert("L"))
    r = np.sqrt(((cx - 0.5) / 0.5) ** 2 + ((cy - 0.5) / 0.5) ** 2)  # ~1.0 at edge mids, ~1.4 corners
    luma = np.array([gray[int(y1[i]):max(int(y1[i]) + 1, int(y2[i])),
                          int(x1[i]):max(int(x1[i]) + 1, int(x2[i]))].mean()
                     for i in range(len(cx))]) if len(cx) else np.zeros(0)
    interior = ~((r > vr) & (luma < vluma))
    return np.stack([cx, cy, bw, bh], axis=1)[interior]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20,
                    help="keep pre-annotation boxes above this confidence (default 0.20)")
    ap.add_argument("--max-per-fire", type=int, default=30,
                    help="evenly sample at most this many positive frames per fire (default 30)")
    ap.add_argument("--vignette-r", type=float, default=1.1,
                    help="corner cutoff for the vignette filter: elliptical radius of a box center "
                         "above which it counts as a corner (edge mids ~1.0, corners ~1.4)")
    ap.add_argument("--vignette-luma", type=float, default=35.0,
                    help="a corner box is dropped only if its region mean luma (0-255) is below this")
    ap.add_argument("--contain", type=float, default=0.9,
                    help="drop a box when this fraction of its area is nested inside a larger box "
                         "(default 0.9); 1.0 disables all but exact containment")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if not WEIGHTS.exists():
        raise SystemExit(f"missing detector weights at {WEIGHTS}")

    device = pick_device()
    model = YOLO(str(WEIGHTS))

    df = scan_frames()
    df = df[df["smoke"]]                              # ONSET GATE: positive (offset>=0) frames only
    df = df[~df["seq"].isin(EVAL_FIRES)]             # whole-fire holdout of the recent eval fires
    n_before = df["seq"].nunique()
    df = df[~df["seq"].str.contains(NIGHT_PATTERN, case=False, regex=True)]  # day-only scope
    n_night = n_before - df["seq"].nunique()
    if n_night:
        print(f"NIGHT EXCLUSION: dropped {n_night} nocturnal fire(s) (day-only scope)")
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
        boxes = frame_boxes(model, img, args.tile, args.stride, args.conf, device, args.batch,
                            args.vignette_r, args.vignette_luma, args.contain)
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

    # zip an ANNOTATIONS-ONLY bundle for CVAT 'Upload annotations' -> "Ultralytics YOLO Detection 1.0".
    # Images are deliberately excluded: CVAT already holds them in the task and binds boxes to frames by
    # filename, and packing the frames (~380 MB) makes the browser upload stall before it ever queues an
    # import request -- the import silently never registers.
    #   The manifest MUST match CVAT's spec exactly (docs.cvat.ai/.../format-yolo-ultralytics): data.yaml
    # sets `train: train.txt` -- a LIST FILE, not a folder -- and train.txt lists images/train/<name>.jpg.
    # An earlier `train: images/train` (a folder that a labels-only zip does not contain) made CVAT throw
    # "Failed to import dataset". Only the boxed frames are listed and packed; empty frames add nothing on
    # an append import.
    zpath = OUT.parent / f"{OUT.name}_labels.zip"
    boxed = sorted(p for p in lbl_dir.glob("*.txt") if p.stat().st_size > 0)
    portable_yaml = "path: ./\ntrain: train.txt\nnames:\n  0: smoke\n"
    train_list = "".join(f"images/train/{p.stem}.jpg\n" for p in boxed)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.yaml", portable_yaml)
        z.writestr("train.txt", train_list)
        for p in boxed:
            z.write(p, f"labels/train/{p.name}")

    print(f"\n{len(df)} frames pre-annotated: {n_boxes} boxes, "
          f"{n_empty} frames with no detection (draw those from scratch).")
    print(f"bundle (images+labels, for `yolo train`) -> {OUT}")
    print(f"labels-only zip (import into CVAT 'Upload annotations') -> {zpath}")


if __name__ == "__main__":
    main()
