"""Tiled-inference probe: is FIgLib's weak signal a RESOLUTION artifact?

The whole-frame pipeline resized 3072x2048 FIgLib frames to 640 px, which pools a
small onset plume down to a few pixels -- the single-frame detector scored AUC 0.454
(worse than random). This asks whether that is a downscaling artifact rather than pure
distribution shift: run the SAME detector on native-resolution 640-px TILES (no downscaling,
the resolution the detector was trained at), take each frame's max confidence over its
tiles, and recompute AUC.

  * AUC jumps well above 0.5  -> resolution was a real culprit; tiled inference is the
    path to a valid positive control.
  * AUC stays ~0.5            -> distribution shift dominates; needs in-distribution training.

    python src/models/figlib_tiled.py --tile 640 --stride 640

Writes data/figlib/features_tiled.npz (frame max-conf) and prints AUC vs whole-frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
FIGLIB = ROOT / "data" / "figlib"
sys.path.insert(0, str(ROOT / "src"))
from models.figlib_temporal import scan_frames, split_by_fire  # noqa: E402
from models.figlib_ttd import EVAL_FIRES, NIGHT_PATTERN  # noqa: E402  (held-out fires; night filter)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tiles_of(img: Image.Image, tile: int, stride: int):
    """Yield native-resolution crops covering the frame (edge-aligned last tile)."""
    W, H = img.size
    xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
    ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    for y in ys:
        for x in xs:
            yield img.crop((x, y, min(x + tile, W), min(y + tile, H)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--weights", default="runs/grouped_proof/weights/best.pt",
                    help="detector weights to score with (repo-relative or absolute)")
    ap.add_argument("--out", default="features_tiled.npz",
                    help="output npz name under data/figlib/")
    ap.add_argument("--eval-only", action="store_true",
                    help="score ONLY the 6 held-out EVAL_FIRES (the Phase C held-out eval set) -- "
                         "the inverse of the training-set filter, for before/after model comparison")
    ap.add_argument("--exclude-eval", action="store_true",
                    help="score the day-only TRAINING fires (drop the 6 EVAL_FIRES and any night "
                         "fire) -- the fusion training set")
    args = ap.parse_args()
    if args.eval_only and args.exclude_eval:
        raise SystemExit("--eval-only and --exclude-eval are mutually exclusive")

    device = pick_device()
    wpath = Path(args.weights)
    model = YOLO(str(wpath if wpath.is_absolute() else ROOT / wpath))
    df = scan_frames()
    if args.eval_only:
        df = df[df["seq"].isin(EVAL_FIRES)].reset_index(drop=True)
        print(f"EVAL-ONLY: scoring the {df['seq'].nunique()} held-out eval fires")
    elif args.exclude_eval:
        df = df[~df["seq"].isin(EVAL_FIRES)
                & ~df["seq"].str.contains(NIGHT_PATTERN, case=False, regex=True)].reset_index(drop=True)
        print(f"TRAIN fires (day-only, eval held out): {df['seq'].nunique()} fires")
    print(f"tiled inference on {len(df)} frames, tile={args.tile} stride={args.stride}, {device}")

    max_conf = np.zeros(len(df), dtype=np.float32)
    for i, path in enumerate(df["path"]):
        img = Image.open(path).convert("RGB")
        crops = list(tiles_of(img, args.tile, args.stride))
        best = 0.0
        for j in range(0, len(crops), args.batch):
            res = model.predict(crops[j : j + args.batch], device=device,
                                imgsz=args.tile, conf=0.001, verbose=False)
            for r in res:
                c = r.boxes.conf
                if len(c):
                    best = max(best, float(c.max()))
        max_conf[i] = best
        if i % 100 == 0:
            print(f"  {i}/{len(df)}  ({len(crops)} tiles/frame)", flush=True)

    out_path = FIGLIB / args.out
    np.savez_compressed(out_path, stems=df["stem"].to_numpy(), conf_tiled=max_conf)
    print(f"\nwrote {out_path}")

    if args.eval_only or args.exclude_eval:
        # No train/test split on a fire subset; the fusion / TTD harness consume this next.
        import pandas as pd
        df["conf_tiled"] = max_conf
        df["bucket"] = pd.cut(df["offset"], [-99999, -1, 600, 1800, 99999],
                              labels=["pre-ignition", "0-10min", "10-30min", ">30min"])
        print("\nmean max-conf by plume stage (eval fires):")
        print(df.groupby("bucket", observed=True)["conf_tiled"].agg(["mean", "size"]).round(3).to_string())
        return

    # AUC on the same per-fire test split as the whole-frame run
    from sklearn.metrics import roc_auc_score
    df = split_by_fire(df, seed=0)
    df["conf_tiled"] = max_conf
    te = df["split"] == "test"
    y = df.loc[te, "smoke"].astype(int).to_numpy()
    print(f"\nTILED single-frame AUC on FIgLib test: "
          f"{roc_auc_score(y, df.loc[te, 'conf_tiled']):.3f}")
    print("  (whole-frame @640 was 0.454)")
    # mean conf by plume development -- does high-res recover the onset signal?
    import pandas as pd
    df["bucket"] = pd.cut(df["offset"], [-99999, -1, 600, 1800, 99999],
                          labels=["pre-ignition", "0-10min", "10-30min", ">30min"])
    print("\nmean TILED max-conf by plume stage:")
    print(df.groupby("bucket", observed=True)["conf_tiled"].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
