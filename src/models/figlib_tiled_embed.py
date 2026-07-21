"""Per-frame TILED embeddings for the learned-GRU positive control.

The tiled probe (figlib_tiled.py) recovered a usable single-frame signal (AUC 0.658)
but only produced a scalar max-confidence per frame -- enough for the persistence rule,
not for a learned GRU, which needs a feature vector. This caches one.

For each frame we run the detector on native-resolution 640-px tiles, find the tile
with the highest detection confidence (the most smoke-like region), and keep THAT
tile's 256-d embedding plus its confidence. Focusing on the max-conf tile is the point:
a whole-frame embedding pools the plume away, but the plume's own tile describes it.

    python src/models/figlib_tiled_embed.py --tile 640 --stride 640

Output: data/figlib/features_tiled_embed.npz {stems, feats(256), confs}. Feed it to
figlib_temporal.py via --features to run the single-frame / persistence / GRU comparison
on the tiled signal.
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
from models.figlib_temporal import scan_frames  # noqa: E402
from models.figlib_tiled import tiles_of, pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    model = YOLO(str(ROOT / "runs/grouped_proof/weights/best.pt"))
    df = scan_frames()
    print(f"tiled embeddings for {len(df)} frames, tile={args.tile} stride={args.stride}, {device}")

    feats = np.zeros((len(df), 256), dtype=np.float32)
    confs = np.zeros(len(df), dtype=np.float32)
    for i, path in enumerate(df["path"]):
        img = Image.open(path).convert("RGB")
        crops = list(tiles_of(img, args.tile, args.stride))
        best_conf, best_tile = 0.0, 0
        for j in range(0, len(crops), args.batch):
            res = model.predict(crops[j : j + args.batch], device=device,
                                imgsz=args.tile, conf=0.001, verbose=False)
            for t, r in enumerate(res):
                c = r.boxes.conf
                cm = float(c.max()) if len(c) else 0.0
                if cm >= best_conf:
                    best_conf, best_tile = cm, j + t
        # embed only the winning tile -- the region the detector found most smoke-like
        emb = model.embed([crops[best_tile]], device=device, imgsz=args.tile, verbose=False)
        feats[i] = emb[0].detach().cpu().numpy().astype(np.float32)
        confs[i] = best_conf
        if i % 100 == 0:
            print(f"  {i}/{len(df)}", flush=True)

    out = FIGLIB / "features_tiled_embed.npz"
    np.savez_compressed(out, stems=df["stem"].to_numpy(), feats=feats, confs=confs)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
