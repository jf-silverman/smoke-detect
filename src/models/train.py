"""Train a single-frame YOLO smoke detector on one split.

This is the BASELINE. Its purpose is not to win -- it is to establish a rigorous
single-frame number on held-out sites, so a temporal model later has something real
to beat, and so the single-frame precision collapse (false positives on clouds/fog)
is documented rather than assumed.

    python src/models/train.py --split grouped --epochs 40
    python src/models/train.py --split loso_brison --epochs 40 --name loso_brison

The --split name refers to a data/processed/<split>.yaml produced by export_yolo.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "runs"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="grouped", help="name of data/processed/<split>.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="ultralytics weights to start from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default=None, help="run name (defaults to split)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train data (smoke test)")
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    data_yaml = PROC / f"{args.split}.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"no split yaml at {data_yaml} -- run src/data/export_yolo.py first")

    device = pick_device()
    name = args.name or args.split
    print(f"training '{name}' on {device}: {args.model}, {args.epochs} epochs, imgsz {args.imgsz}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS),
        name=name,
        fraction=args.fraction,
        patience=args.patience,
        # smoke is small and low-contrast: keep the augs that help, drop the ones
        # that fabricate implausible smoke (heavy mosaic can splice plumes onto
        # backgrounds they never occur over). Conservative, physically-motivated defaults.
        mosaic=0.5,
        mixup=0.0,
        degrees=0.0,      # towers are gravity-aligned; rotation is unphysical
        flipud=0.0,       # smoke rises -- vertical flip inverts the physics
        fliplr=0.5,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    print(f"done -> {RUNS / name}")


if __name__ == "__main__":
    main()
