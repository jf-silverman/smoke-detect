"""Evaluate a trained detector the way an operator would judge it.

Ultralytics reports mAP, which the research showed is close to meaningless for an
amorphous, boundary-less object like smoke. So this script reframes evaluation as the
decision the system actually makes: *given this frame, do we raise an alarm?*

For each test image we ask: did the model emit any box above the confidence
threshold? That is the alarm. Then, image-level:

  - precision / recall / F1 over a confidence sweep  -> the PR curve, not one dot
  - false alarms on the TRUE-NEGATIVE subset specifically -> the precision-collapse
    number. These images contain no smoke; every alarm on them is the model firing
    on clouds/fog/haze/glare, which is the documented single-frame failure mode.

    python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_manifest(split: str, part: str) -> list[Path]:
    f = PROC / "splits" / f"{split}_{part}.txt"
    return [Path(p) for p in f.read_text().split() if p]


def is_positive(img_path: Path) -> bool:
    """A test image is a real positive iff its label file has any box."""
    lbl = PROC / "labels" / f"{img_path.stem}.txt"
    return lbl.exists() and bool(lbl.read_text().strip())


def max_conf_per_image(model: YOLO, images: list[Path], device: str, imgsz: int) -> np.ndarray:
    """Return each image's highest box confidence (0.0 if the model emits nothing)."""
    scores = np.zeros(len(images), dtype=float)
    B = 64
    for i in range(0, len(images), B):
        batch = [str(p) for p in images[i : i + B]]
        results = model.predict(batch, device=device, imgsz=imgsz, conf=0.001, verbose=False)
        for k, r in enumerate(results):
            c = r.boxes.conf
            scores[i + k] = float(c.max()) if len(c) else 0.0
    return scores


def sweep(scores: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    rows = []
    n_pos = int(y_true.sum())
    n_neg = int((~y_true.astype(bool)).sum())
    for t in thresholds:
        alarm = scores >= t
        tp = int((alarm & y_true.astype(bool)).sum())
        fp = int((alarm & ~y_true.astype(bool)).sum())
        fn = n_pos - tp
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / n_pos if n_pos else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(
            {
                "threshold": round(float(t), 3),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "false_alarms_on_negatives": fp,
                "false_alarm_rate_on_negatives": round(fp / n_neg, 4) if n_neg else 0.0,
            }
        )
    return rows


def precision_at_base_rate(tpr: float, fpr: float, p: float) -> float:
    """Precision the model WOULD show at deployment base rate p.

    Aggregate precision on a smoke-heavy test set is flattered by the base rate.
    Real towers see smoke on a tiny fraction of frames, so the honest question is:
    given this recall (TPR) and this false-alarm rate on clean frames (FPR), what
    precision results when positives are rare? This is the precision-collapse made
    explicit -- and it is the number that predicts field usability.
    """
    num = tpr * p
    den = tpr * p + fpr * (1.0 - p)
    return num / den if den else float("nan")


def base_rate_table(rows: list[dict], base_rates=(0.05, 0.01, 0.005)) -> list[dict]:
    out = []
    for r in rows:
        entry = {"threshold": r["threshold"], "recall": r["recall"],
                 "fpr_on_negatives": r["false_alarm_rate_on_negatives"]}
        for p in base_rates:
            entry[f"precision@{p}"] = round(
                precision_at_base_rate(r["recall"], r["false_alarm_rate_on_negatives"], p), 4)
        out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True, help="which split's test manifest to score")
    ap.add_argument("--part", default="test", choices=["test", "val", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device()
    images = load_manifest(args.split, args.part)
    y_true = np.array([is_positive(p) for p in images], dtype=int)
    print(f"{args.split}/{args.part}: {len(images)} images, "
          f"{int(y_true.sum())} positive, {int((1 - y_true).sum())} negative")

    model = YOLO(args.weights)
    scores = max_conf_per_image(model, images, device, args.imgsz)

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = sweep(scores, y_true, thresholds)

    best = max(rows, key=lambda r: r["f1"])
    print(f"\noperating point (best-F1 @ conf={best['threshold']}):")
    print(f"  precision {best['precision']:.3f}  recall {best['recall']:.3f}  F1 {best['f1']:.3f}")
    print(f"  false alarms on {int((1 - y_true).sum())} negatives: "
          f"{best['false_alarms_on_negatives']} "
          f"({best['false_alarm_rate_on_negatives'] * 100:.1f}% of clean frames)")

    print("\nconf   prec   recall  F1     FA-on-neg")
    for r in rows:
        print(f"{r['threshold']:.2f}   {r['precision']:.3f}  {r['recall']:.3f}   "
              f"{r['f1']:.3f}  {r['false_alarm_rate_on_negatives'] * 100:5.1f}%")

    br = base_rate_table(rows)
    test_p = float(y_true.mean())
    print(f"\nprecision-collapse: aggregate precision above is inflated by this test set's")
    print(f"{test_p * 100:.0f}%-positive base rate. At realistic deployment base rates:")
    print(f"  conf   recall   FPR   | prec@5%  prec@1%  prec@0.5%")
    for e in br:
        if e["threshold"] not in (0.05, 0.10, 0.20, 0.30, 0.50):
            continue
        print(f"  {e['threshold']:.2f}   {e['recall']:.3f}  {e['fpr_on_negatives']:.3f} | "
              f"{e['precision@0.05'] * 100:6.1f}%  {e['precision@0.01'] * 100:6.1f}%  "
              f"{e['precision@0.005'] * 100:6.1f}%")

    out = Path(args.out) if args.out else ROOT / "runs" / f"eval_{args.split}_{args.part}.json"
    out.write_text(json.dumps({"split": args.split, "part": args.part,
                               "n_images": len(images), "n_positive": int(y_true.sum()),
                               "test_base_rate": round(test_p, 4),
                               "best": best, "sweep": rows,
                               "base_rate_corrected": br}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
