"""Evaluate a trained detector the way the wildfire field actually judges one.

Ultralytics reports mAP, meaningless for an amorphous object like smoke, and F1, which
weights a missed fire and a false alarm equally -- wrong for a domain where a missed fire
is catastrophic and a false alarm costs a watchstander a glance (they review every "found
fire" before dispatch). Operational camera networks (Pano, ALERTCalifornia) and satellite
products (NOAA) don't optimize F1; they run at high detection rate and report the
false-alarm BURDEN in operator units, and meteorology scores value across cost-loss ratios.

So the HEADLINE here is recall-first:

  - POD (probability of detection = recall) over a confidence sweep, and the MAX POD the
    detector can reach at all -- the recall-first ceiling.
  - at a target POD, the false-alarm burden as FALSE POSITIVES PER CAMERA PER DAY (Pano's
    operational target is < 1), plus FAR (false-alarm ratio) and POFD.
  - Relative Economic Value across cost-loss ratios C/L -- the meteorological score for
    asymmetric costs; small C/L (misses dominate) is the wildfire regime.

F1 and base-rate-corrected precision are still computed, but demoted to CONTEXT: the
base-rate ("precision-collapse") number is the alarm-fatigue constraint -- how often a
human gets pinged -- not a verdict that the detector is bad.

    python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped \
        --target-pod 0.90 --base-rate 0.01 --frames-per-day 500
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
        pofd = fp / n_neg if n_neg else 0.0  # prob of false detection (FA rate on negatives)
        far = fp / (tp + fp) if (tp + fp) else 0.0  # false-alarm RATIO = 1 - precision (met. std)
        rows.append(
            {
                "threshold": round(float(t), 3),
                "precision": round(prec, 4),
                "recall": round(rec, 4),          # == POD / probability of detection / hit rate
                "pod": round(rec, 4),             # explicit field-standard name
                "far": round(far, 4),             # false-alarm ratio FP/(FP+TP) = 1 - precision
                "f1": round(f1, 4),
                "csi": round(tp / (tp + fp + fn), 4) if (tp + fp + fn) else 0.0,  # critical success index
                "false_alarms_on_negatives": fp,
                "false_alarm_rate_on_negatives": round(pofd, 4),  # == POFD
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


# --- domain-correct, asymmetric-cost metrics ------------------------------------
# In wildfire detection a missed fire costs far more than a false alarm (a watchstander
# reviews every "found fire" before dispatch). So the field -- operational camera nets
# (Pano, ALERTCalifornia) and satellite products (NOAA) -- does not optimize F1. It runs
# at high POD and reports the false-alarm BURDEN in operator units (false positives per
# camera per day), and meteorology scores value across cost-loss ratios (relative economic
# value). These helpers report the model that way.

def false_alarms_per_camera_day(pofd: float, base_rate: float, frames_per_day: float) -> float:
    """Expected FALSE alarms per camera per day at a deployment base rate.

    A camera images a bearing every couple of minutes in daylight, so frames_per_day is an
    ASSUMPTION (pyro-sdis is not a continuous feed). False alarms come from the clean-frame
    stream: frames/day * P(clean) * POFD. Pano's operational target is < 1 per camera/day.
    """
    return frames_per_day * (1.0 - base_rate) * pofd


def relative_economic_value(pod: float, pofd: float, base_rate: float, alpha: float) -> float:
    """Relative economic value (Richardson 2000) at cost-loss ratio alpha = C/L.

    The meteorological answer to asymmetric costs. C is the cost of acting on an alarm
    (a human review), L the loss from a missed fire; alpha = C/L is small when misses
    dominate -- the wildfire regime. REV scores the forecast from 0 (no better than the
    climatological default of always/never alarming) to 1 (perfect), for a user with this
    alpha. s = base rate, H = POD, F = POFD.
    """
    s, H, F = base_rate, pod, pofd
    e_forecast = alpha * (s * H + (1 - s) * F) + s * (1 - H)
    e_climate = min(alpha, s)
    e_perfect = alpha * s
    den = e_climate - e_perfect
    return (e_climate - e_forecast) / den if den else float("nan")


def recall_first_point(rows: list[dict], base_rate: float, frames_per_day: float,
                       target_pod: float = 0.90):
    """Highest-POD operating point at/above target_pod, with its false-alarm burden.

    Recall-first: pick the lowest threshold reaching the required detection rate, then
    report what that costs in false alarms per camera per day. If target_pod is
    unreachable, return the max-POD row instead (flagged)."""
    reachable = [r for r in rows if r["recall"] >= target_pod]
    if reachable:
        r = min(reachable, key=lambda r: r["threshold"])  # lowest thr = highest recall margin
        met = True
    else:
        r = max(rows, key=lambda r: r["recall"])
        met = False
    return {
        "target_pod": target_pod, "target_met": met, "threshold": r["threshold"],
        "pod": r["recall"], "far": r.get("far"),
        "pofd": r["false_alarm_rate_on_negatives"],
        "false_alarms_per_camera_day": round(
            false_alarms_per_camera_day(r["false_alarm_rate_on_negatives"], base_rate, frames_per_day), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True, help="which split's test manifest to score")
    ap.add_argument("--part", default="test", choices=["test", "val", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-pod", type=float, default=0.90,
                    help="recall-first operating point: minimum detection rate to hold")
    ap.add_argument("--base-rate", type=float, default=0.01,
                    help="assumed deployment base rate of smoke frames")
    ap.add_argument("--frames-per-day", type=float, default=500.0,
                    help="assumed frames imaged per camera-bearing per day (for FP/camera/day)")
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

    test_p = float(y_true.mean())

    # --- HEADLINE: recall-first, the domain-correct framing ---------------------
    rf = recall_first_point(rows, args.base_rate, args.frames_per_day, args.target_pod)
    max_pod = max(r["recall"] for r in rows)
    print(f"\n=== RECALL-FIRST (a missed fire >> a false alarm) ===")
    print(f"max reachable POD (detection rate): {max_pod:.3f}")
    tag = "" if rf["target_met"] else "  [TARGET UNREACHABLE -- reporting max-POD point]"
    print(f"operating point @ POD>={args.target_pod:.2f}{tag}  (conf {rf['threshold']}):")
    print(f"  POD {rf['pod']:.3f}   FAR {rf['far']:.3f}   POFD {rf['pofd']:.3f}")
    print(f"  false-alarm burden ~ {rf['false_alarms_per_camera_day']:.1f} FP/camera/day "
          f"(@ base rate {args.base_rate}, {args.frames_per_day:.0f} frames/day)")
    print(f"  Pano's operational target is < 1 FP/camera/day -- the gap is the work left.")

    # --- Relative Economic Value across cost-loss ratios (asymmetric-cost score) -
    alphas = (0.002, 0.01, 0.05, 0.1)
    print(f"\nRelative Economic Value at the recall-first point, by cost-loss ratio C/L:")
    print(f"  (small C/L = misses dominate = the wildfire regime; REV in [0,1], higher better)")
    rf_row = next(r for r in rows if r["threshold"] == rf["threshold"])
    rev = {}
    for a in alphas:
        v = relative_economic_value(rf_row["recall"], rf_row["false_alarm_rate_on_negatives"],
                                    args.base_rate, a)
        rev[a] = round(v, 4)
        print(f"    C/L={a:<6}: REV={v:+.3f}")

    # --- context (demoted): best-F1 and the alarm-fatigue / base-rate table ------
    best = max(rows, key=lambda r: r["f1"])
    br = base_rate_table(rows)
    print(f"\n(context) best-F1 point: conf {best['threshold']}, POD {best['recall']:.3f}, "
          f"FAR {best['far']:.3f}, F1 {best['f1']:.3f} -- F1 balances the two errors equally, "
          f"which this domain does not.")
    print(f"(context) alarm-fatigue check -- deployment precision at base rate {args.base_rate}: "
          f"{precision_at_base_rate(rf_row['recall'], rf_row['false_alarm_rate_on_negatives'], args.base_rate)*100:.1f}%")

    out = Path(args.out) if args.out else ROOT / "runs" / f"eval_{args.split}_{args.part}.json"
    out.write_text(json.dumps({"split": args.split, "part": args.part,
                               "n_images": len(images), "n_positive": int(y_true.sum()),
                               "test_base_rate": round(test_p, 4),
                               "deployment_base_rate": args.base_rate,
                               "frames_per_day": args.frames_per_day,
                               "max_pod": round(max_pod, 4),
                               "recall_first": rf, "rev_by_cost_loss": rev,
                               "best_f1": best, "sweep": rows,
                               "base_rate_corrected": br}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
