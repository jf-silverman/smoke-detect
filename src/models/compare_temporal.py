"""Definitive single-frame vs temporal comparison, at MATCHED RECALL.

Comparing a temporal model to a single-frame detector by each one's own best-F1
threshold is misleading -- they put their scores on different scales. The correct
comparison holds recall fixed and asks the operator's real question: *at the recall
you require, how many clean frames still false-alarm?* Lower is better.

Three score functions over the identical grouped test set (same held-out towers):

  single-frame : the detector's per-frame max confidence (the baseline).
  persistence  : rolling MIN of confidence over the last W frames in the burst --
                 the interpretable temporal rule ("only alarm if evidence persisted").
  temporal-gru : the learned GRU head over the confidence sequence.

The persistence rule encodes the literature's mechanism directly: real smoke persists,
flicker false-alarms do not. Whether it helps on THIS data is the empirical question.

    python src/models/compare_temporal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from models.temporal import load_frames, make_windows, run, pick_device  # noqa: E402


def burst_rolling_min(df, W: int) -> np.ndarray:
    """Per-frame causal rolling min of conf within its burst (persistence score)."""
    out = np.zeros(len(df), dtype=float)
    pos = {stem: i for i, stem in enumerate(df["stem"].to_numpy())}
    for _, g in df.groupby("burst", sort=False):
        g = g.sort_values("ts")
        conf = g["conf_raw"].to_numpy()
        for t in range(len(g)):
            lo = max(0, t - W + 1)
            out[pos[g.iloc[t]["stem"]]] = conf[lo : t + 1].min()
    return out


def fpr_at_recall(scores, y_true, target_recall):
    """Lowest false-alarm rate on negatives achievable at >= target recall."""
    y = y_true.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    order = np.unique(scores)[::-1]
    best_fpr = 1.0
    got = False
    for t in order:
        alarm = scores >= t
        rec = (alarm & y).sum() / n_pos
        if rec >= target_recall:
            fpr = (alarm & ~y).sum() / n_neg
            best_fpr = min(best_fpr, fpr)
            got = True
    return best_fpr if got else float("nan")


def prec_at(tpr, fpr, p):
    den = tpr * p + fpr * (1 - p)
    return tpr * p / den if den else float("nan")


def main() -> None:
    W = 8
    device = pick_device()
    print(f"device: {device}\nloading frames + features ...")
    df = load_frames()
    # keep an un-standardized copy of conf for the single-frame + persistence scores
    df["conf_raw"] = df["feat"].map(lambda v: float(v[-1]))

    te = df[df["split"] == "test"].copy()
    y = te["smoke"].to_numpy().astype(int)

    # --- score 1: single frame
    s_single = te["conf_raw"].to_numpy()
    # --- score 2: persistence (rolling min over the burst)
    s_persist_full = burst_rolling_min(df, W)
    stem_to_i = {s: i for i, s in enumerate(df["stem"].to_numpy())}
    s_persist = np.array([s_persist_full[stem_to_i[s]] for s in te["stem"]])

    # --- score 3: learned GRU over conf sequence (standardized, train-only stats)
    train_feat = np.stack(df.loc[df["split"] == "train", "feat"].to_numpy())
    mu, sigma = train_feat.mean(0), train_feat.std(0) + 1e-6
    dfn = df.copy()
    dfn["feat"] = dfn["feat"].map(lambda v: (((v - mu) / sigma)[-1:]).astype(np.float32))
    arrays = {p: make_windows(dfn[dfn.split == p], W) for p in ("train", "val", "test")}
    s_gru = run(arrays, epochs=60, lr=1e-3, device=device, seed=0)

    methods = {"single-frame": s_single, "persistence": s_persist, "temporal-gru": s_gru}

    print("\nFALSE-ALARM RATE ON CLEAN FRAMES at matched recall (lower is better):\n")
    targets = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
    header = "recall  " + "  ".join(f"{m:>13s}" for m in methods)
    print(header)
    table = {"targets": targets, "methods": {}}
    for m in methods:
        table["methods"][m] = {}
    for tr in targets:
        cells = []
        for m, s in methods.items():
            fpr = fpr_at_recall(s, y, tr)
            table["methods"][m][f"recall_{tr}"] = None if np.isnan(fpr) else round(float(fpr), 4)
            cells.append(f"{fpr*100:12.1f}%" if not np.isnan(fpr) else f"{'n/a':>13s}")
        print(f"{tr:.2f}  " + "  ".join(cells))

    print("\nPRECISION @ 1% deployment base rate, at matched recall (higher is better):\n")
    print(header)
    for tr in targets:
        cells = []
        for m, s in methods.items():
            fpr = fpr_at_recall(s, y, tr)
            pr = prec_at(tr, fpr, 0.01)
            cells.append(f"{pr*100:12.2f}%" if not np.isnan(pr) else f"{'n/a':>13s}")
        print(f"{tr:.2f}  " + "  ".join(cells))

    out = RESULTS / "temporal_comparison.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
