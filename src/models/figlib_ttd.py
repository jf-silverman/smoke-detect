"""FIgLib time-to-detection (TTD): how many minutes after ignition until the first alarm?

TTD is the field's headline metric (Pano; SmokeyNet ~3.6 min, 80% of fires within 5 min;
Dewangan et al., 2022) and the one this project has lacked, because pyro-sdis has no ignition
onset. FIgLib does: each frame's filename encodes the signed offset from first visible plume
(`<unixts>_<+/-offset_seconds>.jpg`, offset >= 0 == smoke), so minutes-from-ignition ground
truth is free. This eval REUSES the cached native-resolution tiled per-frame confidences
(`features_tiled.npz` from figlib_tiled.py) -- **no model is run here**, so it costs no GPU.

Leak-safe by leave-one-fire-out (LOFO): for each held-out fire, the alarm threshold is
calibrated on the OTHER fires to a target pre-ignition false-alarm rate (the operator's
trigger-happiness budget), then TTD is measured on the held-out fire. Every fire is used as a
test fire while never being calibrated on -- the right choice for FIgLib's small fire count.

Reported (the operator-relevant PAIR -- 'early' must not be bought with 'cries wolf'):
  * detection rate          -- share of fires ever detected post-ignition. A MISS is right-
                               censored (never averaged in as a small TTD).
  * median / mean TTD        -- minutes, over DETECTED fires only.
  * % of fires within 5 min  -- comparable to SmokeyNet's ~80%.
  * pre-ignition FA rate     -- fraction of pre-ignition frames that alarm, at the operating point.
  * persistence-k variant    -- require k consecutive crossings before alarming; its TTD cost
                               directly extends the persistence sign-flip finding (figlib-findings.md).

    python src/models/figlib_tiled.py --tile 640 --stride 640   # once: caches features_tiled.npz
    python src/models/figlib_ttd.py --far-target 0.05           # this eval (no model, no GPU)

STATUS: Phase A scaffold. The metric machinery and LOFO harness are complete and run on the
18 local fires. Small n (~4 effective test signal per fold is avoided by LOFO, but ~17 fires
total is still small) -> treat magnitudes as directional. Phase B (FIgLib-full via WIFIRE
Commons / PYRONEAR-2025) tightens the numbers; see reports/backlog.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGLIB = ROOT / "data" / "figlib"
RESULTS = ROOT / "results"

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$")


# --- data (mirrors figlib_temporal.scan_frames; kept local so this eval needs only numpy/pandas) ---
def scan_frames() -> pd.DataFrame:
    rows = []
    for seq_dir in sorted((FIGLIB / "images").glob("*/")):
        if not seq_dir.is_dir():
            continue
        for jpg in seq_dir.glob("*.jpg"):
            m = OFFSET_RE.search(jpg.name)
            if not m:
                continue
            offset = int(m.group(1))  # signed SECONDS from first visible plume
            rows.append({"stem": jpg.stem, "seq": seq_dir.name,
                         "offset": offset, "smoke": offset >= 0})
    df = pd.DataFrame(rows)
    return df.sort_values(["seq", "offset"]).reset_index(drop=True)


def load_conf(df: pd.DataFrame, features_path: Path) -> pd.DataFrame:
    """Attach the cached per-frame max confidence to df by stem. No model run."""
    if not features_path.exists():
        raise SystemExit(
            f"no feature cache at {features_path}.\n"
            f"Run `python src/models/figlib_tiled.py --tile 640 --stride 640` first "
            f"(that step runs the detector once and writes features_tiled.npz).")
    arch = np.load(features_path, allow_pickle=True)
    key = "conf_tiled" if "conf_tiled" in arch else "confs"  # tiled = native-res (preferred)
    conf = {s: float(c) for s, c in zip(arch["stems"].astype(str), arch[key])}
    df = df[df["stem"].map(lambda s: s in conf)].reset_index(drop=True)
    df["conf"] = df["stem"].map(conf).astype(float)
    return df


# --- threshold calibration ---
def threshold_for_far(pre_conf: np.ndarray, far_target: float) -> float:
    """Lowest (most sensitive) threshold whose pre-ignition false-alarm rate <= far_target.

    Calibrated on pre-ignition frames (offset < 0) of the *calibration* fires. Returns the
    confidence above which at most far_target of pre-ignition frames fire.
    """
    if len(pre_conf) == 0:
        return 0.0
    # the (1 - far_target) quantile: ~far_target of pre frames exceed it. nextafter nudges the
    # threshold just above the quantile so the budget is respected rather than exactly met.
    q = float(np.quantile(pre_conf, 1.0 - far_target, method="higher"))
    return float(np.nextafter(q, np.inf))


# --- per-fire TTD ---
def fire_ttd(fire_df: pd.DataFrame, thr: float, persist_k: int) -> tuple[bool, float | None]:
    """Return (detected, ttd_minutes) for one fire at threshold thr.

    The alarm fires at the k-th consecutive post-ignition frame with conf >= thr; TTD is that
    frame's minutes-from-ignition. persist_k=1 is the plain single-frame rule.
    """
    post = fire_df[fire_df["offset"] >= 0].sort_values("offset")
    run = 0
    for offset, conf in zip(post["offset"].to_numpy(), post["conf"].to_numpy()):
        run = run + 1 if conf >= thr else 0
        if run >= persist_k:
            return True, offset / 60.0
    return False, None


def pre_ignition_far(fire_df: pd.DataFrame, thr: float) -> float:
    pre = fire_df[fire_df["offset"] < 0]
    return float((pre["conf"] >= thr).mean()) if len(pre) else float("nan")


# --- LOFO evaluation ---
def evaluate(df: pd.DataFrame, far_target: float, persist_k: int) -> dict:
    fires = sorted(df["seq"].unique())
    ttds, detected, held_far = [], [], []
    per_fire = {}
    for held in fires:
        cal = df[df["seq"] != held]
        test = df[df["seq"] == held]
        thr = threshold_for_far(cal.loc[cal["offset"] < 0, "conf"].to_numpy(), far_target)
        det, ttd = fire_ttd(test, thr, persist_k)
        detected.append(det)
        held_far.append(pre_ignition_far(test, thr))
        if det:
            ttds.append(ttd)
        per_fire[held] = {"detected": det, "ttd_min": None if ttd is None else round(ttd, 2),
                          "threshold": round(thr, 4)}
    ttds = np.array(ttds, dtype=float)
    n = len(fires)
    within5_all = float(np.mean([pf["ttd_min"] is not None and pf["ttd_min"] <= 5.0
                                 for pf in per_fire.values()]))
    return {
        "far_target": far_target,
        "persist_k": persist_k,
        "n_fires": n,
        "detection_rate": round(float(np.mean(detected)), 3),
        "median_ttd_min": None if len(ttds) == 0 else round(float(np.median(ttds)), 2),
        "mean_ttd_min": None if len(ttds) == 0 else round(float(np.mean(ttds)), 2),
        "pct_within_5min_of_all": round(within5_all, 3),
        "mean_preignition_far_heldout": round(float(np.nanmean(held_far)), 4),
        "per_fire": per_fire,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(FIGLIB / "features_tiled.npz"),
                    help="cached per-frame confidences (default: native-res tiled)")
    ap.add_argument("--far-target", type=float, default=0.05,
                    help="target pre-ignition false-alarm rate the threshold is calibrated to")
    ap.add_argument("--persist", type=int, default=1,
                    help="require k consecutive crossings before alarming (1 = single-frame)")
    ap.add_argument("--tag", default="", help="suffix for the output json")
    args = ap.parse_args()

    df = load_conf(scan_frames(), Path(args.features))
    print(f"FIgLib TTD: {df['seq'].nunique()} fires, {len(df)} frames "
          f"({int(df['smoke'].sum())} post-ignition / {int((~df['smoke']).sum())} pre-ignition)")
    print(f"features: {args.features}\n")

    # headline at the requested operating point
    head = evaluate(df, args.far_target, args.persist)
    print(f"=== TTD @ pre-ignition FA target {args.far_target:.0%}, persist k={args.persist} "
          f"(leave-one-fire-out) ===")
    print(f"  detection rate            : {head['detection_rate']:.0%} of fires")
    print(f"  median TTD (detected)     : {head['median_ttd_min']} min")
    print(f"  mean TTD (detected)       : {head['mean_ttd_min']} min   "
          f"(SmokeyNet ref ~3.6 min)")
    print(f"  % of ALL fires within 5min: {head['pct_within_5min_of_all']:.0%}   "
          f"(SmokeyNet ref ~80%)")
    print(f"  held-out pre-ignition FA  : {head['mean_preignition_far_heldout']:.1%}")

    # the operator trade-off: sweep the false-alarm budget, and show the TTD cost of persistence
    sweep = {"far_sweep": [], "persist_sweep": []}
    print("\n--- TTD vs pre-ignition false-alarm budget (persist k=1) ---")
    print("FA target   detect%   median TTD   %within5")
    for ft in (0.02, 0.05, 0.10, 0.20):
        r = evaluate(df, ft, 1)
        sweep["far_sweep"].append(r)
        print(f"  {ft:>5.0%}     {r['detection_rate']:>5.0%}     "
              f"{str(r['median_ttd_min']):>8s} min   {r['pct_within_5min_of_all']:>5.0%}")

    print("\n--- TTD cost of requiring persistence (FA target "
          f"{args.far_target:.0%}) ---")
    print("persist k   detect%   median TTD   %within5")
    for k in (1, 2, 3):
        r = evaluate(df, args.far_target, k)
        sweep["persist_sweep"].append(r)
        print(f"  {k:>5d}     {r['detection_rate']:>5.0%}     "
              f"{str(r['median_ttd_min']):>8s} min   {r['pct_within_5min_of_all']:>5.0%}")

    out = RESULTS / f"figlib_ttd{args.tag}.json"
    RESULTS.mkdir(exist_ok=True)
    out.write_text(json.dumps({"headline": head, **sweep}, indent=2))
    print(f"\nsaved -> {out}")
    print("\nNOTE: ~17 fires is small; magnitudes are directional. Phase B (FIgLib-full / "
          "PYRONEAR-2025) tightens them. See reports/backlog.md.")


if __name__ == "__main__":
    main()
