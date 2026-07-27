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
    python src/models/figlib_ttd.py --motion                    # + below-horizon motion probe

Optional `--motion` probe (still no model): tests whether inter-frame motion below the detected
horizon separates onset from pre-ignition frames better than single-frame confidence -- the
author's scanning observation (reports/backlog.md "Motion / change-detection below the horizon").

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
from PIL import Image  # only used by the optional --motion probe

ROOT = Path(__file__).resolve().parents[2]
FIGLIB = ROOT / "data" / "figlib"
RESULTS = ROOT / "results"

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$")
MOTION_CACHE = FIGLIB / "motion_feats.npz"


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


def bootstrap_ci(per_fire: dict, B: int = 2000, seed: int = 0) -> dict:
    """90% CIs by resampling fires with replacement -- n is small, so quantify it.

    Bootstraps the per-fire (detected, ttd) outcomes from one LOFO pass at a fixed operating
    point. Median TTD is over detected fires within each resample.
    """
    rng = np.random.default_rng(seed)
    det = np.array([f["detected"] for f in per_fire.values()], dtype=float)
    ttd = np.array([f["ttd_min"] if f["ttd_min"] is not None else np.nan
                    for f in per_fire.values()], dtype=float)
    n = len(det)
    drs, mts = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        drs.append(det[idx].mean())
        t = ttd[idx][~np.isnan(ttd[idx])]
        mts.append(np.median(t) if len(t) else np.nan)

    def ci(a: list) -> list:
        a = np.array(a, dtype=float)
        a = a[~np.isnan(a)]
        return [round(float(np.percentile(a, 5)), 3), round(float(np.percentile(a, 95)), 3)]

    return {"detection_rate_ci90": ci(drs), "median_ttd_ci90": ci(mts)}


# --- optional motion / change-detection probe (reads frames; still NO model, no GPU) --------------
# Tests the author's scanning observation (reports/backlog.md "Motion / change-detection anchored at
# the horizon"): a plume is often legible only by its inter-frame MOTION, and the diagnostic property
# is not that the smoke stays low -- it rises -- but that it is ANCHORED to the ground: the base sits
# at/near/below the horizon and the growing column stays CONNECTED to the horizon line. Clouds (74%
# of false alarms) drift as change that FLOATS entirely above the skyline, disconnected from it. So
# the feature is horizon-anchored change energy (a vertical run of change that touches the horizon
# band and extends up/down from it) vs floating change. This is separate from the persistence rule
# already tested: persistence suppresses flicker; differencing proposes change. The probe asks a
# cheap, training-free question: do these anchored-motion features separate onset (offset>=0) from
# pre-ignition (offset<0) frames better than the single-frame detector confidence?
#   Deliberately probabilistic: rare wind-driven or low-in-scene fires never clear the ridge, so
# anchoring is a likelihood bump, not a gate.
#   A texture cue (soft diffuse haze + hard column edge coexisting) was tried and REMOVED: it is not
# a stable smoke signature -- a high-Haines-index day gives a hard-edged column with little haze, and
# a windy day gives an all-diffuse column with no hard edge, so the soft x hard product is noise.

def estimate_horizon(gray: np.ndarray) -> int:
    """Row index of the sky->terrain transition: the steepest top-to-bottom drop in row-mean luma.

    Crude but adequate for a separability probe -- sky is brighter than terrain, so the horizon sits
    near the largest negative gradient of the smoothed row-brightness profile, searched in the middle
    band (0.15-0.85 of height) so it never latches onto the very top or bottom edge."""
    rows = gray.mean(axis=1).astype(float)
    k = max(3, len(rows) // 50)
    sm = np.convolve(rows, np.ones(k) / k, mode="same")
    grad = np.diff(sm)
    lo, hi = int(0.15 * len(grad)), int(0.85 * len(grad))
    band = grad[lo:hi]
    return lo + int(np.argmin(band)) if len(band) else len(rows) // 2


def _load_gray(path: Path, width: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.width > width:
        img = img.resize((width, max(1, round(img.height * width / img.width))))
    return np.asarray(img, dtype=np.int16)


FEAT_NAMES = ("anchored_change", "floating_change", "anchored_ratio")
MOTION_SCHEMA = 3  # bump to invalidate stale motion_feats.npz when the feature set/definition changes


def _frame_motion(cur: np.ndarray, prev: np.ndarray, h: int,
                  k_change: float, band_frac: float) -> tuple[float, float, float]:
    """Horizon-anchored change features for one frame vs its predecessor. See FEAT_NAMES.

    `h` is the fire's FIXED horizon row (passed in, not re-estimated per frame: the camera is static,
    so the horizon is constant, and estimating it per-frame lets a bright/dark plume drag its own
    horizon). Change map = |(cur-prev) - spatial_median|, which strips a uniform exposure/gain shift
    (and much dirty-lens flicker) so only localized change survives. A robust (MAD) threshold gives a
    change mask. Around the horizon we take a small band; a column is ANCHORED if it has change in the
    band, and its anchored energy is the band change plus the CONTIGUOUS change run extending upward
    (the rising column) and downward (the base). `cumprod` over the mask gives the contiguous run
    cheaply. Floating change is everything not so connected (cloud drift)."""
    H, W = cur.shape
    d = cur - prev
    diff = np.abs(d - np.median(d)).astype(float)          # strip uniform exposure/gain shift
    mad = float(np.median(np.abs(diff - np.median(diff)))) + 1e-6
    mask = diff > k_change * mad
    total = float(diff.sum())
    if total <= 0:
        return 0.0, 0.0, 0.0

    h = int(np.clip(h, 1, H - 2))
    band = max(1, round(band_frac * H))
    h0, h1 = max(h - band, 0), min(h + band + 1, H)
    cross = mask[h0:h1, :].any(axis=0)                     # (W,) columns whose change touches horizon
    band_energy = diff[h0:h1, :].sum(axis=0)

    up = mask[:h0, :][::-1, :]                             # rows just above the band, going up
    up_run = np.cumprod(up.astype(np.uint8), axis=0).astype(bool)   # True while contiguous from band
    up_energy = (diff[:h0, :][::-1, :] * up_run).sum(axis=0)
    dn = mask[h1:, :]                                      # rows just below the band, going down
    dn_run = np.cumprod(dn.astype(np.uint8), axis=0).astype(bool)
    dn_energy = (diff[h1:, :] * dn_run).sum(axis=0)

    anchored = float(np.where(cross, band_energy + up_energy + dn_energy, 0.0).sum())
    floating = max(total - anchored, 0.0)
    ratio = anchored / (anchored + floating + 1e-6)
    area = float(H * W)
    return anchored / area, floating / area, ratio


def fire_horizon(grays: list[np.ndarray], offsets: np.ndarray) -> int:
    """One FIXED horizon row for a fire: the median per-frame estimate over its PRE-IGNITION frames.

    The camera is static, so the horizon is a fire-level constant. Estimating from pre-ignition frames
    (offset < 0) keeps the plume from biasing its own horizon; falls back to all frames if a fire has
    no pre-ignition frames cached."""
    pre = [g for g, o in zip(grays, offsets) if o < 0]
    use = pre if pre else grays
    return int(np.median([estimate_horizon(g) for g in use]))


def compute_motion_features(df: pd.DataFrame, cache_path: Path, width: int = 512,
                            k_change: float = 4.0, band_frac: float = 0.02) -> dict:
    """Return {stem: (anchored, floating, ratio)} of horizon-anchored frame-to-previous motion.

    Per fire (time-ordered): load frames once, fix ONE horizon from the pre-ignition frames, then
    difference each frame from its predecessor (~60 s earlier) and reduce to the anchored-change
    features (see _frame_motion). Cached to npz with a schema tag; a fire's first frame has no
    predecessor and is omitted."""
    if cache_path.exists():
        a = np.load(cache_path, allow_pickle=True)
        if "schema" in a and int(a["schema"]) == MOTION_SCHEMA and all(n in a for n in FEAT_NAMES):
            cols = np.stack([a[n] for n in FEAT_NAMES], axis=1)
            return {s: tuple(float(v) for v in row) for s, row in zip(a["stems"].astype(str), cols)}
        print(f"  stale motion cache (schema != {MOTION_SCHEMA}); recomputing")
    stems, feats = [], []
    for seq, g in df.sort_values(["seq", "offset"]).groupby("seq"):
        g = g.reset_index(drop=True)
        grays = [_load_gray(FIGLIB / "images" / seq / f"{stem}.jpg", width) for stem in g["stem"]]
        h = fire_horizon(grays, g["offset"].to_numpy())            # one fixed horizon per fire
        prev = None
        for stem, cur in zip(g["stem"], grays):
            if prev is not None and prev.shape == cur.shape:
                stems.append(stem)
                feats.append(_frame_motion(cur, prev, h, k_change, band_frac))
            prev = cur
    feats = np.array(feats, float).reshape(-1, len(FEAT_NAMES))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, stems=np.array(stems), schema=np.array(MOTION_SCHEMA),
             **{n: feats[:, i] for i, n in enumerate(FEAT_NAMES)})
    print(f"  cached {len(stems)} motion features -> {cache_path}")
    return {s: tuple(row) for s, row in zip(stems, feats)}


def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (Mann-Whitney helper; avoids a scipy dependency)."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC of `scores` for binary `labels` (1=positive) via the rank (Mann-Whitney U) identity."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    m = ~np.isnan(scores)
    scores, labels = scores[m], labels[m]
    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(scores)
    u = r[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _fusion(conf: np.ndarray, anchored: np.ndarray) -> np.ndarray:
    """Training-free rank-sum fusion of detector confidence and anchored motion (NaN-safe)."""
    out = np.full(len(conf), np.nan)
    m = ~(np.isnan(conf) | np.isnan(anchored))
    if m.any():
        out[m] = _rankdata(conf[m]) + _rankdata(anchored[m])
    return out


def motion_separability(df: pd.DataFrame, width: int, boot: int = 2000, seed: int = 0) -> dict:
    """Onset-vs-pre-ignition separability of anchored-motion features vs single-frame confidence.

    Reports BOTH a per-fire AUC (computed within each fire, then averaged with a bootstrap-over-fires
    90% CI -- the leak-aware read, since it never pools frames across fires) and a pooled AUC for
    reference. The fusion (conf + anchored motion) is computed within each fire for the per-fire read.
    """
    feats = compute_motion_features(df, MOTION_CACHE, width)
    nan_row = (np.nan,) * len(FEAT_NAMES)
    cols = {n: df["stem"].map(lambda s, i=i: feats.get(s, nan_row)[i]).to_numpy(float)
            for i, n in enumerate(FEAT_NAMES)}
    conf = df["conf"].to_numpy(float)
    label = (df["offset"].to_numpy() >= 0).astype(int)          # 1 = onset, 0 = pre-ignition
    seqs = df["seq"].to_numpy()
    valid = ~np.isnan(cols["anchored_change"])
    order = ["single_frame_conf", *FEAT_NAMES, "conf_plus_anchored_ranksum"]

    def series(name, mask):
        if name == "single_frame_conf":
            return conf[mask]
        if name == "conf_plus_anchored_ranksum":
            return _fusion(conf[mask], cols["anchored_change"][mask])
        return cols[name][mask]

    # pooled AUC (all valid frames together)
    pooled = {n: auc(series(n, valid), label[valid]) for n in order}

    # per-fire AUC: compute within each fire, keep the vector of fire-level AUCs
    per_fire = {n: [] for n in order}
    for seq in np.unique(seqs):
        m = (seqs == seq) & valid
        lab = label[m]
        if lab.sum() == 0 or (lab == 0).sum() == 0:          # need both classes in the fire
            continue
        for n in order:
            a = auc(series(n, m), lab)
            if not np.isnan(a):
                per_fire[n].append(a)

    rng = np.random.default_rng(seed)

    def agg(vals):
        vals = np.array(vals, float)
        if len(vals) == 0:
            return None, [None, None]
        boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(boot)]
        return (round(float(vals.mean()), 3),
                [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)])

    mean_auc, ci = {}, {}
    for n in order:
        mean_auc[n], ci[n] = agg(per_fire[n])

    return {
        "order": order,
        "n_frames_scored": int(valid.sum()),
        "n_fires_scored": len(per_fire["anchored_change"]),
        "n_onset": int(label[valid].sum()), "n_preignition": int((label[valid] == 0).sum()),
        "auc_per_fire_mean": mean_auc,
        "auc_per_fire_ci90": ci,
        "auc_pooled": {n: (None if np.isnan(v) else round(v, 3)) for n, v in pooled.items()},
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
    ap.add_argument("--motion", action="store_true",
                    help="also run the motion / below-horizon change-detection separability probe "
                         "(reads frames; still no model). Caches to data/figlib/motion_feats.npz")
    ap.add_argument("--motion-width", type=int, default=512,
                    help="downscaled width for the motion probe's frame differencing (default 512)")
    args = ap.parse_args()

    df = load_conf(scan_frames(), Path(args.features))
    print(f"FIgLib TTD: {df['seq'].nunique()} fires, {len(df)} frames "
          f"({int(df['smoke'].sum())} post-ignition / {int((~df['smoke']).sum())} pre-ignition)")
    print(f"features: {args.features}\n")

    # headline at the requested operating point, with bootstrap CIs (n is small)
    head = evaluate(df, args.far_target, args.persist)
    ci = bootstrap_ci(head["per_fire"])
    head["ci90"] = ci
    print(f"=== TTD @ pre-ignition FA target {args.far_target:.0%}, persist k={args.persist} "
          f"(leave-one-fire-out) ===")
    print(f"  detection rate            : {head['detection_rate']:.0%} of fires   "
          f"90% CI [{ci['detection_rate_ci90'][0]:.0%}, {ci['detection_rate_ci90'][1]:.0%}]")
    print(f"  median TTD (detected)     : {head['median_ttd_min']} min   "
          f"90% CI [{ci['median_ttd_ci90'][0]}, {ci['median_ttd_ci90'][1]}] min")
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

    motion = None
    if args.motion:
        print("\n--- Motion / horizon-anchored change-detection probe (onset vs pre-ignition) ---")
        print("    reading frames (cached after first run)...")
        motion = motion_separability(df, args.motion_width)
        print(f"  scored {motion['n_frames_scored']} frames across {motion['n_fires_scored']} fires "
              f"({motion['n_onset']} onset / {motion['n_preignition']} pre-ignition)")
        print(f"    {'feature':<28s} {'per-fire AUC (90% CI)':<26s} pooled")
        for name in motion["order"]:
            m = motion["auc_per_fire_mean"][name]
            lo, hi = motion["auc_per_fire_ci90"][name]
            p = motion["auc_pooled"][name]
            flag = "  <- baseline" if name == "single_frame_conf" else ""
            ms = "n/a" if m is None else f"{m:.3f}"
            cis = "" if lo is None else f"[{lo:.3f}, {hi:.3f}]"
            ps = "n/a" if p is None else f"{p:.3f}"
            print(f"    {name:<28s} {ms:>5s} {cis:<20s} {ps:>6s}{flag}")
        print("  (per-fire AUC is the leak-aware read -- computed within each fire, then averaged.\n"
              "   >0.5 separates onset. anchored_ratio is the key motion feature; floating_change is\n"
              "   the cloud-drift control (want ~0.5); the fusion tests whether motion ADDS to the\n"
              "   detector. A win promotes anchored motion to a Phase C temporal channel -- backlog.md.)")

    out = RESULTS / f"figlib_ttd{args.tag}.json"
    RESULTS.mkdir(exist_ok=True)
    payload = {"headline": head, **sweep}
    if motion is not None:
        payload["motion_probe"] = motion
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {out}")
    print("\nNOTE: ~17 fires is small; magnitudes are directional. Phase B (FIgLib-full / "
          "PYRONEAR-2025) tightens them. See reports/backlog.md.")


if __name__ == "__main__":
    main()
