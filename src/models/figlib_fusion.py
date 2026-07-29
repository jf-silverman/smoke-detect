"""Motion temporal channel — a leak-free per-frame learned fusion of appearance + anchored motion.

The motion probe ([figlib_ttd.py](figlib_ttd.py) --motion) showed anchored-motion change carries
onset signal complementary to the appearance detector, but a *naive* equal-weight sum of the two
HURTS (it dilutes the stronger feature). This trains the honest combiner the reports promised: a
per-frame logistic regression over [conf, anchored_change, anchored_ratio, floating_change] ->
onset probability.

LEAK-FREE by construction: the appearance confidence comes from the zero-shot base detector
(gcp_grouped_1280, never trained on FIgLib) and the motion features are training-free, so fitting a
combiner on the training fires and applying it to the held-out EVAL_FIRES has no leakage. The learned
score is written as a `conf_tiled` npz so the existing TTD harness scores it unchanged -- the real
payoff test is whether it LOWERS time-to-detection vs appearance alone (run figlib_ttd --eval-only on
the fused npz and compare to the base npz).

    # prereqs (base conf on both splits, same model; motion caches):
    #   figlib_tiled.py --exclude-eval --weights runs/gcp_grouped_1280/weights/best.pt --out features_trainfires_base.npz
    #   figlib_tiled.py --eval-only    --weights runs/gcp_grouped_1280/weights/best.pt --out features_evalfires_base.npz
    #   figlib_ttd.py --exclude-eval --motion  (writes motion_feats.npz)
    #   figlib_ttd.py --eval-only --motion --motion-cache data/figlib/motion_feats_eval.npz ...
    python src/models/figlib_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGLIB = ROOT / "data" / "figlib"
sys.path.insert(0, str(ROOT / "src"))
from models.figlib_temporal import scan_frames  # noqa: E402

MOTION_FEATS = ("anchored_change", "anchored_ratio", "floating_change")
FEATURES = ("conf", *MOTION_FEATS)


def build_xy(conf_npz: Path, motion_npz: Path):
    """Join appearance conf + motion features per frame; return (X, y, stems, seqs, offsets)."""
    frames = scan_frames().set_index("stem")
    conf_a = np.load(conf_npz, allow_pickle=True)
    conf = dict(zip(conf_a["stems"].astype(str), conf_a["conf_tiled"].astype(float)))
    m = np.load(motion_npz, allow_pickle=True)
    mot = {s: {n: float(m[n][i]) for n in MOTION_FEATS}
           for i, s in enumerate(m["stems"].astype(str))}
    rows, stems, seqs, offs, ys = [], [], [], [], []
    for s in mot:                                   # motion is the limiting set (drops 1st frame/fire)
        if s not in conf or s not in frames.index:
            continue
        rows.append([conf[s], *(mot[s][n] for n in MOTION_FEATS)])
        stems.append(s); seqs.append(frames.at[s, "seq"])
        offs.append(int(frames.at[s, "offset"])); ys.append(int(frames.at[s, "offset"] >= 0))
    return (np.array(rows, float), np.array(ys, int), np.array(stems),
            np.array(seqs), np.array(offs, int))


def _auc(score, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, score) if len(set(y)) == 2 else float("nan")


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr, _, seqtr, _ = build_xy(FIGLIB / "features_trainfires_base.npz",
                                     FIGLIB / "motion_feats.npz")
    Xev, yev, sev, seqev, offev = build_xy(FIGLIB / "features_evalfires_base.npz",
                                           FIGLIB / "motion_feats_eval.npz")
    # drop any night frames that linger in the training motion cache
    from models.figlib_ttd import NIGHT_PATTERN
    import re
    keep = np.array([re.search(NIGHT_PATTERN, s, re.I) is None for s in seqtr])
    Xtr, ytr, seqtr = Xtr[keep], ytr[keep], seqtr[keep]
    print(f"train: {len(Xtr)} frames / {len(set(seqtr))} fires; eval: {len(Xev)} frames / "
          f"{len(set(seqev))} fires")

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)

    print("\nlearned weights (standardized -- sign & magnitude are directly comparable):")
    for name, w in zip(FEATURES, clf.coef_[0]):
        print(f"  {name:<16} {w:+.3f}")

    # leave-one-fire-out AUC on the training fires (a leak-aware read of the fit)
    loo = []
    for held in sorted(set(seqtr)):
        tr = seqtr != held; te = ~tr
        if len(set(ytr[te])) < 2:
            continue
        sc = StandardScaler().fit(Xtr[tr])
        c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr[tr]), ytr[tr])
        loo.append(_auc(c.predict_proba(sc.transform(Xtr[te]))[:, 1], ytr[te]))
    print(f"\ntrain leave-one-fire-out per-fire AUC: fused {np.nanmean(loo):.3f}  "
          f"(conf-alone {np.nanmean([_auc(Xtr[seqtr==h][:,0], ytr[seqtr==h]) for h in sorted(set(seqtr)) if len(set(ytr[seqtr==h]))==2]):.3f})")

    # apply to the held-out eval fires -> fused score, and write it for the TTD harness
    fused_ev = clf.predict_proba(scaler.transform(Xev))[:, 1]
    out = FIGLIB / "features_evalfires_fused.npz"
    np.savez_compressed(out, stems=sev, conf_tiled=fused_ev.astype(np.float32))
    print(f"\nwrote {out}")

    print("\nheld-out eval per-fire AUC (onset vs pre-ignition):")
    print(f"  {'fire':<20} {'conf-alone':>11} {'fused':>8}")
    for f in sorted(set(seqev)):
        mk = seqev == f
        if len(set(yev[mk])) < 2:
            continue
        print(f"  {f.split('_')[1][:18]:<20} {_auc(Xev[mk,0], yev[mk]):>11.3f} "
              f"{_auc(fused_ev[mk], yev[mk]):>8.3f}")
    print(f"  {'POOLED':<20} {_auc(Xev[:,0], yev):>11.3f} {_auc(fused_ev, yev):>8.3f}")
    print("\nnext: TTD on the fused score vs appearance alone --")
    print("  python src/models/figlib_ttd.py --eval-only --features data/figlib/features_evalfires_fused.npz --tag _evalfused")


if __name__ == "__main__":
    main()
