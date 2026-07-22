"""FIgLib positive control: does temporal context help where onset IS present?

The pyro-sdis result was a negative one -- temporal context did not beat a single
frame, because that dataset's short bursts capture already-established scenes and its
confusers are persistent. The claim we made was mechanistic: temporal helps *when the
data contains ignition onset against a clean background*, and pyro-sdis mostly lacks it.

FIgLib is the direct test of that claim. Every sequence spans ~40 min before to ~40 min
after a fire's first visible plume, one frame per minute -- exactly the onset dynamics
pyro-sdis is missing. The filename encodes the signed offset from plume appearance
(`<unixts>_<+/-offset_seconds>.jpg`), so labels are free: offset >= 0 is smoke.

We run the IDENTICAL pipeline as pyro-sdis -- same frozen detector as feature
extractor, same GRU head, same matched-recall comparison -- changing only the dataset.
If temporal now wins, the mechanism is confirmed both directions.

Split is by FIRE (whole sequences held out), the leak-safe analogue of site-holdout.

    python src/models/figlib_temporal.py --extract   # once, caches features
    python src/models/figlib_temporal.py             # train + compare

Caveat: the detector is pyro-sdis-trained, so this is also a zero-shot distribution transfer
(French towers -> California). We first check the single-frame signal is usable (AUC well
above 0.5); if it were not, a null temporal result would be confounded by distribution shift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
FIGLIB = ROOT / "data" / "figlib"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from models.temporal import make_windows, run  # noqa: E402
from models.compare_temporal import burst_rolling_min, fpr_at_recall, prec_at  # noqa: E402

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def scan_frames() -> pd.DataFrame:
    rows = []
    for seq_dir in sorted((FIGLIB / "images").glob("*/")):
        if not seq_dir.is_dir():
            continue
        for jpg in seq_dir.glob("*.jpg"):
            m = OFFSET_RE.search(jpg.name)
            if not m:
                continue
            offset = int(m.group(1))
            rows.append({"path": str(jpg), "stem": jpg.stem, "seq": seq_dir.name,
                         "offset": offset, "smoke": offset >= 0})
    df = pd.DataFrame(rows)
    # order within a sequence by offset (== chronological)
    return df.sort_values(["seq", "offset"]).reset_index(drop=True)


def extract_features(df: pd.DataFrame, imgsz: int = 640, batch: int = 64) -> None:
    from ultralytics import YOLO
    device = pick_device()
    model = YOLO(str(ROOT / "runs/grouped_proof/weights/best.pt"))
    paths = df["path"].tolist()
    print(f"extracting {len(paths)} FIgLib embeddings on {device}")
    feats = np.zeros((len(paths), 256), dtype=np.float32)
    confs = np.zeros(len(paths), dtype=np.float32)
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        emb = model.embed(chunk, device=device, imgsz=imgsz, verbose=False)
        det = model.predict(chunk, device=device, imgsz=imgsz, conf=0.001, verbose=False)
        for k in range(len(chunk)):
            feats[i + k] = emb[k].detach().cpu().numpy().astype(np.float32)
            c = det[k].boxes.conf
            confs[i + k] = float(c.max()) if len(c) else 0.0
        if (i // batch) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(paths)}", flush=True)
    out = FIGLIB / "features.npz"
    np.savez_compressed(out, stems=df["stem"].to_numpy(), feats=feats, confs=confs)
    print(f"saved -> {out}")


def split_by_fire(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    seqs = sorted(df["seq"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(seqs)
    n = len(seqs)
    n_test = max(2, round(n * 0.20))
    n_val = max(1, round(n * 0.15))
    test, val = set(seqs[:n_test]), set(seqs[n_test : n_test + n_val])
    df = df.copy()
    df["split"] = np.where(df["seq"].isin(test), "test",
                           np.where(df["seq"].isin(val), "val", "train"))
    return df


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores)) if len(set(y)) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true", help="run detector + cache features")
    ap.add_argument("--features", default=str(FIGLIB / "features.npz"),
                    help="feature archive to use (e.g. features_tiled_embed.npz)")
    ap.add_argument("--tag", default="", help="suffix for the output json (e.g. _tiled)")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--conf-only", action="store_true", help="GRU over conf alone (drop embedding)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = scan_frames()
    print(f"FIgLib: {df['seq'].nunique()} sequences, {len(df)} frames "
          f"({int(df['smoke'].sum())} smoke / {int((~df['smoke']).sum())} clean)")

    if args.extract:
        extract_features(df)
        return

    print(f"features: {args.features}")
    arch = np.load(args.features, allow_pickle=True)
    idx = {s: i for i, s in enumerate(arch["stems"].astype(str))}
    df = df[df["stem"].map(lambda s: s in idx)].reset_index(drop=True)
    rows = df["stem"].map(idx).to_numpy()
    feats, confs = arch["feats"][rows], arch["confs"][rows]
    df["conf_raw"] = confs
    # feat = embedding + conf, matching the pyro-sdis pipeline; "burst" == fire sequence
    df["burst"] = df["seq"]
    df["ts"] = df["offset"]  # offset is the within-sequence time axis
    df = split_by_fire(df, seed=args.seed)

    # single-frame sanity: is the zero-shot detector signal usable at all?
    te_mask = df["split"] == "test"
    sf_auc = auc(df.loc[te_mask, "conf_raw"].to_numpy(), df.loc[te_mask, "smoke"].astype(int).to_numpy())
    print(f"\nzero-shot single-frame detector AUC on FIgLib test: {sf_auc:.3f}")
    print(f"  split: train {(df.split=='train').sum()}  val {(df.split=='val').sum()}  "
          f"test {(df.split=='test').sum()} frames")

    device = pick_device()
    # standardize on train frames only (leak-safe)
    full = np.concatenate([feats, confs[:, None]], axis=1).astype(np.float32)
    if args.conf_only:
        full = confs[:, None].astype(np.float32)
        print("  conf-only GRU: 1-d per-frame feature")
    df["feat"] = list(full)
    tr_feat = np.stack(df.loc[df.split == "train", "feat"].to_numpy())
    mu, sigma = tr_feat.mean(0), tr_feat.std(0) + 1e-6
    dfn = df.copy()
    dfn["feat"] = dfn["feat"].map(lambda v: ((v - mu) / sigma).astype(np.float32))

    arrays = {p: make_windows(dfn[dfn.split == p], args.window) for p in ("train", "val", "test")}
    for p in ("train", "val", "test"):
        y = arrays[p][1]
        print(f"  {p:5s}: {len(y):5d} windows  {int(y.sum())} pos  {int((y==0).sum())} neg")
    s_gru = run(arrays, epochs=args.epochs, lr=1e-3, device=device, seed=args.seed)

    te = df[te_mask].copy()
    y = te["smoke"].astype(int).to_numpy()
    s_single = te["conf_raw"].to_numpy()
    s_persist_full = burst_rolling_min(df, args.window)
    stem_to_i = {s: i for i, s in enumerate(df["stem"].to_numpy())}
    s_persist = np.array([s_persist_full[stem_to_i[s]] for s in te["stem"]])

    methods = {"single-frame": s_single, "persistence": s_persist, "temporal-gru": s_gru}
    targets = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]

    print("\nFALSE-ALARM RATE on clean frames at matched recall (lower is better):\n")
    header = "recall  " + "  ".join(f"{m:>13s}" for m in methods)
    print(header)
    table = {"single_frame_auc": round(sf_auc, 4), "n_sequences": int(df["seq"].nunique()),
             "targets": targets, "methods": {m: {} for m in methods}}
    for tr in targets:
        cells = []
        for m, s in methods.items():
            fpr = fpr_at_recall(s, y, tr)
            table["methods"][m][f"recall_{tr}"] = None if np.isnan(fpr) else round(float(fpr), 4)
            cells.append(f"{fpr*100:12.1f}%" if not np.isnan(fpr) else f"{'n/a':>13s}")
        print(f"{tr:.2f}  " + "  ".join(cells))

    print("\nPRECISION @ 1% base rate at matched recall (higher is better):\n")
    print(header)
    for tr in targets:
        cells = []
        for m, s in methods.items():
            fpr = fpr_at_recall(s, y, tr)
            pr = prec_at(tr, fpr, 0.01)
            cells.append(f"{pr*100:12.2f}%" if not np.isnan(pr) else f"{'n/a':>13s}")
        print(f"{tr:.2f}  " + "  ".join(cells))

    out = RESULTS / f"figlib_temporal_comparison{args.tag}.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
