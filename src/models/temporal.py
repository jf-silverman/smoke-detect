"""A temporal smoke classifier -- the differentiator over the single-frame detector.

The single-frame baseline (and even the hard-negative-mined version) false-alarms on
static haze, fog banks and glare because *one frame* cannot tell a motionless grey
patch from a growing plume. The literature's central result (SmokeyNet: +26 precision
points from frame-to-frame context) is that the *time axis* resolves exactly this: a
plume appears, spreads and drifts; haze does not.

pyro-sdis turns out to support this directly. Frames arrive in bursts -- ~29 s apart,
contiguous from one camera -- and 720 of 1,182 bursts contain the smoke *onset*
(clear frames, then the plume appears). That is the signal this model learns.

Design (deliberately lightweight, and clear about it):

  * The frozen BASELINE detector is a per-frame feature extractor (see
    extract_features.py). We never fine-tune it here, so any gain is attributable to
    temporal context over the *same* features the single-frame model saw.
  * For each frame t we build a CAUSAL window [t-W+1 .. t] within its burst -- only
    past and present frames, never the future. That is the real deployment decision:
    "given everything up to now, alarm?"
  * A small GRU consumes the window; its final state -> one logit -> alarm score.
  * SAME site-grouped split (same held-out towers) and SAME operator-framed metrics
    as the baseline, so the comparison is apples-to-apples.

    python src/models/temporal.py --epochs 40

Writes results/eval_temporal_test.json in the exact schema evaluate.py uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from data.splits import grouped_split, site_of  # noqa: E402
from models.evaluate import base_rate_table, sweep  # noqa: E402

BURST_GAP_S = 900  # >15 min gap starts a new burst (matches the data analysis)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- data


def load_frames() -> pd.DataFrame:
    """Per-frame table with features, label, site, split and burst id."""
    meta = pd.read_parquet(PROC / "meta.parquet")
    meta["stem"] = meta["image_name"].map(lambda n: Path(n).stem)
    meta["ts"] = pd.to_datetime(meta["date"], format="%Y-%m-%dT%H-%M-%S")
    meta["smoke"] = meta["annotations"].astype(str).str.strip().astype(bool)

    # site-grouped split -- identical held-out towers as the detector baseline
    split_df, report = grouped_split(meta)
    print(report)
    stem_split = dict(zip(split_df["image_name"].map(lambda n: Path(n).stem), split_df["split"]))
    meta["split"] = meta["stem"].map(stem_split)
    meta["site"] = meta["camera"].map(site_of)

    # cached embeddings
    arch = np.load(PROC / "features_proof.npz", allow_pickle=True)
    idx = {s: i for i, s in enumerate(arch["stems"].astype(str))}
    keep = meta["stem"].map(lambda s: s in idx)
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} frames with no cached feature")
        meta = meta[keep].copy()
    rows = meta["stem"].map(idx).to_numpy()
    feats = arch["feats"][rows]
    confs = arch["confs"][rows]
    # append the single-frame confidence as one extra feature: a strong prior the
    # temporal head is free to keep, override, or contextualize.
    meta["feat"] = list(np.concatenate([feats, confs[:, None]], axis=1).astype(np.float32))

    # bursts: contiguous frames per camera within BURST_GAP_S
    meta = meta.sort_values(["camera", "ts"]).reset_index(drop=True)
    gap = meta.groupby("camera")["ts"].diff().dt.total_seconds()
    new = (gap > BURST_GAP_S) | gap.isna()
    meta["burst"] = meta["camera"].astype(str) + "#" + new.groupby(meta["camera"]).cumsum().astype(str)
    return meta


def make_windows(df: pd.DataFrame, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal windows: for frame t, stack features of [t-W+1 .. t] within its burst.

    Frames near a burst's start are left-padded by repeating the earliest available
    frame, so every frame yields exactly one W-length window.
    """
    X, y = [], []
    for _, g in df.groupby("burst", sort=False):
        g = g.sort_values("ts")
        f = np.stack(g["feat"].to_numpy())  # (n, D)
        labels = g["smoke"].to_numpy().astype(np.float32)
        n = len(g)
        for t in range(n):
            lo = t - W + 1
            if lo < 0:
                pad = np.repeat(f[0:1], -lo, axis=0)
                win = np.concatenate([pad, f[0 : t + 1]], axis=0)
            else:
                win = f[lo : t + 1]
            X.append(win)
            y.append(labels[t])
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


# -------------------------------------------------------------------------- model


class TemporalHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (B, W, D)
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)  # decision at frame t


def run(split_arrays, *, epochs: int, lr: float, device: str, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    Xtr, ytr = split_arrays["train"]
    Xva, yva = split_arrays["val"]
    Xte, yte = split_arrays["test"]
    d_in = Xtr.shape[-1]

    model = TemporalHead(d_in).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # emphasise the rare negatives (smoke-heavy corpus) so the head learns to WITHHOLD
    n_pos, n_neg = float(ytr.sum()), float((ytr == 0).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = torch.utils.data.DataLoader(tr, batch_size=256, shuffle=True)
    Xva_t, yva_t = torch.from_numpy(Xva).to(device), torch.from_numpy(yva).to(device)

    best_val, best_state, patience, bad = 1e9, None, 8, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_t), yva_t).item()
        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  val_loss {vloss:.4f}  best {best_val:.4f}")
        if bad >= patience:
            print(f"  early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(Xte).to(device))).cpu().numpy()
    return scores


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=8, help="causal frames incl. present")
    ap.add_argument("--conf-only", action="store_true", help="use only the detector conf feature")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "eval_temporal_test.json"))
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}\nloading frames + features ...")
    df = load_frames()
    if args.conf_only:
        # keep only the detector-confidence dimension (last column): isolates whether
        # the TEMPORAL TRAJECTORY of evidence beats a single frame, without the noisy
        # global embedding the GRU can't use.
        df["feat"] = df["feat"].map(lambda v: v[-1:].astype(np.float32))
        print("  conf-only: 1-d per-frame feature")

    # standardize features using TRAIN frames only (leak-safe); unnormalized 256-d
    # embeddings otherwise swamp the GRU and it collapses to the majority class.
    train_feat = np.stack(df.loc[df["split"] == "train", "feat"].to_numpy())
    mu = train_feat.mean(axis=0)
    sigma = train_feat.std(axis=0) + 1e-6
    df["feat"] = df["feat"].map(lambda v: ((v - mu) / sigma).astype(np.float32))

    print(f"\nbuilding causal windows (W={args.window}) ...")
    arrays = {}
    for part in ("train", "val", "test"):
        sub = df[df["split"] == part]
        X, y = make_windows(sub, args.window)
        arrays[part] = (X, y)
        print(f"  {part:5s}: {len(y):6d} windows  {int(y.sum()):6d} pos  {int((y==0).sum()):6d} neg")

    print("\ntraining temporal head ...")
    scores = run(arrays, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    y_true = arrays["test"][1].astype(int)
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = sweep(scores, y_true, thresholds)
    best = max(rows, key=lambda r: r["f1"])
    br = base_rate_table(rows)
    test_p = float(y_true.mean())

    print(f"\noperating point (best-F1 @ score={best['threshold']}):")
    print(f"  precision {best['precision']:.3f}  recall {best['recall']:.3f}  F1 {best['f1']:.3f}")
    print(f"  false alarms on {int((y_true==0).sum())} clean frames: "
          f"{best['false_alarms_on_negatives']} "
          f"({best['false_alarm_rate_on_negatives']*100:.1f}%)")
    print("\n  score  recall   FPR   | prec@5%  prec@1%  prec@0.5%")
    for e in br:
        if e["threshold"] not in (0.05, 0.10, 0.20, 0.30, 0.50):
            continue
        print(f"  {e['threshold']:.2f}   {e['recall']:.3f}  {e['fpr_on_negatives']:.3f} | "
              f"{e['precision@0.05']*100:6.1f}%  {e['precision@0.01']*100:6.1f}%  "
              f"{e['precision@0.005']*100:6.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "temporal-gru", "window": args.window,
        "n_images": len(y_true), "n_positive": int(y_true.sum()),
        "test_base_rate": round(test_p, 4), "best": best,
        "sweep": rows, "base_rate_corrected": br}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
