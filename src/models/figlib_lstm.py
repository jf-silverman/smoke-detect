"""Motion temporal channel v2 — a leak-free CAUSAL LSTM over [conf + anchored motion].

The per-frame logistic fusion ([figlib_fusion.py](figlib_fusion.py)) underdelivered: a *per-frame*
linear combiner leaned almost entirely on appearance (conf weight +0.77 vs motion <=0.18), its training
leave-one-fire-out AUC dropped below conf-alone (0.623 vs 0.689), and its only real lift was a noisy
operational one. The diagnosis was mechanistic: the onset signal that lowers time-to-detection lives in
the *temporal transition* (the plume appearing and growing over successive frames), which a per-frame
model cannot see by construction. This tests that diagnosis with the smallest model that CAN see it — a
one-layer causal LSTM over the same four features, run left-to-right so frame t only ever sees frames
<= t (a legitimate online detector; NEVER bidirectional, or TTD would peek at the future).

LEAK-FREE, identical to the per-frame fusion: appearance `conf` is the zero-shot base detector
(gcp_grouped_1280, never trained on FIgLib) and the motion features are training-free. The LSTM is fit
on the 17 training fires and applied to the 6 held-out EVAL_FIRES; the per-timestep onset probability is
written as a `conf_tiled` npz so the existing TTD harness scores it unchanged. The payoff test is TTD vs
the base detector and vs the per-frame fusion -- does *sequence* modeling extract what the linear
combiner could not?

RESULT (2026-07-28): this is a documented NEGATIVE — the apparent win is a temporal-POSITION LEAK, not
motion signal. Unablated, the LSTM looks great (train LOO AUC 0.858 vs conf 0.689; several eval fires at
1.000; TTD 0.97 min). But FIgLib sorts each fire by offset with a monotonic step label centered on
ignition, so a sequence model can score near-perfectly by learning "elapsed frames -> onset" without ever
using the features. The built-in controls prove it: `ABLATE=zero` (all features zeroed) still gives LOO
0.994 / eval-pooled 1.000 -- pure positional prior; `ABLATE=shuffle` (time order permuted) collapses the
LSTM to LOO 0.612 (= the per-frame LR's 0.623) and eval-pooled 0.499 (chance). So a sequence model's
leak-free ceiling here is no better than the per-frame combiner, and FIgLib's fixed onset-centered
structure cannot validly evaluate a temporal model for TTD. A real temporal test needs onset-position-
randomized or continuous-feed data (see backlog). Kept as the reproducible demonstration of the leak.

    # prereqs (same caches the per-frame fusion consumes):
    #   figlib_tiled.py --exclude-eval ... --out features_trainfires_base.npz
    #   figlib_tiled.py --eval-only    ... --out features_evalfires_base.npz
    #   figlib_ttd.py --exclude-eval --motion            (writes motion_feats.npz)
    #   figlib_ttd.py --eval-only --motion --motion-cache data/figlib/motion_feats_eval.npz ...
    python src/models/figlib_lstm.py                    # unablated (leaked) run
    ABLATE=zero    python src/models/figlib_lstm.py     # control: pure positional prior -> ~0.99
    ABLATE=shuffle python src/models/figlib_lstm.py     # control: position destroyed -> collapses to LR
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
FIGLIB = ROOT / "data" / "figlib"
sys.path.insert(0, str(ROOT / "src"))
from models.figlib_fusion import FEATURES, build_xy  # noqa: E402  (same feature assembly as v1)
from models.figlib_ttd import NIGHT_PATTERN  # noqa: E402

SEED = 0
HIDDEN = 24
EPOCHS = 300
LR = 0.01


def pick_device() -> str:
    # small model + tiny sequences: CPU is deterministic and as fast here; avoids MPS RNN quirks
    return "cpu"


# --- ablation controls (ABLATE env var) -- to catch a temporal-POSITION leak -------------------
# FIgLib sorts each fire by offset and the label is a perfect step function (pre-ignition then onset),
# every sequence centered on ignition. A sequence model can score high just by learning "elapsed frames
# -> onset" (position), never touching the smoke features -- an artifact FIgLib's fixed structure cannot
# falsify. These controls quantify how much of the LSTM lift is that leak:
#   zero        -> all features 0: any AUC>0.5 is PURE positional prior (the smoking gun)
#   shuffle     -> permute time order within each fire (features+labels travel together): destroys the
#                  position->label monotonicity, keeps per-frame content. AUC collapse => position-reliant
#   conf_only   -> keep appearance, zero motion
#   motion_only -> keep motion, zero appearance
ABLATE = os.environ.get("ABLATE", "none")


def _ablate_features(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    if ABLATE == "zero":
        X[:] = 0.0
    elif ABLATE == "conf_only":
        X[:, 1:] = 0.0
    elif ABLATE == "motion_only":
        X[:, 0] = 0.0
    return X


def to_sequences(X, y, stems, seqs, offs):
    """Group aligned frames into one chronological sequence per fire (sorted by onset offset).

    Under ABLATE=shuffle the within-fire time order is permuted (features+labels together) so absolute
    position no longer predicts the label; all other modes keep chronological order.
    """
    rng = np.random.default_rng(SEED)
    out = {}
    for s in sorted(set(seqs)):
        m = seqs == s
        order = np.argsort(offs[m])
        if ABLATE == "shuffle":
            order = rng.permutation(order)
        out[s] = (X[m][order], y[m][order], stems[m][order], offs[m][order])
    return out


class OnsetLSTM(nn.Module):
    def __init__(self, n_feat: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=True)  # unidirectional = causal
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: [B, T, F] -> logits [B, T]
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


def _auc(score, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, score) if len(set(y.tolist())) == 2 else float("nan")


def train_model(train_seqs, scaler, device, seed=SEED):
    """Fit a causal LSTM on the given {fire: (X,y,...)} dict; return the trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OnsetLSTM(len(FEATURES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # class weight from the pooled train frames (onset is the minority pre-ignition-heavy split varies)
    ally = np.concatenate([y for _, (_, y, _, _) in train_seqs.items()])
    pos_w = torch.tensor([(ally == 0).sum() / max(1, (ally == 1).sum())], dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    tensors = {s: (torch.tensor(scaler.transform(X), dtype=torch.float32, device=device).unsqueeze(0),
                   torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0))
               for s, (X, y, _, _) in train_seqs.items()}
    order = list(tensors)
    model.train()
    for _ in range(EPOCHS):
        np.random.shuffle(order)
        for s in order:
            xb, yb = tensors[s]
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    return model


@torch.no_grad()
def score_seq(model, X, scaler, device):
    """Per-timestep onset probability for one fire's chronological frames (causal)."""
    model.eval()
    xb = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device).unsqueeze(0)
    return torch.sigmoid(model(xb)).squeeze(0).cpu().numpy()


def main() -> None:
    from sklearn.preprocessing import StandardScaler

    device = pick_device()
    Xtr, ytr, stmtr, seqtr, offtr = build_xy(FIGLIB / "features_trainfires_base.npz",
                                             FIGLIB / "motion_feats.npz")
    keep = np.array([re.search(NIGHT_PATTERN, s, re.I) is None for s in seqtr])
    Xtr, ytr, stmtr, seqtr, offtr = Xtr[keep], ytr[keep], stmtr[keep], seqtr[keep], offtr[keep]
    Xev, yev, stmev, seqev, offev = build_xy(FIGLIB / "features_evalfires_base.npz",
                                             FIGLIB / "motion_feats_eval.npz")
    Xtr, Xev = _ablate_features(Xtr), _ablate_features(Xev)
    print(f"train: {len(Xtr)} frames / {len(set(seqtr))} fires; eval: {len(Xev)} frames / "
          f"{len(set(seqev))} fires; device={device}; ABLATE={ABLATE}")

    train_seqs = to_sequences(Xtr, ytr, stmtr, seqtr, offtr)
    eval_seqs = to_sequences(Xev, yev, stmev, seqev, offev)

    # ---- leave-one-fire-out on the training fires: a leak-aware read, comparable to the LR (0.623) ----
    loo_lstm, loo_conf = [], []
    for held in sorted(train_seqs):
        tr = {s: v for s, v in train_seqs.items() if s != held}
        sc = StandardScaler().fit(np.concatenate([v[0] for v in tr.values()]))
        m = train_model(tr, sc, device)
        Xh, yh, _, _ = train_seqs[held]
        if len(set(yh.tolist())) == 2:
            loo_lstm.append(_auc(score_seq(m, Xh, sc, device), yh))
            loo_conf.append(_auc(Xh[:, 0], yh))
    print(f"\ntrain leave-one-fire-out per-fire AUC: LSTM {np.nanmean(loo_lstm):.3f}  "
          f"(conf-alone {np.nanmean(loo_conf):.3f};  per-frame LR was 0.623)")

    # ---- fit on all 17 train fires, apply causally to the 6 held-out eval fires ----
    scaler = StandardScaler().fit(Xtr)
    model = train_model(train_seqs, scaler, device)

    stems_out, score_out = [], []
    print("\nheld-out eval per-fire AUC (onset vs pre-ignition):")
    print(f"  {'fire':<20} {'conf-alone':>11} {'LSTM':>8}")
    per_fire = []
    for f in sorted(eval_seqs):
        X, y, stems, _ = eval_seqs[f]
        p = score_seq(model, X, scaler, device)
        stems_out.extend(stems.tolist()); score_out.extend(p.tolist())
        if len(set(y.tolist())) == 2:
            ca, la = _auc(X[:, 0], y), _auc(p, y)
            per_fire.append((f, ca, la))
            print(f"  {f.split('_')[1][:18]:<20} {ca:>11.3f} {la:>8.3f}")
    # pooled AUC over all eval frames
    allp = np.array(score_out); allc = Xev[:, 0]
    print(f"  {'POOLED':<20} {_auc(allc, yev):>11.3f} {_auc(allp, yev):>8.3f}")

    if ABLATE == "none":
        out = FIGLIB / "features_evalfires_lstm.npz"
        np.savez_compressed(out, stems=np.array(stems_out), conf_tiled=np.array(score_out, np.float32))
        print(f"\nwrote {out}")
        print("next: python src/models/figlib_ttd.py --eval-only "
              "--features data/figlib/features_evalfires_lstm.npz --tag _evallstm")
    else:
        print(f"\n(ABLATE={ABLATE}: control run, npz not written)")


if __name__ == "__main__":
    main()
