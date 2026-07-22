# Hard-negative mining — findings

**Status: proof scale, directional.** Same underfit regime as the baseline (yolo11n, 15
epochs, ~20% base data). The question was narrow: does emphasizing the frames the baseline
false-alarms on actually pull down the false-alarm rate on held-out clean frames? It does,
substantially.

In the [recall-first framing](metrics.md), this is the lever that lowers the **false-alarm
burden** (FP/camera/day) at a held detection rate — the complement to resolution, which raises
the detection *ceiling*. F1 barely moves here, which is exactly the point: F1 is the wrong metric
for this domain (it weights a missed fire like a false alarm), so a change that halves the
false-alarm burden while holding recall barely registers in F1.

## What was mined

Running the baseline over the 3,852 clean (no-smoke) frames from the **training sites only**
(never val/test — that would leak): **2,305 of them, 59.8%, drew a false alarm** at conf 0.05.
The majority of clean training frames fool the single-frame model. Those 2,305 hard negatives
were oversampled 3× and added to a fixed 20% base sample (11,705 training images total, ~60%
of them now hard negatives), and the model was retrained from scratch.

## Result: the false-alarm rate on clean frames roughly halved

Held-out sites (marguerite + serre-de-barre), best-F1 operating point (conf 0.05):

| | baseline | hard-neg | change |
|---|---|---|---|
| precision (aggregate) | 0.938 | 0.967 | +0.029 |
| recall | 0.676 | 0.632 | −0.044 |
| F1 | 0.786 | 0.765 | −0.021 |
| **false alarms on clean frames** | **42.0%** | **20.0%** | **−22 pts (halved)** |

The aggregate F1 barely moved (and dipped slightly, because recall traded down a little), which
is exactly why F1 is the wrong headline for this problem. The metric that matters —
false alarms on frames containing no smoke — was cut in half. At conf 0.30 the effect is even
sharper: **8% → 2%**, a 4× reduction.

## Result: deployment-realistic precision roughly doubled

Recomputing precision at realistic base rates (smoke is rare in the field):

| operating point | precision @ 1% base rate — baseline | hard-neg |
|---|---|---|
| conf 0.05 | 1.6% | 3.1% |
| conf 0.30 | 3.7% | 11.3% |
| conf 0.50 | 10.7% | 15.9% |

Still low in absolute terms — these are underfit proof models and the base-rate math is
unforgiving — but the mining **roughly doubled** precision at every operating point, purely by
teaching the model what a false alarm looks like. That is the intended, literature-backed effect
(SKLFS separable negative sampling / OHEM), reproduced here.

## Honest caveats

- **Not a perfectly controlled A/B.** The baseline trained on Ultralytics' internal random 20%
  sample; the hard-neg model on a fixed seeded 20% base + hard negatives. A fully rigorous
  comparison would retrain the baseline on the identical base sample. The effect size (halving)
  is far larger than that sampling difference would plausibly explain, but the clean A/B is a
  follow-up.
- **Recall dropped** (0.676 → 0.632). The model became more conservative. Whether that trade is
  acceptable depends on the operating point and the human-review stage downstream — which is
  exactly the decision the PR curve exists to inform.
- Proof scale throughout. Direction is trustworthy; absolute numbers are not final.

## What this sets up

The false-alarm floor is now lower but still real, and recall gave a little ground. The
literature pointed to a **temporal model** as the next fix (SmokeyNet's +26 precision points
from frame-to-frame context). We built it — and on pyro-sdis it did **not** beat the
single-frame detector at matched recall. [The temporal findings](temporal-findings.md) explain
why: the remaining false alarms are *persistent* structures (76% of them), not the flicker a
temporal model suppresses, and this dataset's short bursts lack the ignition-onset dynamics
that make temporal context pay off on FIgLib. So hard-negative mining — teaching the detector
what a false alarm looks like — turned out to be the move that actually attacked the persistent
confusers, while the time axis did not. That is the honest ordering of what worked here.

## Reproduce

    python src/models/mine_hard_negatives.py
    python src/models/train.py --split grouped_hardneg --epochs 15 --name grouped_hardneg
    python src/models/evaluate.py --weights runs/grouped_hardneg/weights/best.pt --split grouped_hardneg

Ranked mined frames: `results/hard_negatives.csv`. Eval sweeps: `results/eval_grouped_*_test.json`.
