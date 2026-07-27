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

## Caveats

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
confusers, while the time axis did not. That is the actual ordering of what worked here.

## Full-scale, native resolution: the two levers combined

**Status: full-scale, and the answer to "does it hold at scale, combined with resolution?"** The
proof above was 640-px and 20% data. Here we mined with the **converged full-scale 1280 model**
itself (native resolution), then retrained at 1280 on full data — the first time the project's two
proven levers (native-resolution training + hard-negative mining) ran *together*.

A finding fell out of the mining step alone: the converged full-scale model **still false-alarms on
57.6%** of clean training frames — barely below the underfit baseline's 59.8%. Resolution and
convergence raise the recall ceiling but do essentially nothing to the confusers, so mining still
has a job to do. We added the 2,217 mined hard negatives at a deliberately **gentle ~22% share**
(the proof run's ~60% share halved false alarms but cost recall; recall is king here, so we traded
a smaller false-alarm cut for a smaller recall hit).

Held-out towers, vs native-resolution training alone (both @1280):

| | native-res only | + hard-neg (combined) | change |
|---|---:|---:|---:|
| max POD (recall ceiling) | 0.827 | 0.803 | −0.024 |
| FP/camera/day @ max POD | ~173 | **~133** | **−23%** |
| precision @ 1% base rate | 2.3% | **2.9%** | +0.6 pt |
| REV @ C/L=0.01 | +0.48 | **+0.54** | better |
| REV @ C/L=0.002 (harshest) | −0.22 | −0.26 | slightly worse |

False-alarm rate on clean frames **at matched recall** (the fair test):

| recall | native-res only | combined | change |
|---|---:|---:|---:|
| 0.56 | 11.0% | 10.6% | −0.4 pt |
| 0.61 | 12.5% | 11.2% | −1.2 pt |
| 0.66 | 17.3% | 13.7% | **−3.5 pt** |
| 0.72 | 20.0% | 18.1% | −1.9 pt |

**Read:** mining still lowers the false-alarm burden at full scale and native resolution — the
operating-point burden falls ~23% (173 → 133 FP/camera/day), matched-recall false alarms drop
0.4–3.5 pts, and deployment precision rises 2.3% → 2.9%. The effect is **gentler than the proof
run's halving**, by design: the light 22% share protects the recall ceiling (only −2.4 pts, vs the
proof's −4.4). It is a real, incremental win in the moderate-cost regime (best REV at C/L=0.01),
but not a silver bullet — 133 FP/camera/day is still far from Pano's < 1 target, and in the
harshest miss-averse regime the small recall loss slightly outweighs the false-alarm saving. The
levers combine, but they do not close the deployability gap on their own. See [metrics.md](metrics.md).

*Caveat:* the combined run was stopped at epoch 24 (val plateaued, mAP50 0.699) against the
no-hard-neg run's full 40 (peak 0.707 at epoch 23) — both plateaued, so near-matched, not identical.

## Reproduce

    # proof scale (640, 20% data)
    python src/models/mine_hard_negatives.py
    python src/models/train.py --split grouped_hardneg --epochs 15 --name grouped_hardneg
    python src/models/evaluate.py --weights runs/grouped_hardneg/weights/best.pt --split grouped_hardneg

    # full scale + native resolution, combined (mine with the converged 1280 model)
    python src/models/mine_hard_negatives.py --weights runs/grouped_full_1280/weights/best.pt \
        --imgsz 1280 --base-frac 1.0 --oversample 3 --tag _1280full
    python src/models/train.py --split grouped_hardneg_1280full --imgsz 1280 --batch 8 --epochs 40 \
        --name grouped_hardneg_1280full
    python src/models/evaluate.py --weights runs/grouped_hardneg_1280full/weights/best.pt \
        --split grouped_hardneg_1280full --imgsz 1280 --target-pod 0.90 \
        --out results/eval_grouped_hardneg_1280full_test.json

Ranked mined frames: `results/hard_negatives.csv`. Eval sweeps: `results/eval_grouped_*_test.json`.
