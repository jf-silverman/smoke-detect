# Baseline findings — single-frame YOLO on pyro-sdis

**Status: proof run, deliberately underfit.** yolo11n, 15 epochs, 20% of the training
data (~4,800 images), ~55 min on Apple MPS. These numbers are directional, not final —
the point was to validate the pipeline and see whether the predicted failure modes appear.
They do, cleanly.

## Setup

- Train/val/test are **site-disjoint** (grouped split). Test sites = `marguerite` +
  `serre-de-barre`, seen in neither training nor validation.
- Evaluated the way the field judges a detector — **recall-first**, not F1. A missed fire is
  catastrophic; a false alarm costs a watchstander a glance. So the headline is **detection rate
  (<abbr title="Probability of detection — the fraction of real fires the detector catches; equal to recall">POD</abbr>)** and the **false-alarm burden**, with F1 demoted to context. See
  [metrics.md](metrics.md) for the full rationale and field grounding.

## Result 1 — the model detects smoke on unseen sites

On held-out sites, the recall-first view (max reachable detection rate, and its false-alarm
burden):

| max POD (detection rate) | FP/camera/day @ max POD* | <abbr title="False-alarm ratio — FP/(FP+TP) = 1 − precision; the share of raised alarms that are wrong">FAR</abbr> |
|---|---|---|
| 0.676 | ~208 | 0.062 |

<sub>*assumed 1% base rate, 500 frames/camera/day — extrapolation, see [metrics.md](metrics.md).
Pano's operational target is < 1 FP/camera/day.</sub>

An underfit proof model reaching POD 0.68 on **sites it has never seen** is a legitimately
encouraging baseline (in the range of the ~0.70 F1 PyroNear reports on their hard in-the-wild
benchmark). But POD caps at 0.68 here — the model *cannot* be pushed to the ≥0.90 detection rate
this domain wants, because it never detects the small plumes ([resolution](resolution-findings.md)
raises that ceiling). *(For reference, the best-F1 point is conf 0.05: precision 0.938, recall
0.676, F1 0.786 — but F1 weights a missed fire like a false alarm, which this domain does not.)*

## Result 2 — the false-alarm burden (the alarm-fatigue constraint)

The 0.938 precision above is **an artifact of the test set's 90%-positive base rate** — not a
field number. The real signal is the burden: at that operating point, **42% of the clean,
no-smoke frames triggered a false alarm** — the single-frame model firing on clouds, fog, and
glare, exactly as the literature predicts. This is not a verdict that the model is failing; it is
the *alarm-fatigue constraint* — how often a watchstander gets pinged — which is what governs
whether a high-recall detector is actually reviewable.

Real detection towers see smoke on a tiny fraction of frames. Recomputing precision from the
measured recall (TPR) and false-alarm-rate (FPR) at realistic base rates:

| conf | recall | FA-rate on clean frames | precision @ 5% | @ 1% | @ 0.5% |
|---|---|---|---|---|---|
| 0.05 (best-F1) | 0.676 | 42.0% | 7.8% | **1.6%** | 0.8% |
| 0.10 | 0.575 | 28.9% | 9.5% | 2.0% | 1.0% |
| 0.30 | 0.319 | 8.3% | 16.8% | 3.7% | 1.9% |
| 0.50 | 0.099 | 0.8% | 38.5% | 10.7% | 5.6% |

**At a 1% deployment base rate, the naive 94% precision would be ~2%.** This is the benchmark-vs-
field gap the state-of-the-art report warned about, reproduced on our own model — and it is why a
a headline of 94% precision is meaningless here. Framed correctly (see [metrics.md](metrics.md)),
this is the alarm-fatigue constraint, and the relative-economic-value view makes the verdict
sharp: at the cost-loss ratios where misses dominate, this detector adds only marginal value —
it misses ~32% of fires *and* over-alarms — so both levers below are needed.

## What this tells us to do next

1. **Report recall-first**, not F1 or aggregate precision: POD, the max-POD ceiling, false-alarm
   burden as FP/camera/day, and relative economic value across cost-loss ratios (all in the eval
   JSON). See [metrics.md](metrics.md).
2. **Raise the detection ceiling.** POD caps at 0.68 because the model misses small plumes;
   native-resolution inference/training lifts it to 0.86 ([resolution](resolution-findings.md)).
3. **Lower the false-alarm burden.** The burden is set by the false-alarm rate on clean frames,
   so hard-negative mining (curating the clouds/fog/glare it fires on) directly attacks the ~208
   FP/camera/day. The [confuser corpus](confuser-corpus.md) shows 74% of them are clouds.
4. **Temporal context** was the literature's proposed fix — but it did *not* transfer here
   ([temporal](temporal-findings.md)); the leverage on pyro-sdis is in the negatives, not time.

## Resolution is a hidden lever

The FIgLib work prompted the question of whether downscaling to 640 also hurts pyro-sdis. It
does — pyro-sdis smoke is small (median box shorter-side 28 px native, ~14 px at 640, 60% under
16 px at 640), and running at native 1280 lifts the *reachable recall ceiling* from 0.68 to 0.86.
This became its own thread, including a full 640-vs-1280 training comparison and the recall-first
reframing (a missed fire costs far more than a false alarm). See
[resolution-findings.md](resolution-findings.md).

## Reproduce

    python src/models/train.py --split grouped --epochs 40           # full baseline
    python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped
    python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped \
        --imgsz 1280 --out results/eval_grouped_proof_test_1280.json  # resolution probe

Raw sweep + base-rate table: `runs/eval_grouped_test.json`.

## Glossary

Hover tooltips appear on first use above (`<abbr>`); definitions are given here in text so they
are reachable on touch devices and by screen readers. If a tooltip does not show on GitHub, its
HTML sanitizer stripped the `title` attribute — this table is the source of truth. Full metric
rationale: [metrics.md](metrics.md).

| Term | Meaning |
|---|---|
| **POD** | Probability of detection — the fraction of real fires the detector catches. Equal to recall. The recall-first headline. |
| **FAR** | False-alarm ratio — FP/(FP+TP) = 1 − precision. The share of raised alarms that are wrong. |
| **FP/camera/day** | False-alarm burden: false positives per camera per day at a target POD (assumes 1% base rate, 500 frames/camera/day — an extrapolation). Pano's operational target is < 1. |
| **base rate** | Fraction of frames that actually contain smoke in deployment (~1% assumed). The test set's ~90% positive rate inflates precision far above field values. |
| **REV** | Relative Economic Value — a detector's value to a user with a given cost/loss ratio; 0 = no better than always/never alarming, 1 = perfect. |
| **F1 / mAP** | Frame-level scores computed but demoted: F1 weights a missed fire like a false alarm; mAP relies on box-IoU, ill-defined for boundary-less smoke. |
