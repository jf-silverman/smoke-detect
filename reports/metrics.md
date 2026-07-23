# How performance is measured here — and why not F1

Early reports in this project leaned on F1 and base-rate-corrected precision. That was a
mistake in emphasis, and this note fixes it. In wildfire detection the two errors are not
equally costly, so a metric that treats them equally is the wrong objective. Here is what the
field actually uses, and what we adopted.

## The asymmetric cost

A **missed fire** can cost lives and millions of dollars. A **false alarm** costs a watchstander
a few seconds — every operational system keeps a human in the loop who reviews each candidate detection
before suppression resources are dispatched. The loss function is steeply asymmetric, and any
score that weights a false negative like a false positive (F1, accuracy, and the Critical Success
Index all do) is miscalibrated for this domain.

The reason false alarms are not *free*, though, is **alarm fatigue**: if the false-positive rate
is high enough, operators stop trusting the alarms and real ones get ignored too — the failure
mode behind the field's cautionary tale of a system with a 79% field false-positive rate. So the
objective is not recall at any cost. It is:

> **maximize the detection rate subject to a false-alarm rate low enough that a human still
> reviews every alarm** — i.e. false-positives-per-camera-per-day within a watchstander's budget.

## What the field actually reports

- **Operational camera networks** — the closest analog to this project — headline two numbers,
  and F1 is not among them. [Pano AI](https://www.pano.ai/solution) and
  [ALERTCalifornia](https://www.kpbs.org/news/science-technology/2026/05/05/states-across-the-wildfire-prone-western-us-are-using-ai-for-early-detection)
  (1,060+ cameras) track **time-to-detection** and **false-positives-per-camera-per-day**. Pano's
  stated operating point: detection within ~15 min of ignition at **< 1 false positive per camera
  per day**.
- **Academic smoke detection** (SmokeyNet / FIgLib, and the
  [multimodal work](https://arxiv.org/pdf/2212.14143)) leads with **time-to-detection** (3.7–4.9
  min; the share of fires detected within 5 min) and **probability of detection
  ([POD](#pod) = recall)**.
- **Government / satellite** ([NOAA GOES active fire](https://www.star.nesdis.noaa.gov/goesr/product_land_fire.php),
  the new [Next-Gen Fire System](https://www.noaa.gov/news-release/noaa-unveils-powerful-convergence-of-ai-and-science-with-revolutionary-next-generation-fire-system))
  validate against airborne truth with the meteorology triad: **POD** (hit rate),
  **[FAR](#far)**
  (false-alarm *ratio* = FP/(FP+TP) = 1 − precision), and
  **[CSI](#csi)** (Critical Success Index).
- **The formal answer to asymmetric cost** comes from meteorological forecast verification: the
  **cost-loss ratio** and **relative economic value ([REV](#rev))**
  ([cost/loss & relative value](https://www.cawcr.gov.au/projects/verification/value/relativevalue_more.html);
  Richardson 2000). C is the cost of acting on an alarm, L the loss from a miss; α = C/L is
  *small* when misses dominate — the wildfire regime — and the theory says: at small α, operate
  at a low threshold (high POD, accept more false alarms). REV scores the detector from 0
  (no better than always/never alarming) to 1 (perfect) for a user with a given α.

## What we adopted (`src/models/evaluate.py`)

- **POD (recall)** over a confidence sweep, and the **maximum reachable POD** — the recall-first
  ceiling. A detector that caps at POD 0.68 cannot be operated safely no matter the threshold.
- **False-alarm burden as FP/camera/day** at a target POD, next to Pano's < 1 target.
- **Relative economic value across cost-loss ratios**, the asymmetric-cost score.
- **FAR, [POFD](#pofd), CSI** reported per threshold (field vocabulary).
- **F1 and base-rate-corrected precision demoted to context** — the base-rate precision number is
  the *alarm-fatigue constraint* (how often a human is pinged), not a verdict that the model is
  bad.

## The current picture, in these terms

Grouped-test numbers (assumed 1% base rate, 500 frames/camera/day — an extrapolation, see
caveat):

| configuration | max POD | FP/camera/day @ max POD | REV @ C/L=0.01 | REV @ C/L=0.002 |
|---|---:|---:|---:|---:|
| 640-train, infer @640 | 0.676 | ~208 | +0.26 | −1.05 |
| 640-train, infer @1280 | **0.859** | ~388 | +0.08 | −0.50 |
| 1280-train, infer @1280 (full-scale) | 0.827 | ~173 | **+0.48** | **−0.22** |

Read the last column — the misses-dominate regime this domain lives in. **No config reaches
positive value there yet**; the least-bad is the full-scale 1280-trained model at −0.22, a clear
step up from the −0.50/−1.05 of the others. And that model is the best config on every axis that
matters: training at native 1280 to convergence **holds the recall ceiling** (POD 0.83, within a
hair of the 0.86 from downscaled inference) while **roughly halving the false-alarm burden** (~173
vs ~388 FP/camera/day) and posting the **best REV in both cost-loss columns** (+0.48 at C/L=0.01,
−0.22 at C/L=0.002). That is the resolution lever paying off in *training*, not just inference —
the earlier proof 1280 run only looked worse because it was undertrained (val metrics still
climbing at epoch 15; see [resolution-findings](resolution-findings.md)). The remaining gap to
positive value in the misses-dominate regime is the false-alarm burden: push POD up (native-
resolution training) *and* drive false alarms down (hard-negative mining, the confuser corpus)
toward the < 1 FP/camera/day an operator can live with. Neither lever alone gets there.

## Caveats

- **FP/camera/day is an extrapolation.** pyro-sdis is not a continuous feed, so it needs an
  assumed frame cadence (500/camera/day) and deployment base rate (1%). The *ratios between
  configs* are trustworthy; the absolute per-day numbers are illustrative.
- **Time-to-detection is not yet computed.** It needs onset sequences; pyro-sdis lacks them, but
  FIgLib has them, so [TTD](#ttd) is a natural addition on that data ([figlib-findings](figlib-findings.md)).
- The **1280-train row is now a full-scale, converged run** (40 epochs, full data); the two
  640-train rows remain proof scale. So the *training-resolution* comparison is trustworthy in
  absolute terms; the cross-config false-alarm ratios still favor direction over absolutes.

## Reproduce

    python src/models/evaluate.py --weights runs/grouped_proof/weights/best.pt --split grouped \
        --target-pod 0.90 --base-rate 0.01 --frames-per-day 500

## Glossary

Each term below is a linkable heading — the highlighted term in the text jumps here, and your browser's **Back** button returns you to where you were reading. Definitions are given in text (the reliable fallback, since GitHub does not render hover tooltips).

#### POD

Probability of detection — the fraction of real fires the detector catches. Equal to recall / hit rate / true-positive rate. The recall-first headline.

#### FAR

False-alarm ratio — FP/(FP+TP) = 1 − precision. The share of *raised alarms* that are wrong. (Meteorology's FAR; not the same as the false-alarm *rate*.)

#### POFD

Probability of false detection — false-alarm *rate* on negative (no-smoke) frames = FP/(FP+TN). Drives the per-camera-per-day burden.

#### CSI

Critical Success Index — TP/(TP+FP+FN), a single skill score. Reported for field vocabulary but weights misses and false alarms equally, so not the objective here.

#### REV

Relative Economic Value — a forecast's value to a user with a given cost/loss ratio; 0 = no better than always- or never-alarming, 1 = perfect (Richardson, 2000).

#### C/L (α)

Cost-loss ratio — cost of acting on an alarm ÷ loss from a miss. Small when misses dominate (the wildfire regime), which argues for a high-POD, low-threshold operating point.

#### TTD

Time-to-detection — minutes from a fire's ignition to the first alarm. Needs onset sequences (FIgLib), so not yet computed on pyro-sdis.

#### FP / TP / FN / TN

False positive / true positive / false negative / true negative.

#### mAP

mean Average Precision — the standard object-detection score (area under the precision–recall curve, averaged over classes and IoU thresholds). Computed but demoted, because box-IoU is ill-defined for boundary-less smoke.

#### base rate

The fraction of frames that actually contain smoke in deployment (~1% assumed here). Precision is highly sensitive to it; the test set's ~90% positive rate inflates precision far above field values.
