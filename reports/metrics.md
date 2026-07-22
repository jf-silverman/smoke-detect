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
  and F1 is not among them. [Pano AI](https://www.pano.ai/) and
  [ALERTCalifornia](https://www.kpbs.org/news/science-technology/2026/05/05/states-across-the-wildfire-prone-western-us-are-using-ai-for-early-detection)
  (1,060+ cameras) track **time-to-detection** and **false-positives-per-camera-per-day**. Pano's
  stated operating point: detection within ~15 min of ignition at **< 1 false positive per camera
  per day**.
- **Academic smoke detection** (SmokeyNet / FIgLib, and the
  [multimodal work](https://arxiv.org/pdf/2212.14143)) leads with **time-to-detection** (3.7–4.9
  min; the share of fires detected within 5 min) and **probability of detection (POD = recall)**.
- **Government / satellite** ([NOAA GOES active fire](https://www.star.nesdis.noaa.gov/goesr/product_land_fire.php),
  the new [Next-Gen Fire System](https://www.noaa.gov/news-release/noaa-unveils-powerful-convergence-of-ai-and-science-with-revolutionary-next-generation-fire-system))
  validate against airborne truth with the meteorology triad: **POD** (hit rate), **FAR**
  (false-alarm *ratio* = FP/(FP+TP) = 1 − precision), and **CSI** (Critical Success Index).
- **The formal answer to asymmetric cost** comes from meteorological forecast verification: the
  **cost-loss ratio** and **relative economic value (REV)**
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
- **FAR, POFD, CSI** reported per threshold (field vocabulary).
- **F1 and base-rate-corrected precision demoted to context** — the base-rate precision number is
  the *alarm-fatigue constraint* (how often a human is pinged), not a verdict that the model is
  bad.

## The current picture, in these terms

Honest grouped-test numbers (assumed 1% base rate, 500 frames/camera/day — an extrapolation, see
caveat):

| configuration | max POD | FP/camera/day @ max POD | REV @ C/L=0.01 | REV @ C/L=0.002 |
|---|---:|---:|---:|---:|
| 640-train, infer @640 | 0.676 | ~208 | +0.26 | −1.05 |
| 640-train, infer @1280 | **0.859** | ~388 | +0.08 | **−0.50** |
| 1280-train, infer @1280 (proof) | 0.578 | ~120 | +0.34 | −1.37 |

Read the last column — the misses-dominate regime this domain lives in. **No proof config reaches
positive value there**: even at POD 0.86 you miss 14% of fires while generating ~388 false alarms
a day, so alarming on everything still competes. The least-bad is the highest-POD config (native-
resolution inference), because in that regime *reducing misses* is what buys value. That is the
whole recall-first argument, quantified: push POD up (resolution, a converged full-scale model),
then drive the false-alarm burden down (hard-negative mining, the confuser corpus) toward the < 1
FP/camera/day an operator can live with. Neither lever alone gets there.

## Honest caveats

- **FP/camera/day is an extrapolation.** pyro-sdis is not a continuous feed, so it needs an
  assumed frame cadence (500/camera/day) and deployment base rate (1%). The *ratios between
  configs* are trustworthy; the absolute per-day numbers are illustrative.
- **Time-to-detection is not yet computed.** It needs onset sequences; pyro-sdis lacks them, but
  FIgLib has them, so TTD is a natural addition on that data ([figlib-findings](figlib-findings.md)).
- Proof scale throughout — direction over absolutes.

## Reproduce

    python src/models/evaluate.py --weights runs/grouped_proof/weights/best.pt --split grouped \
        --target-pod 0.90 --base-rate 0.01 --frames-per-day 500
