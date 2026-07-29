# Wildfire Smoke Detection — Findings Report

A single technical read on early wildfire smoke detection from fixed-camera imagery. The theme
throughout is **field-realistic evaluation**: measuring what a detector would actually do on a tower,
not what it scores on a flattering benchmark. Each section summarizes a result and links to its full
working notes in [`research/`](../research/); the companion
[research narrative](research-narrative.md) tells the same story as it actually unfolded, decision by
decision.

> **Scope & maturity.** Most numbers are *proof-scale* — read the direction, not the third decimal.
> The native-resolution training row is a converged full-scale run (40 epochs, full data); the rest
> are one flag change from full scale. Absolute deployment burdens (FP/camera/day) are extrapolations
> at an assumed base rate, not measured field rates.

## Contents

1. [The problem, and the frame](#1-the-problem-and-the-frame)
2. [How performance is measured — and why not F1](#2-how-performance-is-measured--and-why-not-f1)
3. [The field in brief](#3-the-field-in-brief)
4. [Baseline: the precision collapse](#4-baseline-the-precision-collapse)
5. [Resolution: one lever, both axes](#5-resolution-one-lever-both-axes)
6. [Into the negatives: hard-negative mining and the confuser corpus](#6-into-the-negatives-hard-negative-mining-and-the-confuser-corpus)
7. [Temporal: a negative result, and a sign-flip](#7-temporal-a-negative-result-and-a-sign-flip)
8. [Time-to-detection](#8-time-to-detection)
9. [Motion: horizon-anchored change](#9-motion-horizon-anchored-change)
10. [Where the gap stands, and what's next](#10-where-the-gap-stands-and-whats-next)
11. [Works cited](#11-works-cited)

---

## 1. The problem, and the frame

The only comparable smoke detector with a published *field* number produced a **79% false-positive
rate** in real deployment (Govil et al., 2020). The most-cited academic model, SmokeyNet, looks far
stronger on paper — but it was never field-deployed, and its headline score is **F1, a metric that
weights a missed fire exactly like a false alarm** (Dewangan et al., 2022). That is the wrong trade
for wildfire, where a missed fire is catastrophic and a false alarm costs a watchstander a glance.

The gap between benchmark and field — and the misleading yardstick behind it — is the problem this
project is built around. The opening decision was not to build a detector but to **measure what a
detector would actually do in the field**: its detection rate and its false-alarm *burden*. Every
later choice — site-holdout splits, operator metrics, base-rate correction — descends from that
frame.

## 2. How performance is measured — and why not F1

In wildfire, sensitivity and specificity are not equally important. A missed fire is catastrophic; a
false alarm costs a watchstander a glance, because a human reviews every candidate detection before
any suppression resource moves. F1 — which weights the two errors equally — encodes the wrong cost
model. So evaluation here is rebuilt on what the field and the meteorology-verification literature
actually use:

- **Probability of detection (POD)** — the recall ceiling, asked as *how high can detection go*.
- **False-alarm burden** — false positives per camera per day, the operator's real constraint
  (Pano AI's operational target is **< 1**; Pano AI, 2024).
- **Precision at the deployment base rate** — because smoke is rare (~1% of frames), a detector's
  test-set precision badly overstates its field precision; base-rate correction gives the real number.
- **Relative economic value across cost-loss ratios** — meteorology's score for asymmetric costs.

mAP and F1 are still computed, but demoted to context. Full rationale and formulae:
[`research/metrics.md`](../research/metrics.md).

## 3. The field in brief

Smoke is not an object — it has no fixed shape, translucent edges, and looks like the very things it
must be told apart from (cloud, fog, haze, dust). That is why single-frame detectors fire on nearly
every cloud, and why the literature reaches for frame-to-frame context. Public data traces almost
entirely to two lineages (HPWREN/FIgLib in California; PyroNear in Europe); operational vendors
(Pano, ALERTCalifornia) release little and publish less. The full survey — datasets, methods, how
performance is really measured, and when these tools do and don't work — is in
[`research/state-of-smoke-detection.md`](../research/state-of-smoke-detection.md), backed by the
source reviews in [`research/`](../research/).

## 4. Baseline: the precision collapse

A single-frame YOLO detector on [pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis)
(33,636 images, French towers; Pyronear, 2025), evaluated on **leak-safe splits** — the 40 camera
IDs are really 8 physical towers, so whole *sites* are held out and the model is only tested on
terrain it never trained on. It reproduces the documented failure mode: **42% false alarms on clean
frames**, and once corrected to a 1% deployment base rate, **precision ≈ 1.6%** — the field number
the benchmark hides. Detail: [`research/baseline-findings.md`](../research/baseline-findings.md).

## 5. Resolution: one lever, both axes

The sharpest turn in the project came from a physics question, not a model change: a plume at ignition
is a few dozen pixels in a 3072-wide frame, and the pipeline was downscaling every frame to 640 px —
pooling those pixels into nothing. Running the *same* detector on native-resolution **tiles** instead
of a downscaled whole frame moved two things at once:

| configuration | detection rate (POD) | false-alarm burden* |
|---|---|---|
| single-frame baseline (infer @640) | 0.68 | ~208 FP/camera/day |
| + native-resolution inference (@1280) | **0.86** | ~388 FP/camera/day |
| + native-resolution **training** (@1280, full-scale) | 0.83 | ~173 FP/camera/day |
| + hard-negative mining (@1280, both levers) | 0.80 | **~133 FP/camera/day** |

<sub>*extrapolated at an assumed 1% base rate and 500 frames/camera/day, not a measured rate.</sub>

**Resolution raises the detection ceiling.** The 640 model structurally caps at POD 0.68 (it never
sees the small plumes); native-resolution inference reaches 0.86, and training at native resolution
to convergence holds 0.83 at **roughly half the false-alarm burden**. Detail:
[`research/resolution-findings.md`](../research/resolution-findings.md).

## 6. Into the negatives: hard-negative mining and the confuser corpus

Where resolution buys recall, the false-alarm *burden* is bought in the negatives. **Hard-negative
mining** cut false alarms **42% → 20%** and doubled precision@1%. **Combining both proven levers** —
native-resolution training *plus* mining, the deployable recipe — cuts the burden a further ~23%
(173 → 133 FP/camera/day) and lifts precision@1% (2.3% → 2.9%), at the cost of ~2.4 points of recall
ceiling. A real but *incremental* win; the converged model still false-alarms on **57.6% of clean
frames**, so resolution buys recall, not calm. Detail:
[`research/hard-negative-findings.md`](../research/hard-negative-findings.md).

To see *what* it fires on, we clustered the 2,305 false-alarm frames into named failure modes. The
result is one clean number — **74% of the false alarms are clouds** (cumulus, backlit stratus, broken
overcast) — a measured version of the documented single-frame failure mode. No such public corpus
existed, so [`results/confuser_corpus.csv`](../results/confuser_corpus.csv) is a small original
contribution. Detail: [`research/confuser-corpus.md`](../research/confuser-corpus.md).

## 7. Temporal: a negative result, and a sign-flip

Frame-to-frame context is the literature's unanimous fix (SmokeyNet's +26 precision points; Dewangan
et al., 2022). We built it and it **did not transfer to pyro-sdis** — at matched recall, no temporal
method beat the single-frame detector. Measuring *why* reframed the project: **76% of the false alarms
are persistent structures** (fixed cloud banks, glare, ridge haze), not the flicker a persistence rule
suppresses. On pyro-sdis, the leverage is in the negatives, not the time axis. Reported as the
negative result it is. Detail: [`research/temporal-findings.md`](../research/temporal-findings.md).

The control that confirmed the mechanism came on [**FIgLib**](../research/figlib-findings.md), the
onset-sequence dataset. The first run looked dead (AUC 0.454, worse than random) until the resolution
fix: native-resolution tiling lifted AUC to **0.658**, and the positive control then landed —
requiring temporal persistence **cuts** false alarms 12–19 points on FIgLib, where the *same rule
raised them* on pyro-sdis. Same rule, opposite sign, split by whether the data contains ignition
onset. The mechanism, confirmed both ways.

## 8. Time-to-detection

Time-to-detection (TTD) is the metric operators and the academic reference actually lead with, and the
project had lacked it. A cache-only, leak-safe leave-one-fire-out harness on 23 held-out California
onset fires (day-only, matching the FIgLib benchmark) gives the first numbers: a zero-shot detector
detects **48% of fires at a median 8 minutes** (17% within 5 min; pre-ignition false-alarm rate
~5.7%). Resolution shows up here too —
native tiling detects more fires *and* detects them minutes sooner. But the harness also surfaces the
motivating miss: with this **proof-scale** zero-shot detector, the most recent CA fires (Palisades,
Coches, Tenaja) are missed — a distribution-shift signal that a French-tower detector does not
transfer to recent California smoke. Closing that miss is exactly what Phase C set out to do (§10),
and it is where the story turns from diagnosis to fix. Detail:
[`research/ttd-findings.md`](../research/ttd-findings.md).

## 9. Motion: horizon-anchored change

The newest thread came from a domain observation while hand-correcting labels: a plume is often
legible only by its **motion** across frames, invisible in any single still. A training-free probe
tests whether inter-frame change **anchored to the horizon** (connected to the ground, unlike
free-floating cloud drift) separates ignition onset from pre-ignition frames.

It does, and — importantly — it is *complementary* to the appearance detector. On the 17 day-only
training fires, horizon-anchored motion separates onset at **per-fire AUC 0.708** (90% CI
[0.606, 0.807], excluding chance) versus **0.570** for the detector's own confidence, with a
floating-change *control* near chance (0.566) — so it is anchoring, not merely motion, doing the work.
Where it pays off most is exactly the hard imaging cases: the faint distant plume of `so-w-mobo-c`
(conf AUC 0.07 → anchored 0.79), the oversaturated Dehesa plume (0.12 → 0.80), and `syp-w-mobo-c`
(0.23 → 0.97) — plumes the appearance detector scores near chance. It beats the detector's confidence
in **11 of 17** fires (mean per-fire lift +0.14; anchored AUC ≥ 0.7 in 11 fires, ≥ 0.8 in 9), but at
this sample size that head-to-head win-count is **not itself significant** (one-sided sign-test
p = 0.17). The reproducible, defensible claim is narrower than a clean victory: anchored motion
*separates onset above chance* and *carries signal the appearance detector misses* — a complementary
channel, not a replacement. The measurement mattered too: pooling frames across fires understated it
(0.64) because absolute motion scale varies by scene; scored *within* each fire it is 0.71.

The same probe produced **three clean negative results**, each reported as such: a texture cue
(diffuse-haze × hard-edge) that breaks under the very atmospheric conditions that matter; and two
horizon-estimator "upgrades" (a per-column smoothed skyline, and a luminosity sky/ground mask) that
both *underperformed* the simplest flat horizon and were reverted. Detail:
[`research/motion-findings.md`](../research/motion-findings.md).

## 10. Where the gap stands, and what's next

**Phase C ran, and it moved the held-out numbers.** The recent-CA miss is a distribution-shift
problem, so we fine-tuned the full-scale pyro-sdis detector on our own **hand-corrected** Californian
smoke boxes (pseudo-labeled by the base detector, then corrected in CVAT by the author; night fire
excluded; strict whole-fire holdout), trained on a GCP L4 for ~$0.35. Measured on the 6 held-out
recent CA fires against the *exact base model it started from*:

- **Detection 4/6 → 5/6** — Coches (2025) flips from missed to detected.
- **Pre-ignition false alarms 16.7% → 7.1%** — more than halved, *while detecting more*. Phase C
  dominates its base at this operating point, and behaves far more consistently across unseen fires
  (a threshold set on 5 fires transfers to the 6th) — that consistency is the shift closing.
- **A correction to our own earlier claim**: the "misses Palisades/Coches/Tenaja" line was the
  *proof-scale* zero-shot model. The full-scale base already recovers Palisades and Tenaja; Phase C's
  honest adds are Coches plus the halved false-alarm rate. Vista (2024) is still missed by both.
- **Motion generalizes**: horizon-anchored motion separates onset on the held-out fires at per-fire
  AUC **0.742** (CI excludes chance) — the cue transfers to recent CA smoke it was never tuned on.

Directional (n = 6, wide CIs), but the first evidence that in-distribution fine-tuning closes part of
the gap. Detail: [`research/phase-c-findings.md`](../research/phase-c-findings.md).

**What's still open.** The deployable single-frame recipe reaches ~133 FP/camera/day at POD 0.80 —
still far from the < 1 an operator can live with. The live threads: **motion as a learned temporal
channel** (the payoff test is whether it *lowers TTD*, not just raises separability); **more fires**
(Phase B — FIgLib-full) to tighten the n = 6 eval; and a matched-false-alarm comparison. All tracked
in [`research/backlog.md`](../research/backlog.md).

## 11. Works cited

- Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., &
  Nguyen, M. H. (2022). FIgLib & SmokeyNet: Dataset and deep learning model for real-time wildland
  fire smoke detection. *Remote Sensing, 14*(4), 1007. https://doi.org/10.3390/rs14041007
- Govil, K., Welch, M. L., Ball, J. T., & Pennypacker, C. R. (2020). Preliminary results from a
  wildfire detection system using deep learning on remote camera images. *Remote Sensing, 12*(1),
  166. https://doi.org/10.3390/rs12010166
- Pano AI. (2024). *Pano Rapid Detect: solution overview* [Product page]. https://www.pano.ai/solution
- Pyronear. (2025). *pyro-sdis* [Dataset]. Hugging Face. https://huggingface.co/datasets/pyronear/pyro-sdis

<sub>Per-topic sources (operational networks, NOAA GOES/NGFS, meteorological verification) are cited
inline in [`research/metrics.md`](../research/metrics.md) and the source reviews in
[`research/`](../research/).</sub>
