# The State of Wildfire Smoke Detection — Findings and Project Recommendation

*Research synthesis, July 2026. Sources in `research/`. Figures verified against primary
sources except where marked otherwise.*

---

## 1. The finding that should shape your project

**Benchmark performance in this field does not predict field performance, and the gap is
enormous.**

The only deployed wildfire smoke detector with a published field number is Govil et al.
(2020), tested across 65 HPWREN cameras over nine days. After suppressing repeat detections,
**only 21% of its alerts were real fires — a 79% false-positive rate**, with misses never
quantified. Meanwhile **SmokeyNet**, the most-cited academic model (83.49% accuracy, 82.59%
F1, 3.12-minute time-to-detection on FIgLib), **was never field-deployed at all.** Those are
test-set numbers.

This is not an isolated gap. PyroNear's 2025 cross-dataset evaluation shows **Nemo at 86.8%
F1 on its own test set and 63.2% cross-dataset**; SmokeFrames-2.4k drops 82.8% → 69.9%. The
authors attribute the inflated in-dataset scores to "overfitting issues because of the
partitioning." A realistic, diverse benchmark sits around **70% F1**.

The GAO's institutional verdict is bluntly agnostic: the effectiveness of these technologies
"is still being assessed." And an Alberta study of 4,934 fires found each hour of
reporting-delay reduction cut suppression cost by only **~0.25%** — which doesn't make
detection worthless, but does mean the standard early-detection-saves-money pitch is weakly
supported. The real payoff (avoided catastrophic tail events, evacuation lead time) is not
what most papers measure.

**Implication:** a project that reports a high number on a curated benchmark demonstrates
nothing that the field hasn't already shown to be misleading. A project that measures and
explains the *gap* demonstrates judgment.

## 2. Why smoke is hard: it is not an object

Three structural facts drive every design decision:

- **No boundary.** Smoke is semi-transparent and diffuse; a pixel is *partly* smoke. This is
  closer to alpha matting than object detection. Human annotators measurably disagree about
  where a plume ends, which puts an **annotation-noise ceiling** on <abbr title="Intersection over Union — overlap between predicted and ground-truth boxes; ill-defined for a boundary-less object like smoke">IoU</abbr> — a perfect model
  cannot beat inter-annotator agreement. This is why the serious papers abandon standard
  metrics: SKLFS evaluates at **AP@0.1**, and PyroNear drops <abbr title="mean Average Precision — the standard object-detection score; relies on box-IoU, which is ill-defined for smoke">mAP</abbr> entirely in favor of
  precision/recall on smoke *presence*.
- **Early smoke is tiny.** In PyroNear2024, smoke boxes average **1.04% of image area**. On a
  2048×3072 frame, an incipient plume does not survive downsampling to 224×224. Hence tiling.
- **A single frame is genuinely ambiguous.** Cloud, fog, marine layer, dust, haze, and glare
  all look like smoke in a still image — to models *and* to humans. What distinguishes smoke
  is that it **emerges from a fixed point and grows.** The evidence for ignition lives in the
  difference between frames, not in any frame.

That third fact is the most important one in this report, and it has a number attached.

## 3. What the methods comparison actually shows

SmokeyNet's ablation on FIgLib is unusually clean:

| Model | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| ResNet50 (single frame) | 68.51 | 74.30 | **63.35** | 89.89 |
| ResNet34 + LSTM (temporal) | 79.35 | 79.21 | 82.00 | 76.74 |
| **SmokeyNet (CNN+LSTM+ViT)** | **83.49** | **82.59** | **89.84** | 76.45 |
| Human experts | 78.5 | 82.8 | 93.5 | 74.4 |

Read the precision column, not the accuracy column. **Temporal context buys +26 points of
precision over the single-frame CNN.** The single-frame model has 90% recall at 63% precision
— it fires on every cloud. The entire gain from temporal modeling is
**false-positive suppression**, exactly as the ambiguity argument predicts. PyroNear
independently corroborates: their CNN-LSTM adds +9.3% recall at equal precision over a
single-frame YOLOv8s, and cuts detection time from 1:46 to 1:05.

Meanwhile the most-published approach — YOLO on a scraped fire/smoke dataset — reports
mAP@50 of 92.6% on curated data with large obvious plumes, and **F1 of 0.60–0.70 on realistic
early smoke.** Highest headline number, lowest signal, and everyone has done it.

Satellite is a *different problem*, not a harder one. At 375 m (VIIRS) to 2 km (GOES) per
pixel, a fresh ignition is **sub-pixel**. Satellites do plume mapping and confirmed-fire
monitoring; ground cameras do early detection. Don't conflate them.

## 4. Data: what to use and what to avoid

| Dataset | Size | License | Verdict |
|---|---|---|---|
| **pyro-sdis** | 3.28 GB, 33,636 imgs (28,103 w/ smoke) | **Apache-2.0** | ✅ Best start. YOLO-ready, real tower imagery, far/small/low-contrast plumes. *Verified directly.* |
| **D-Fire** | ~2–4 GB, 21,527 imgs | **CC0** | ✅ Best negatives — 9,838, incl. deliberate fire-like distractors. Varied cameras, so no fixed-background leakage. |
| **FIgLib** | ~30 GB, ~25k imgs, 315 fires | none (provided as-is) | ✅ Where the real problem lives: 81-frame sequences at ±40 min from ignition. Enables time-to-detection. |
| **Nemo** | 1.12 GB, 2,934 imgs | Apache-2.0 | ✅ Only dataset with **density ordinals** (low/mid/high). Cheap novelty. |
| FASDD | ~95k imgs | unclear | ⚠️ Preprint **withdrawn by its own authors**; aggregates other public sets (contaminated). |
| FLAME | 41.7 GB | IEEE login | ⚠️ A *single prescribed burn* shot continuously. Near-degenerate; you'll hit 99% and it means nothing. |
| Smoke100k / SMOKE5K | — | NC / unclear | ⚠️ Synthetic. Pretraining only; fatal if used for evaluation. |
| RIS-Fire | — | — | ❌ **Does not appear to exist.** Nearest real thing is FireRisk, a land-cover task. |

### The trap you must not fall into

**FIgLib frames are 60 seconds apart from fixed cameras.** Frames at t=+300s and t=+360s are
effectively the same photograph — same ridge, same vegetation, plume a few percent larger. A
random 80/20 split puts a **near-twin of every test image into training**, and what you
measure is memorization of 101 camera backgrounds, not smoke.

The same applies to Nemo, pyro-sdis, and SmokeViz — every dataset drawn from a continuous
feed. **If you see >95% validation accuracy on FIgLib, you have leaked.**

Split by **fire**, and then also by **camera**. Two different fires from the same camera still
share the entire background.

Also note: **the canonical FIgLib benchmark is day-only** (the authors dropped 19 night fires),
and its ~50/50 class balance is artificial. Real deployment has a minuscule prior on smoke, so
a model tuned on FIgLib's balance will produce a far worse false-positive rate than its
<abbr title="Precision–Recall curve — precision plotted against recall across confidence thresholds">PR curve</abbr> suggests.

## 5. How to measure performance — and what good means

Do **not** lead with accuracy (a trivial always-negative classifier scores near-100% at real base
rates), and do not lead with mAP (IoU is structurally wrong for a boundary-less object).

**Report instead:**
1. **Precision, recall, F1, <abbr title="Area Under the Precision–Recall curve — a threshold-independent summary of the precision/recall trade-off">AUC-PR</abbr>** — with the full PR curve, and an explicitly justified
   operating point.
2. **Time-to-detection** — minutes into the fire sequence at which confidence first crosses
   threshold, averaged per test fire. FIgLib's timestamps make this directly computable.
   Comparable to SmokeyNet's 3.12 min.
3. **False positives per camera per day** on held-out negative sequences. This is what
   operators actually care about, and almost nobody reports it.
4. **Both an easy and a hard split**, side by side.

**What good looks like:** F1 in the **low-to-mid 80s is genuinely strong**. High 80s or 90s
means your test set is too easy or your split leaked. Pyronear reports ~70% F1 on their hard
in-the-wild benchmark and ~91% on a curated holdout — same models, same lineage. That spread
*is* the lesson.

## 6. When these tools are useful — and when they aren't

**Useful:** the pre-911 window. Daylight, clear air, high-contrast sky, fixed camera with a
learned background, small plume on a ridgeline, in terrain where no human is watching.
ALERTCalifornia claims its AI beat 911 reporting >30% of the time in its first season (1,200+
fires) — though this is operator-reported, and I found **no independent, agency-audited
catalogue of camera-first detections.** That absence is itself a finding.

**Not useful:**
- **Night.** Plumes aren't self-luminous. Optical detection collapses to flame/glow detection,
  by which point the incipient stage is over. **Every major dataset deletes night frames** —
  so every published accuracy number in this field is a *daytime* number. Thermal IR sees heat,
  not smoke, and needs line-of-sight to the source.
- **Fog, marine layer, cloud, dust, haze, glare.** The dominant false-positive class.
- **Prescribed and agricultural burns.** Essentially indistinguishable to a plume detector.
  **No published work benchmarks this**, despite it being a first-order operational nuisance.
- **Behind a ridge.** Invisible until the plume tops it — by which point early detection is moot.
- **Already-large fires.** Detection is pointless; everyone already knows.

**The realistic framing:** on a clear afternoon with a good sky background, smoke detection is
close to solved. What is *not* solved is the false-positive rate at operational base rates,
night, generalization across cameras and geographies, and the human dispatch pipeline the alert
lands in. **The bottleneck is the deployment envelope, not the backbone.** Every deployed system
found is human-in-the-loop — AI proposes, a watchstander confirms. The model is not the system.

## 7. Recommendation

Build **a tiled temporal smoke detector on real camera data, evaluated the way an operator
would evaluate it.** Concretely:

**Stage 1 — Baseline that fails informatively.** Tiled single-frame CNN (ResNet34 or
EfficientNet-B0, 224×224 tiles) on pyro-sdis, split by camera. **The goal is not a good number
— it is to reproduce the precision collapse.** Show 90% recall at ~63% precision, then open the
false positives and demonstrate that nearly all of them are clouds and fog.

**Stage 2 — The differentiator: add temporal context.** A SmokeyNet-style CNN → LSTM →
attention-pool over tiles, on FIgLib's sequences. Present the ablation ladder as four bars:
single-frame → + frame-differencing channel → + LSTM → + cross-tile attention. Show the
precision curve moving, and explain *why*: a static hazy frame is ambiguous, but growth from a
fixed point is not. **That is a causal story about the data, which is what distinguishes a data
scientist from someone who can call `.fit()`.**

**Stage 3 — Report what nobody reports.** Time-to-detection, false-positives-per-camera-per-day,
and a held-out-camera generalization number *even though it will be worse* (94% on a random
split, 71% when whole cameras are held out). Then evaluate zero-shot on D-Fire to quantify distribution shift.

**Optional high-leverage extension:** build a **curated hard-negative corpus** — fog banks,
marine layer, dust, glare, contrails, prescribed burns — with per-confuser error breakdowns.
The literature review found that **no such public corpus exists**, and it is the single
highest-leverage, lowest-cost contribution an individual can make to this field.

**What to avoid:** plain YOLO on a scraped Kaggle fire dataset (done to death, and the metric is
meaningless for smoke); FLAME (degenerate); satellite as the main project (different problem,
weak labels); CLIP/SAM zero-shot as the core bet (genuinely unvalidated for smoke — fine as a
stretch section, not as the thesis).

**The one-sentence pitch this project earns you:** *"Everyone reports 92% mAP on curated smoke
benchmarks. I measured what happens on cameras the model has never seen, found the precision
collapse, showed it's caused by clouds, cut the false-alarm rate in half by mining hard
negatives — and when I tested the temporal fix everyone assumes, I found it doesn't transfer to
this dataset and explained why. Here is the false-alarm rate an operator would actually live
with."*

---

> **Update (post-experiment).** This section is the research *plan*, written before building
> anything. The experiments revised it on one major point. Stage 2 predicted a temporal model
> would be the differentiator that fixes the precision collapse. It was built and it **did not
> transfer to pyro-sdis** — at matched recall no temporal method beat the single-frame detector,
> because 76% of the false alarms are *persistent* confusers (not the flicker temporal models
> suppress) and this dataset's short bursts lack FIgLib's ignition-onset dynamics. The fix that
> actually worked was Stage-4-style **hard-negative mining** (false alarms 42% → 20%). The
> temporal literature summary above (§4) remains accurate *for FIgLib*; the lesson is that the
> temporal advantage is dataset-dependent, not universal. See
> [`temporal-findings.md`](temporal-findings.md) and [`hard-negative-findings.md`](hard-negative-findings.md).

---

### Caveats on this report
- Figures for pyro-sdis (license, size, counts) and the Govil/SmokeyNet attribution were
  verified against primary sources during this research.
- Several performance numbers (D-Fire YOLO table, some MDPI abstracts) are **reported-by-source**
  and were not hand-verified against original PDFs.
- No verified operational false-alarm rate exists for any *currently deployed* statewide network.
  The 79% figure is from a 2019 research field test. Current systems are presumably better;
  **nobody has published the number.**

---

## Glossary

Key acronyms above are given as hover tooltips on first use (`<abbr>`); definitions live here in
text so they are reachable on touch devices and by screen readers. If a tooltip does not show on
GitHub, its HTML sanitizer stripped the `title` attribute — this table is the source of truth.

| Term | Meaning |
|---|---|
| **mAP** | mean Average Precision — the standard object-detection score (area under the precision–recall curve, averaged over IoU thresholds). Relies on box-IoU, which is ill-defined for boundary-less smoke; serious smoke papers drop it. |
| **AP@0.1** | Average Precision at a lenient IoU threshold of 0.10 — used by SKLFS because tight box overlap is meaningless for diffuse smoke. |
| **IoU** | Intersection over Union — overlap between a predicted and a ground-truth box. |
| **PR curve / AUC-PR** | Precision–Recall curve and the area under it — a threshold-independent summary of the precision/recall trade-off; preferred here over accuracy or mAP. |
| **F1** | Harmonic mean of precision and recall. Weights a missed fire like a false alarm, so misleading for this asymmetric-cost domain; reported only for comparison. |
| **TTD** | Time-to-detection — minutes into a fire sequence at which confidence first crosses threshold (SmokeyNet: ~3.1 min). |
| **FP/camera/day** | False positives per camera per day — the operational false-alarm burden; almost nobody reports it. |
| **base rate** | Fraction of frames that actually contain smoke in deployment (tiny); FIgLib's artificial ~50/50 balance inflates apparent precision. |
| **CNN / LSTM / ViT** | Convolutional Neural Network / Long Short-Term Memory / Vision Transformer — SmokeyNet stacks all three (CNN per tile → LSTM across frames → attention across tiles). |
| **FIgLib** | Fire Ignition Library — HPWREN onset sequences (~40 min before/after ignition, 1 frame/min); the dataset where temporal context pays off. |
| **HPWREN** | High Performance Wireless Research and Education Network — the fixed-camera network FIgLib is curated from. |
| **VIIRS / GOES** | Satellite sensors (375 m / 2 km per pixel) used for plume mapping and confirmed-fire monitoring — a different problem from early ground-camera detection. |
| **GAO** | U.S. Government Accountability Office — cited for the institutional verdict that effectiveness "is still being assessed." |
