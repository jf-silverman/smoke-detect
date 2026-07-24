# Time-to-detection (TTD) — Phase A findings

**Status: Phase A, directional.** The field's headline metric — how many minutes after ignition
until the first alarm — computed for the first time in this project, on the 18 local FIgLib onset
sequences using a **zero-shot** pyro-sdis detector (never trained on FIgLib). The absolute numbers
are weak and the fire count is small (wide confidence intervals); the value here is a **leak-safe
TTD harness** and a clean result that the [resolution lever](resolution-findings.md) moves TTD, not
just AUC. Phases B–C (more fires; an in-distribution detector) sharpen the magnitudes — see
[backlog.md](backlog.md#pre-scoping-3--hpwren--time-to-detection).

## Why TTD, and why it was missing here

Operational systems and the academic reference lead with [time-to-detection](#ttd), not F1: Pano
targets detection within ~15 min, and SmokeyNet reports ~3.6 min average with ~80% of fires caught
within 5 min ([Dewangan et al., 2022](#works-cited)). This project lacked TTD because pyro-sdis has
no ignition *onset* — its bursts capture already-established scenes. FIgLib is built for it: each
frame's filename encodes the signed offset from first visible plume, so minutes-from-ignition
ground truth is free.

## Method — leak-safe, censoring-aware

[`figlib_ttd.py`](../src/models/figlib_ttd.py) reuses the cached native-resolution tiled per-frame
confidences (no model is re-run), and:

- **Calibrates the alarm threshold to a pre-ignition false-alarm budget.** The operator's real
  constraint is trigger-happiness *before* the fire — so the threshold is set to allow at most a
  target fraction of pre-ignition frames to alarm. TTD is then whatever it is at that threshold.
- **Leave-one-fire-out ([LOFO](#lofo)).** For each held-out fire the threshold is calibrated on the
  *other* fires, then TTD is measured on the held-out one — every fire is a test fire, none is
  calibrated on. The right choice for a small fire count.
- **Censoring-aware.** A fire never detected within the sequence is a **miss** (right-censored),
  never averaged into TTD as if it were small. Detection rate and TTD-given-detected are reported
  separately — the pair, so "early" is never bought with "cries wolf".
- **Bootstrap 90% CIs** over fires, because n=18 is small.

## Headline (native-res tiled, 5% pre-ignition FA budget)

| metric | value | 90% CI | SmokeyNet ref (in-distribution) |
|---|---|---|---|
| detection rate | 56% of fires | [33%, 72%] | — |
| median TTD (detected) | 8.0 min | [4.0, 10.0] | ~3.6 min |
| % of all fires within 5 min | 17% | — | ~80% |
| held-out pre-ignition FA | 6.1% | — | — |

We are well short of SmokeyNet — as expected. Theirs is an *in-distribution* CNN+LSTM+ViT trained
on FIgLib; ours is a French-tower [detector](#auc) run **cold** on California (AUC 0.658). This is a
floor, and it establishes the harness the in-distribution model (Phase C) will be scored on.

## Resolution lowers TTD, not just AUC

Same detector, same fires — only the inference resolution differs:

![Native-resolution tiling detects more fires and sooner across the useful false-alarm-budget
range; the whole-frame @640 signal only catches up at permissive budgets where a near-random
detector fires indiscriminately.](figures/ttd_result.png)

| signal | detection rate | median TTD | within 5 min |
|---|---|---|---|
| whole-frame @640 (AUC 0.454) | 33% | 12.5 min | 11% |
| **native-res tiled (AUC 0.658)** | **56%** | **8.0 min** | **17%** |

At the operationally-relevant **tight** budgets (2–10%), tiling catches ~20 pts more fires and
detects them ~4–5 min earlier. The two curves cross only at a permissive ~17–20% budget — and that
crossover is an artifact, not a win for whole-frame: a near-random signal (AUC 0.454) at a loose
threshold alarms on almost everything, producing trivially "early" detections with disastrous
precision. Read the tight-budget end, where skill actually shows. This extends the project's
central resolution finding onto the time axis.

## The trade-offs, quantified (native-res tiled)

**Loosening the false-alarm budget buys earlier, more complete detection** (the operator's dial):

| pre-ignition FA budget | detection rate | median TTD | within 5 min |
|---|---|---|---|
| 2% | 44% | 8.5 min | 11% |
| 5% | 56% | 8.0 min | 17% |
| 10% | 61% | 4.0 min | 33% |
| 20% | 72% | 1.0 min | 39% |

**Requiring temporal persistence *hurts* TTD here** — the opposite of its effect on the
false-alarm-at-matched-recall axis ([figlib-findings.md](figlib-findings.md)):

| persistence k | detection rate | median TTD | within 5 min |
|---|---|---|---|
| 1 (single frame) | 56% | 8.0 min | 17% |
| 2 | 28% | 10.0 min | 0% |
| 3 | 22% | 10.5 min | 0% |

Persistence suppresses *false* alarms by demanding the signal hold across frames — but on this weak
zero-shot signal it also suppresses the *first true* detection, delaying or missing it. Cutting
false alarms and detecting early are different objectives; a rule that helps one can hurt the other.
Whether an in-distribution detector (with a strong, stable onset signal) reverses this is a Phase-C
question.

## Caveats

- **Zero-shot detector.** AUC 0.658 on FIgLib; a France-trained model only partly transfers to
  California. Magnitudes are a floor, not the achievable TTD.
- **Small n (18 fires).** CIs are wide (detection rate ±~20 pts). Directions are trustworthy;
  point values are not final. Phase B (FIgLib-full / PYRONEAR-2025) tightens them.
- **Day-only, curated onset.** FIgLib is daytime and built around clean ignitions; real deployment
  is harder.

## What sharpens this next

- **Phase B — more fires** (FIgLib-full ~315 fires via WIFIRE Commons; PYRONEAR-2025) → tighter CIs.
- **Phase C — an in-distribution detector** trained on FIgLib's own boxes → the AUC 0.658 floor
  rises, and TTD should drop toward the SmokeyNet range. The same harness scores it unchanged.

## Reproduce

    python src/models/figlib_tiled.py --tile 640 --stride 640     # once: caches confidences (runs detector)
    python src/models/figlib_ttd.py --far-target 0.05             # TTD eval (no model, no GPU)
    python src/models/figlib_ttd.py --features data/figlib/features.npz --tag _wholeframe  # whole-frame
    python src/models/plot_ttd.py                                 # figure

Results: `results/figlib_ttd.json` (tiled), `results/figlib_ttd_wholeframe.json` (whole-frame).

## Glossary

Each term below is a linkable heading — the highlighted term in the text jumps here, and your
browser's **Back** button returns you to where you were reading. Definitions are in text (the
reliable fallback, since GitHub does not render hover tooltips).

#### TTD

Time-to-detection — minutes from a fire's ignition (first visible plume) to the first alarm.
Reported over *detected* fires; a miss is right-censored, not counted as a small TTD.

#### LOFO

Leave-one-fire-out — hold out one fire, calibrate the alarm threshold on the others, measure TTD on
the held-out fire; repeat for every fire. Leak-safe and data-efficient for a small fire count.

#### AUC

Area Under the ROC Curve — ranks smoke vs clean frames; 0.5 = random. The zero-shot detector scores
0.658 tiled (native-res) vs 0.454 whole-frame @640.

#### pre-ignition false-alarm budget

The fraction of pre-ignition (no-fire-yet) frames allowed to alarm — the operator's
trigger-happiness constraint, and the dial the threshold is calibrated to.

#### POD

Probability of detection — here, the share of fires ever detected post-ignition (the detection
rate). Distinct from *how fast* (TTD).

## Works Cited

- Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., &
  Nguyen, M. H. (2022). FIgLib & SmokeyNet: Dataset and deep learning model for real-time wildland
  fire smoke detection. *Remote Sensing, 14*(4), 1007. https://doi.org/10.3390/rs14041007
