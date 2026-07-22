# FIgLib positive control — findings

**Status: confirmed, after a resolution fix.** The pyro-sdis result claimed temporal context
fails there *specifically because* that dataset lacks ignition onset. FIgLib is the direct
test: every sequence spans ~40 min before to ~40 min after a fire's first visible plume, one
frame per minute — the onset dynamics pyro-sdis is missing. We ran the identical pipeline on 18
FIgLib fire sequences (1,434 frames, near-balanced 726 smoke / 708 clean), holding out whole
fires.

The first attempt was confounded, and the fix — prompted by a good question about resolution —
is itself the most instructive part of this report. Once the detector was allowed to see the
frames at native resolution, the positive control **succeeded**: requiring temporal persistence
cuts false alarms by 12–19 points on FIgLib, the mirror image of pyro-sdis where it *raised*
them.

## The plot twist: it was resolution, not (only) distribution shift

The whole-frame pipeline resized FIgLib's native 3072×2048 frames down to 640 px before
inference — and the pyro-sdis detector scored **AUC 0.454, worse than random**. It looked like
pure distribution shift (French towers → California). It was mostly downscaling. An onset plume ~40 px
wide in the native frame becomes ~8 px at 640, pooled away to nothing.

Re-running the *same* detector on native-resolution 640-px **tiles** (no downscaling, the
resolution it was trained at), taking each frame's max confidence over its tiles:

| inference | FIgLib test AUC |
|---|---|
| whole frame, resized to 640 | 0.454 (worse than random) |
| **native-resolution tiles** | **0.658** |

A +0.20 AUC jump from resolution alone. This is exactly why SmokeyNet tiles into 224-px patches
— small early plumes only survive at native pixel density. Distribution shift is still present (0.658,
not 0.85+; a French-tower detector is not at home in California), but it is no longer
disqualifying. The lesson generalizes: **for small-object detection, downscaling in the
inference path can matter more than the model does.**

## With a usable base signal, temporal wins — the mirror of pyro-sdis

The model-independent persistence rule (rolling min of confidence over the window, no trained
features) vs single-frame, at matched recall, on the **tiled** base signal:

| recall | single-frame FA | persistence FA | delta (neg = temporal helps) |
|---|---:|---:|---:|
| 0.70 | 45.6% | 26.9% | **−18.8 pts** |
| 0.60 | 31.9% | 14.4% | **−17.5 pts** |
| 0.50 | 21.9% | 6.2% | **−15.6 pts** |
| 0.40 | 17.5% | 5.0% | −12.5 pts |

(Persistence still hurts at very high recall ≥0.8, where you accept nearly every frame anyway.)
Requiring temporal persistence cuts false alarms by 12–19 points across the useful operating
range — because on FIgLib the pre-ignition negatives are genuinely transient (empty sky, passing
artifacts), the flicker a temporal rule is built to suppress.

## The headline: the same rule flips sign with the dataset

The one model-independent comparison, side by side (persistence minus single-frame false-alarm
rate; **negative = temporal helps**):

| recall | pyro-sdis (established scenes) | FIgLib (onset sequences, tiled) |
|---|---:|---:|
| 0.70 | −0.8 pts | **−18.8 pts** |
| 0.60 | −0.4 pts | **−17.5 pts** |
| 0.50 | **+6.6 pts** (hurts) | **−15.6 pts** (helps) |
| 0.40 | **+6.4 pts** (hurts) | **−12.5 pts** (helps) |

The **same rule flips sign with the dataset.** On pyro-sdis, requiring persistence *hurts* at
the tight operating points — the confusers are persistent (fixed cloud banks, glare, ridge
haze), so persistence keeps them while costing recall. On FIgLib, requiring persistence *helps*
by 12–19 points — the pre-ignition negatives are transient. This is the mechanistic claim
confirmed from both directions at once: **temporal context pays off on onset data and backfires
on established-scene data.** It is not a universal fix; it is a fix for a specific data regime,
and now we can show exactly which.

## Caveats

- **Distribution shift is real but not disqualifying.** The tiled base signal is AUC 0.658, not 0.85+
  — a France-trained detector only partly transfers to California. Absolute false-alarm rates
  are still soft; the *matched-recall deltas and their sign* are the trustworthy part.
- **The learned GRU loses to the parameter-free rule — a data-size story.** We did cache tiled
  *embeddings* (the max-confidence tile per frame) and ran the learned GRU. It underperformed
  both single-frame and the persistence rule (e.g. at recall 0.60, false alarms 77% vs
  persistence's 14%). A conf-only GRU did better than the embedding version but still lost to
  persistence. The reason is data, not mechanism: 18 fires is ~867 training windows, far too few
  to fit a sequence model, so a hand-built inductive bias (require persistence) beats a learned
  one. The temporal *signal* is real and strong — the persistence rule proves it — but a learned
  temporal *model* needs far more fires to earn its keep. (This mirrors the pyro-sdis GRU, which
  also drowned in noisy embedding dims.) Proving a *learned* model wins would need many more
  sequences or an in-distribution detector; the persistence result already confirms the mechanism.
- **18 sequences, 4 held-out test fires.** Small; magnitudes are noisy, the direction is clear.
- Proof scale throughout (underfit, zero-shot detector as the feature source).

## What a fully conclusive control would add (future work)

Tiled *embeddings* + the learned GRU (above), and/or an in-distribution detector trained on FIgLib's
own bounding boxes (SmokeyNet's setup). Both would raise the base AUC and let the learned
temporal model be tested directly. The resolution finding here says the cheapest high-value move
is simply to stop downscaling.

## Reproduce

    # data: subset of FIgLib sequences into data/figlib/images/<seq>/ (gitignored)
    python src/models/figlib_temporal.py --extract     # whole-frame features (the confounded run)
    python src/models/figlib_temporal.py               # AUC 0.454 + confounded comparison
    python src/models/figlib_tiled.py --tile 640 --stride 640   # native-res tiles -> AUC 0.658

Comparisons: `results/figlib_temporal_comparison.json` (whole-frame),
`results/figlib_tiled_comparison.json` (tiled). pyro-sdis counterpart:
`results/temporal_comparison.json`. See also [temporal-findings.md](temporal-findings.md).
