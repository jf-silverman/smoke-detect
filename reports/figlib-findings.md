# FIgLib positive control — findings

**Status: partial confirmation with an honest confound.** The pyro-sdis result claimed
temporal context fails there *specifically because* that dataset lacks ignition onset. FIgLib
is the direct test: every sequence spans ~40 min before to ~40 min after a fire's first
visible plume, one frame per minute — the onset dynamics pyro-sdis is missing. We ran the
identical pipeline (same frozen detector as feature extractor, same GRU, same matched-recall
comparison) on 18 FIgLib fire sequences (1,434 frames, near-balanced 726 smoke / 708 clean),
holding out whole fires.

The clean version of this control **could not be run**, for an instructive reason. But the one
model-independent signal that survives **flips sign in exactly the direction the hypothesis
predicts.**

## The confound: a pyro-sdis detector is blind on FIgLib

The detector is trained on French detection towers; FIgLib is Southern California (different
terrain, cameras, smoke appearance, including a night fire and monochrome imagers). Zero-shot,
its per-frame signal is **worse than random**:

| feature | FIgLib test AUC |
|---|---|
| detector confidence (conf head) | **0.454** |
| in-domain classifier on its 256-d embedding | 0.504 |
| same, pre-ignition vs only >30-min developed plumes | 0.506 |

The embedding carries essentially no transferable smoke signal — even for large, well-developed
plumes. The likely reason is structural and is the whole point of SmokeyNet's design: FIgLib is
2048×3072, and an onset plume is a tiny, distant patch. A whole-frame descriptor resized to 640
pools it away to nothing. **SmokeyNet tiles into 224-px patches precisely so small early plumes
survive** — our whole-frame pipeline cannot see them. There *is* a faint trace (mean detector
conf rises with plume development: 0.067 pre-ignition → 0.127 after 30 min), but it is far too
weak to build a clean comparison on.

So a *conclusive* positive control — a learned temporal model decisively beating single-frame —
is not achievable by reusing the pyro-sdis detector. It needs an **in-domain, tiled** feature
extractor trained on FIgLib itself. That is a real build, not a cheap proof, and is scoped as
future work below.

## The signal that survives: the persistence effect flips sign

One comparison is *model-independent* and needs no trained features: the persistence rule
(rolling min of confidence over the window) vs single-frame, at matched recall. Because it only
smooths the confidence trajectory, it is meaningful even where the absolute signal is weak. It
answers: does *requiring temporal persistence* reduce or increase false alarms?

Change in false-alarm rate, persistence minus single-frame (**negative = temporal helps**):

| recall | pyro-sdis (established scenes) | FIgLib (onset sequences) |
|---|---:|---:|
| 0.70 | −0.8 pts | **−8.8 pts** |
| 0.60 | −0.4 pts | −1.2 pts |
| 0.50 | **+6.6 pts** (hurts) | **−13.8 pts** (helps) |
| 0.40 | **+6.4 pts** (hurts) | +0.0 pts |

The **same rule flips sign with the dataset.** On pyro-sdis, requiring persistence *hurts* at
the tight operating points — the confusers are persistent, so persistence keeps them while
costing recall. On FIgLib, requiring persistence *helps* by up to 13.8 points — the pre-ignition
negatives are genuinely transient (empty sky, momentary artifacts), exactly the flicker a
temporal rule is meant to suppress. This is the mechanistic claim confirmed from the opposite
direction: temporal context pays off on onset data and backfires on established-scene data.

## What did not work, and why it doesn't undercut the above

The learned GRU failed on FIgLib (worse than single-frame at every operating point). That is
expected and uninformative here: it was learning from a feature stream that is itself ~random
(AUC 0.45). Garbage features, garbage head. The persistence rule is informative precisely
because it sidesteps the broken features and operates on the raw confidence trajectory.

## Honest caveats

- **Zero-shot base signal is near-random (AUC 0.45).** Absolute false-alarm rates on FIgLib
  (40–78%) are not trustworthy; only the *matched-recall sign* of the persistence effect is.
- **18 sequences, 4 held-out test fires.** Small. The sign-flip is directionally clear but the
  magnitudes are noisy.
- Proof scale throughout.

## What a conclusive positive control requires (future work)

Train an **in-domain, tiled** detector on FIgLib (SmokeyNet's setup: 224-px tiles, CNN
backbone, bbox labels ship with the dataset), then run the same GRU-vs-single-frame comparison.
With a real per-frame signal, the learned temporal model — not just the persistence rule —
can be tested. The prediction, given the persistence sign-flip above, is that temporal will
win on FIgLib as clearly as it lost on pyro-sdis.

## Reproduce

    # data: subset of FIgLib sequences into data/figlib/images/<seq>/ (gitignored)
    python src/models/figlib_temporal.py --extract     # cache features
    python src/models/figlib_temporal.py               # AUC check + matched-recall comparison

Comparison: `results/figlib_temporal_comparison.json`. pyro-sdis counterpart:
`results/temporal_comparison.json`. See also [temporal-findings.md](temporal-findings.md).
