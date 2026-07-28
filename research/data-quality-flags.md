# Data-quality flags (FIgLib)

A running log of FIgLib sequences that could confound training or evaluation — split into **imaging
defects** (an optical/sensor problem in otherwise in-scope frames, flagged but kept) and
**structural anomalies** (the sequence violates an assumption baked into the analysis harnesses).
The point is to keep an auditable record so a later anomaly can be traced to a known cause rather
than mistaken for signal. Most items are flagged, not removed; the exception is the night fire,
which is out of scope and now excluded in code.

## Imaging defects (flagged, kept)

| sequence | frames | defect | status | noticed |
|---|---|---|---|---|
| `20170722_FIRE_mg-n-iqeye` | 81 total (30 in the Phase C pre-annotation bundle) | **Dirty lens cover.** Greyish-blue fuzzy blobs across the frame, most visible against the sky and the haze-faded distant mountains. The plume is hard to see in any single frame — it reads only as *movement* across successive frames. | flagged; kept in data | 2026-07-27, during CVAT correction |
| `20190728_Dehesa_lp-n-mobo` | 80 total (30 in the Phase C pre-annotation bundle) | **Sensor oversaturation.** The upper side of the plume is blown out into clipped/overexposed highlights, so its top edge is lost against the bright sky — only the lower plume is legible. Compounds the general coarseness and color shift of these HPWREN stills. | flagged; kept in data | 2026-07-28, during CVAT correction |

## Structural / label anomalies (violate a harness assumption)

| sequence | anomaly | who's affected | handling |
|---|---|---|---|
| `20191006_FIRE_lp-e-mobo-c` | **Two spatially separated fires in one view** (Lyons Peak East). Fire A sits at the far-left edge (`cx≈0.01, cy≈0.48`) and is present from the labeled ignition; Fire B appears on the right (`cx≈0.78–0.80, cy≈0.50`) only at **~+13 min**. Both hand-boxed in CVAT. | The harnesses assume **one fire, one ignition per sequence**: TTD scores time-to-detection against the single labeled t=0 (wrong clock for B), and the motion probe's onset-vs-pre-ignition split mislabels the +0→+13 window (onset for A, pre-ignition for B). | kept; **fine as Phase C training data** (YOLO is natively multi-instance). Flagged for the TTD/motion aggregates — see handling note below. | noticed 2026-07-28, during CVAT correction |
| `20201202_BondFire-nightime_sp-w-mobo-c` | **Night fire.** Optical smoke detection collapses to flame/glow detection at night; the whole field (and the canonical FIgLib benchmark) is day-only. Leaked in because no day/night filter existed. | Had **0 boxes** in the pre-annotation bundle (nothing to correct in CVAT). But it was in the TTD 24-fire set (counted as a *detection* at 23 min via glow, not plume) and the 18-fire motion set. | **Excluded in code** now: name-based `NIGHT_PATTERN` drop in both `src/data/figlib_preannotate.py` and `src/models/figlib_ttd.py` (the latter unconditional). Numbers become day-only after the next TTD/motion rerun. | noticed 2026-07-28, during CVAT correction |

## Why keep the imaging defects for now

**Dehesa (oversaturation).** A clipped upper plume is a real field condition — a bright sky behind a
plume routinely blows out the thin, translucent top edge — so a detector that must work on tower
imagery has to cope with it rather than be spared it. Kept as a hard positive; the note flags that a
box on this plume necessarily covers only the legible lower portion, not the true top extent, which
is relevant if these boxes are ever used to judge localization tightness.

**Dirty lens (motion-only legibility).** The plume being legible only through inter-frame motion (not in a still) is itself informative —
it is a concrete example of the motion cue discussed in the [backlog](backlog.md#motion--change-detection-anchored-at-the-horizon)
and a natural stress-test frame set for any temporal/differencing detector (see
[backlog: Motion / change-detection anchored at the horizon](backlog.md#motion--change-detection-anchored-at-the-horizon)).
Removing it would
discard exactly the hard case worth studying. If it later shows up as a systematic false-negative
cluster in a single-frame eval, revisit whether to down-weight or hold it out.

## Handling the two-fire sequence (`20191006`)

Three options, in increasing effort:

1. **Keep for training, document for analysis (current choice).** It stays a valid multi-instance
   positive for Phase C. For the TTD/motion aggregates it is 1 of 23 (TTD) / 17 (motion) fires and the distortion is
   small and *conservative* — the muddled +0→+13 window can only make the motion probe look *worse*
   (some true-onset-of-B frames are labeled pre-ignition), so it does not inflate our headline. Left
   in, flagged here.
2. **Drop it from the motion per-fire AUC only.** The onset-vs-pre-ignition split is genuinely
   ill-defined for this sequence, so excluding it from the motion probe (while keeping it for TTD and
   training) removes the noise at its actual source. A one-line `seq != ...` guard, cheap; worth doing
   if the motion result is ever reported as a primary number rather than a probe.
3. **Model two fires explicitly.** Give Fire B its own onset (~+13 min, estimable from when its boxes
   first appear) and let the harness track fires, not sequences. Most correct, but it means the
   harness must support multiple fires per sequence — a real change for one edge case. Deferred unless
   more multi-fire sequences surface.

Recommendation: **stay on (1)** through the current proof-scale stage, and switch the motion probe to
(2) if/when it graduates from probe to reported metric.
