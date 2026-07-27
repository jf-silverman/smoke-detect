# Data-quality flags (FIgLib)

A running log of FIgLib sequences with a known imaging defect that could confound training or
evaluation. **Flagged, not removed** — the frames stay in the data until we decide a defect
actually distorts a result. The point is to keep an auditable record so a later anomaly can be
traced to a known cause rather than mistaken for signal.

| sequence | frames | defect | status | noticed |
|---|---|---|---|---|
| `20170722_FIRE_mg-n-iqeye` | 81 total (30 in the Phase C pre-annotation bundle) | **Dirty lens cover.** Greyish-blue fuzzy blobs across the frame, most visible against the sky and the haze-faded distant mountains. The plume is hard to see in any single frame — it reads only as *movement* across successive frames. | flagged; kept in data | 2026-07-27, during CVAT correction |

## Why keep it for now

The plume being legible only through inter-frame motion (not in a still) is itself informative —
it is a concrete example of the motion cue discussed in the [backlog](backlog.md#motion--change-detection-below-the-horizon)
and a natural stress-test frame set for any temporal/differencing detector. Removing it would
discard exactly the hard case worth studying. If it later shows up as a systematic false-negative
cluster in a single-frame eval, revisit whether to down-weight or hold it out.
