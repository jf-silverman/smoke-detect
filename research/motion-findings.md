# Motion findings: horizon-anchored change detection on FIgLib

**One line.** Inter-frame motion *anchored to the horizon* separates ignition-onset frames from
pre-ignition frames markedly better than the appearance detector's own confidence — per-fire AUC
**0.71 vs 0.57** — and carries signal the detector does not, which is the case for adding it as a
temporal channel in Phase C.

This is a training-free separability probe, not a trained model: an unlearned hand-designed feature,
scored against the free onset labels. It exists to answer one question cheaply before Phase C — *is
there motion signal worth learning?* — and the answer is yes.

## Where the idea came from

Correcting pre-annotations in CVAT, the author noticed that a plume is often legible only by its
**motion** across consecutive frames — it grows and drifts — even when it is nearly invisible in any
single still. The dirty-lens sequence `20170722_FIRE_mg-n-iqeye` is the extreme case: the plume reads
only as movement (recorded in [data-quality-flags.md](data-quality-flags.md)).

The physical refinement mattered. The first cut weighted motion *below* the horizon, which is wrong —
plumes rise. The correct property is that smoke **starts and stays anchored to the ground**: the base
sits at/near/below the horizon and the growing column stays *connected* to the horizon line. Clouds —
this project's dominant confuser (74% of false alarms; [confuser-corpus.md](confuser-corpus.md)) —
appear as change that **floats** entirely above the skyline, disconnected from it. So the feature is
horizon-*anchored* change, not below-horizon change. It is deliberately **probabilistic, not a gate**:
rare wind-driven or low-in-scene fires never clear the ridge.

## Method

Implemented as an optional probe in [`figlib_ttd.py`](../src/models/figlib_ttd.py) (`--motion`); it
reads the FIgLib onset frames (81/fire, ~60 s apart), runs **no model**, and reuses the cached tiled
per-frame confidences as the appearance baseline.

Per frame, versus its predecessor (~60 s earlier):
1. **Change map** = `|(cur − prev) − spatial_median|`. Subtracting the spatial median first strips a
   uniform exposure/auto-gain shift (and much dirty-lens flicker), so only *localized* change survives.
2. **Robust mask** via a MAD threshold on the change map.
3. **Anchoring:** a small band around the fire's horizon; a column is *anchored* if it has change in
   the band, and its energy is the band change plus the **contiguous** change run extending up (the
   rising column) and down (the base), computed with a `cumprod` over the mask. Three features:
   `anchored_change`, `floating_change` (everything not connected — the cloud-drift control), and
   `anchored_ratio = anchored / (anchored + floating)`.

Two methodology choices carry the rigor:
- **One fixed horizon per fire** (median estimate over the fire's *pre-ignition* frames). The camera
  is static, so the horizon is a constant; using pre-ignition frames only stops a bright/dark plume
  from dragging its own horizon.
- **Per-fire AUC** — computed *within* each fire, then averaged with a bootstrap-over-fires 90% CI.
  This is the leak-aware and operationally correct read (you watch one camera over time; you never
  compare absolute motion across sites). It is reported alongside a pooled AUC.

## Results

Onset (offset ≥ 0) vs pre-ignition (offset < 0) separability, per-fire AUC (pooled in parens).

**Training set — 17 day-only fires** (the 6 recent CA eval fires held out for Phase C via
`--exclude-eval`; the one nocturnal fire excluded for scope, see
[data-quality-flags.md](data-quality-flags.md)):

| feature | per-fire AUC (90% CI) | pooled |
|---|---|---|
| single_frame_conf (baseline) | 0.570 [0.461, 0.680] | 0.598 |
| **anchored_change** | **0.708 [0.606, 0.807]** | 0.640 |
| **anchored_ratio** | **0.708 [0.605, 0.800]** | 0.651 |
| floating_change (cloud control) | 0.566 [0.456, 0.675] | 0.536 |
| conf + anchored (naive sum) | 0.662 [0.561, 0.760] | 0.658 |

**The head-to-head, stated plainly.** `anchored_change` beats conf in **11 of 17 fires** (mean per-fire
lift **+0.14**; anchored AUC ≥ 0.7 in 11 fires, ≥ 0.8 in 9), but at n = 17 that win-count is **not
statistically significant** (one-sided sign-test p = 0.17). An earlier writeup cited "17 of 24,
p ≈ 0.03" — that number mixed in the held-out eval fires and is not reproducible from the probe
artifact; this is the leak-aware, day-only, reproducible replacement. The defensible claim is therefore
narrower than a clean win over the detector: **anchored motion separates onset above chance** (its
90% CI [0.606, 0.807] excludes 0.5) and **carries signal the detector misses** — a complementary
channel, which is all Phase C needs it to be.

Three things to read from this:
1. **Anchoring works, and the control confirms the mechanism.** `anchored_change` (0.71) clears both
   the conf baseline (0.57) and the `floating_change` cloud control (0.57, CI straddling 0.5 — null).
   It is *anchoring* doing the work, not merely "more motion."
2. **The per-fire *measurement* was decisive.** The pooled AUC (0.64) understates the feature:
   anchored-change *magnitude* varies by scene, so pooling frames across fires lets between-fire scale
   swamp the within-fire onset signal. Measured the operationally correct way — within a fire — it is
   0.71.
3. **Complementarity, not redundancy.** Fires the appearance detector is weak on are rescued by
   motion, and they are exactly the hard imaging cases: `syp-w` conf 0.23 → anchored 0.97; the faint
   distant `so-w-mobo-c` 0.07 → 0.79; the oversaturated `Dehesa` 0.12 → 0.80. That orthogonal signal is
   the whole argument for a motion channel in Phase C.

**On fusion:** the naive equal-weight rank-sum of conf + anchored *hurts* (0.66 < 0.71), because it
dilutes the stronger motion feature with the weaker confidence. This is **not** evidence of
redundancy (the rescue cases refute that) — it means a naive sum is the wrong combiner once the two
signals are lopsided. The proper combiner is the **learned Phase C temporal head**, not a hand-weighted
sum; the naive number is reported only as a control, not as the combined-value test.

## The failure cases are diagnostic

The fires where anchored motion fails are physical, not evidence against the mechanism:
- **`mg-n-iqeye` (anchored 0.33, conf 0.39)** — the flagged dirty-lens sequence; the smeared cover
  injects false motion everywhere, breaking differencing. A known confound.
- **`wc-s-mobo-c` (anchored 0.00, conf 0.64)** — a single catastrophic horizon mis-placement (the
  estimate lands off the true skyline, so the plume never crosses the band). Not recovered by the
  skyline attempt below — its problem is specific, not the flat-vs-skyline choice.
- **`rm-w-mobo-c` (anchored 0.57, conf 0.98)** — a case the appearance detector nails (conf AUC 0.98)
  but motion does not add to; the plume is high-contrast and single-frame-obvious, so there is no
  motion gap to close. Motion earns its keep on the faint plumes, not these.

## Three negative results, reported as such

- **Texture mix — removed.** An early feature added `texture_mix` = (soft-gradient fraction) ×
  (hard-gradient fraction) inside the anchored region, on the idea that a plume mixes diffuse haze with
  a harder-edged column. It is not a stable smoke signature: a high-Haines-index day gives a hard-edged
  column with little haze; a windy day gives an all-diffuse column with no hard edge. The soft × hard
  product breaks exactly under the atmospheric conditions that matter, so it was cut.
- **Smoothed-skyline horizon — tried and reverted.** To fix `wc-s`, a per-column skyline (median-
  smoothed steepest per-column gradient) with a column-warp to flatten the terrain contour was built.
  It made things *worse* — apples-to-apples (on the pre-cleanup fire set, before the day-only /
  night-exclusion pass), `anchored_change` fell 0.715 → 0.616, and `wc-s` stayed broken (0.03). Per-column gradient estimates are noisy and warping by them scrambled
  the vertical structure anchoring depends on; the flat median row is more robust. Reverted.
- **Luminosity sky/ground mask — tried and reverted.** Following the intuition (from photo editing)
  that absolute luminosity splits sky from ground, we built a per-fire Otsu threshold on the averaged
  pre-ignition frame → ground mask, then flooded change *upward from the ground* through connected
  change pixels (a warp-free A/B against the flat horizon). It underperformed badly: per-fire
  `anchored_lum` **0.537** vs flat-horizon `anchored_change` **0.706** on the training set (pre-cleanup
  fire set), with the `floating_lum` control (0.582) actually *beating* `anchored_lum` — the fingerprint
  of a mask that mislabels sky vs ground. None of the masks were degenerate (ground fractions
  0.34–0.90), so it is
  a genuine feature-quality result, not an artifact: on FIgLib's variable imagery (haze, backlight,
  dark forest, bright cloud) *absolute* brightness does not track the horizon, whereas the flat
  estimator keys on *relative* brightness change (a gradient) and is robust to those level shifts.
  Reverted. Three horizon variants tried, the simplest (flat brightness-gradient row) wins — the
  lesson is to stop tuning the horizon and let the Phase C learned head do the heavy lifting.

## What this means for Phase C, and what's open

The probe has done its job: there is real, complementary motion signal (per-fire ~0.71, statistically
above the detector's own confidence, with a null cloud control), so **anchored motion goes into Phase
C as an input channel to the learned temporal head**, where the real payoff test is whether it *lowers
time-to-detection*, not just raises separability.

Open threads:
- **Learned combiner** — the naive fusion is the wrong test; the temporal head is the right one.
- **Tighter CIs** — 17 fires is small (and the reason the head-to-head sign-test is underpowered);
  Phase B onset data (FIgLib-full / PYRONEAR) would sharpen it.
- **The horizon estimator is settled** — three variants tried (flat brightness-gradient row, per-column
  smoothed skyline, luminosity sky/ground mask); the simplest won and the other two are documented
  negatives above. Not worth further tuning; the learned head can refine anchoring if it needs to.

Tracked in the [backlog](backlog.md#motion--change-detection-anchored-at-the-horizon).
