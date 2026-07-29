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

## Generalization to the held-out recent-CA fires

Measured standalone on the 6 held-out `EVAL_FIRES` (the Phase C test set, never used to tune the
probe), `anchored_change` per-fire AUC is **0.742 [0.648, 0.850]** — CI excludes 0.5, and a touch
*higher* than the 0.708 on the training fires — with the `floating_change` control at **0.554** (near
chance). So the anchoring cue **transfers to recent California smoke it was never tuned on**. It beats
the fine-tuned Phase C detector's confidence in only 2/6 fires (mean lift +0.05): the stronger
in-distribution detector narrows the complementarity gap, but the naive fusion (0.756) still edges
either single signal, hinting at residual orthogonal signal. See [phase-c-findings.md](phase-c-findings.md).

## The learned per-frame combiner — tested (2026-07-28): a mixed, appearance-dominated result

The motion-findings above promised a *learned* combiner in place of the naive sum. The first, simplest
one — a **per-frame logistic regression** over `[conf, anchored_change, anchored_ratio, floating_change]`
→ onset probability ([`figlib_fusion.py`](../src/models/figlib_fusion.py)) — is now run. It is
**leak-free by construction**: the appearance `conf` is the zero-shot base detector (`gcp_grouped_1280`,
never trained on FIgLib) and the motion features are training-free, so the LR is *fit on the 17 training
fires and applied to the 6 held-out `EVAL_FIRES`* with no leakage. The fused score is written as a
`conf_tiled` npz and scored through the unchanged TTD harness.

The verdict is **mixed and underwhelming** — worth recording plainly rather than dressing up.

**Learned weights (standardized — sign and magnitude comparable):**

| feature | weight |
|---|---|
| conf | **+0.773** |
| anchored_ratio | +0.175 |
| anchored_change | −0.101 |
| floating_change | −0.043 |

The combiner leans overwhelmingly on **appearance**. Motion enters mainly through `anchored_ratio`
(+0.18); `anchored_change` even takes a small *negative* weight (the two anchored features are
correlated, so the LR splits the credit), and `floating_change` — the cloud control — is correctly
near zero. So a *linear* combiner extracts only a little from motion once a strong appearance detector
is in the mix.

**Frame-level AUC — the discouraging part.** A leave-one-fire-out fit on the *training* fires makes
cross-fire generalization **worse**, not better:

| training LOO per-fire AUC | value |
|---|---|
| conf-alone | **0.689** |
| fused (conf + motion) | 0.623 |

On the held-out eval fires the *pooled* AUC does rise (conf 0.728 → fused 0.793), but per fire it is
inconsistent — it helps exactly one of the two weak fires and hurts the other:

| held-out fire | conf-alone | fused |
|---|---|---|
| Vista (2024) | 0.729 | **0.531** ↓ |
| Tenaja (2024) | 0.982 | 0.976 |
| Bahrman (2024) | 0.956 | 0.959 |
| Palisades (2025) | 0.640 | **0.462** ↓ |
| Highway (2025) | 0.945 | 0.940 |
| Coches (2025) | 0.730 | **0.852** ↑ |
| **pooled** | 0.728 | 0.793 |

The pooled gain is **Coches-driven** (0.73 → 0.85); Vista and Palisades — the other two soft fires —
get *worse*. So the linear fusion is not a reliable frame-ranking improvement, and the training LOO
says it can actively hurt.

**Operational (TTD / detection) — a modest, noisy positive that points the other way.** Scored through
the TTD harness, the fused signal detects **more fires** and, at matched false-alarm rate, is
**comparable-to-faster**. Per fire at the 5% pre-ignition-FA target, leave-one-fire-out:

| fire | base conf | fused |
|---|---|---|
| Vista | miss | miss |
| Tenaja | 4.0 min | 4.0 min |
| Bahrman | 2.9 min | **0.9 min** |
| Palisades | 0.0 min | 1.0 min |
| Highway | 10.0 min | **8.0 min** |
| Coches | **miss** | **33.0 min** ← rescued (late) |
| **detection** | **4/6** | **5/6** |
| achieved FA | 16.7% | 15.0% |

At **matched ~16% achieved FA** (from the FA-budget sweep), the picture favors fusion:

| at ~15–17% achieved pre-ignition FA | detection | median TTD |
|---|---|---|
| base conf (`gcp_grouped_1280`) | 4/6 | 2.44–3.45 min |
| **fused (conf + motion)** | **5/6** | **1.0 min** |

So operationally the fusion detects one more fire (rescues Coches, though at a barely-useful 33 min)
and is at least as fast — faster on Bahrman and Highway — at equal-or-lower FA.

**Reconciling the two reads.** Frame-AUC and TTD disagree because they measure different things:
AUC scores *overall frame ranking*, TTD scores the *first onset frame to cross threshold*. Motion can
lower TTD (help the early-onset frames of Bahrman/Highway, rescue Coches) while adding noise elsewhere
(hurting Vista/Palisades pooled ranking). But this is a fragile, **n = 6** operational win resting
heavily on one late Coches rescue, and it partly overlaps what the Phase C appearance fine-tune already
achieved (Coches rescue, lower FA) by a different route.

**Bottom line.** A per-frame *linear* combiner underdelivers: it leans on appearance, its
cross-validated training AUC drops below conf-alone, its eval frame-AUC is inconsistent, and its
operational lift is small and noisy. This is a legitimate negative-leaning result — *a lopsided-signal
problem does not become a win just by learning linear weights.* Its value is diagnostic: the signal
that helps TTD lives in the **onset transition** (the early frames), which a per-frame model cannot
target by construction. That is precisely the case for a **temporal (sequence) model** that can weight
the change *over time* — the next test, run with clear-eyed expectations rather than hope.

## What this means for Phase C, and what's open

The probe has done its job: there is real motion signal that **separates onset above chance** on both
training and held-out fires (per-fire ~0.71–0.74, CI excluding 0.5, with a null cloud control) and is
*complementary* to the appearance detector — though the head-to-head win-count over the detector is
not itself significant at these sample sizes. So **anchored motion goes into Phase C as an input
channel to the learned temporal head**, where the real payoff test is whether it *lowers
time-to-detection*, not just raises separability.

Open threads:
- **Learned combiner — first cut done, underwhelming (see section above).** The per-frame LR is
  appearance-dominated and its frame-AUC is mixed-to-negative; a small operational (TTD) lift exists
  but is n=6-noisy. The **temporal sequence head** (weighting the onset transition over time) is the
  next test — the per-frame result motivates it precisely because a per-frame model *can't* target the
  onset transition.
- **Tighter CIs** — 17 fires is small (and the reason the head-to-head sign-test is underpowered);
  Phase B onset data (FIgLib-full / PYRONEAR) would sharpen it.
- **The horizon estimator is settled** — three variants tried (flat brightness-gradient row, per-column
  smoothed skyline, luminosity sky/ground mask); the simplest won and the other two are documented
  negatives above. Not worth further tuning; the learned head can refine anchoring if it needs to.

Tracked in the [backlog](backlog.md#motion--change-detection-anchored-at-the-horizon).
