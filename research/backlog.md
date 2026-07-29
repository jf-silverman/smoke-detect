# Backlog — experiments to try

A running list of ideas worth pursuing, with enough context to pick any one up cold. Ordered
roughly by leverage. Completed threads have their own findings reports (see the [README](../README.md)).

## Next up — the chosen thrust (combine-levers run is done)

- **HPWREN → time-to-detection (and a real learned temporal model).** The field's headline metric
  (Pano, SmokeyNet's 3.1 min) is time-to-detection, and this project conspicuously lacks it because
  pyro-sdis has no ignition-onset sequences. Bring in the HPWREN archive / PYRONEAR-2025 onset
  clips to (a) compute **time-to-detection** and (b) train a *learned* temporal model on enough
  onset fires to finally beat the parameter-free persistence rule (18 FIgLib fires were far too
  few). Concrete data path + TTD-eval sketch in [Pre-scoping #3](#pre-scoping-3--hpwren--time-to-detection)
  below. Builds on [temporal-findings.md](temporal-findings.md) and [figlib-findings.md](figlib-findings.md).
  - **Phase A — DONE** ([ttd-findings.md](ttd-findings.md)): leak-safe LOFO TTD harness
    ([`figlib_ttd.py`](../src/models/figlib_ttd.py)) on 24 local fires (expanded from 18 after
    fixing an extraction bug that hid the nested 2024–2025 archives). Zero-shot headline
    50% detection / median TTD 8 min at a 5% pre-ignition FA budget; resolution lowers TTD, not
    just AUC. Wide CIs (small n); the recent CA fires (Palisades, Coches, Tenaja) all missed —
    the case for Phase C.
  - **Phase B — next:** pull more onset fires (FIgLib-full via WIFIRE Commons; PYRONEAR-2025) to
    tighten the CIs. FIgLib-full (~20–25 GB) fits local; PYRONEAR video set may want cloud
    ([gcp-plan.md](gcp-plan.md)).
  - **Phase C — DONE (first result, 2026-07-28), see [phase-c-findings.md](phase-c-findings.md).**
    Fine-tuned `gcp_grouped_1280` on the hand-corrected day-only set (17 fires, 452 boxes, 85
    negatives) on a GCP L4 (~$0.35). Held-out eval on the 6 recent CA fires: detection **4/6 → 5/6**
    (Coches rescued), pre-ignition FA **16.7% → 7.1%**; anchored motion generalizes (0.742 on the
    held-out fires). Corrected an overclaim: the full-scale base already gets Palisades/Tenaja, so the
    "recent miss" was partly proof-scale. **Still open:** a *learned* temporal head fusing motion
    (payoff = lower TTD), Phase B more fires to tighten n=6, matched-FA comparison.
    - _(original plan, for the record)_ train an in-distribution detector on FIgLib boxes → the AUC
    0.658 floor rises and TTD should drop toward the SmokeyNet range; then test a *learned* temporal
    head. Best run on a cloud GPU ([gcp-plan.md](gcp-plan.md)) rather than the ~92-min/epoch local box.
    - **Data-source decision (2026-07): pseudo-label our own local FIgLib fires.** Phase C needs
      bounding boxes, which canonical FIgLib lacks (only onset labels via the filename offset). The
      external box sources were evaluated and set aside:
      - **PYRONEAR-2025** (Lostanlen et al., 2024; arXiv:2402.05349) — *eliminated.* Its US data is
        all old HPWREN SoCal (2016–2021) + ALERTWildfire Nevada (2015–2019); its only 2024–2025 fires
        are Chile/France, so it cannot close the recent-CA miss (it predates the Jan-2025 Palisades
        fires). Worse for leak: its image pool literally includes AI-For-Mankind, and its video test
        set shares 4 same-date HPWREN fires with our 24 TTD fires.
      - **AI-For-Mankind** — real HPWREN boxes but leak-entangled (pHash audit: 21% overlap, 8
        exact-duplicate FIgLib test fires); no clean cutoff. Usable only with those fires excluded.
      - **D-Fire** (Brazil) — leak-free but cross-distribution; raises the box-training floor without
        addressing California specifically.
      - **Chosen path — pseudo-labeling.** We already hold the frames and onset labels for our local
        fires; only boxes are missing. Pre-annotate positive frames (offset ≥ 0) with the salvaged
        1280 detector (`best.pt`), hand-correct in a box tool (makesense.ai / CVAT / Roboflow Label
        Assist), gate boxes on the onset labels so the model never labels a pre-ignition frame, and
        hold out **whole fires** (train on a disjoint fire set; keep the 24 TTD fires for eval only).
        Zero external leak, most in-distribution, and correction effort concentrates on the tiny
        early plumes that matter most for TTD. Main risk (error amplification) is defused by the
        human-review step — semi-supervised, not naive self-training.
      - **Training-set composition — add pre-ignition negatives (planned).** The pseudo-label
        bundle (`figlib_preannotate.py`) is all positive-offset frames (531 frames; 220 with boxes,
        311 empty). The empty post-ignition frames already act as some background negatives, but the
        set has **no clear-sky / cloud negatives**, exactly the confuser class this project's headline
        failure mode fires on (74% of false alarms are clouds; the converged 1280 model still
        false-alarms on 57.6% of clean frames — see [hard-negative-findings.md](hard-negative-findings.md)).
        So mix in a **negatives bundle** at training time (no CVAT work — negatives are empty-label by
        definition):
        - *Source:* pre-ignition frames (use a safe margin, offset < −120 s, to avoid the ambiguous
          just-before-ignition boundary) from the **18 training fires only** — the 6 recent eval
          fires stay entirely out of training, pre-ignition frames included, preserving whole-fire
          holdout.
        - *Easy negatives:* a random sample of those pre-ignition frames (identical scene/weather to
          the positives, minus the plume — teaches the model the difference is the plume, not the
          site).
        - *Hard negatives:* run the detector over the pre-ignition frames and preferentially include
          the ones where it **false-alarms** (high-conf pre-ignition boxes = cloud/glare confusers) —
          the same hard-negative-mining lever that cut the burden ~23% before, now sourced in-distribution.
        - *Ratio:* start ~1:1 negatives : boxed-positives (~220 each), skewed toward hard negatives;
          tune against recall (too many negatives erodes the recall ceiling).
        - *Implementation:* extend `figlib_preannotate.py` with a `--negatives` mode that emits
          pre-ignition frames + empty labels into a separate set, merged with the CVAT-corrected
          positives when the Phase C training config is built.

## Net-new California data — evaluated 2026-07-28 (next data steps)

The recurring bottleneck is recent-California onset data that does **not leak** into the FIgLib /
recent-CA eval. Our base is pyro-sdis (French towers); our eval is FIgLib (HPWREN So-Cal cameras) plus
the 6 recent-CA `EVAL_FIRES` (2024–2025). **Leakage lens:** any HPWREN-derived set shares cameras and
sequences with FIgLib and is leaky by construction — this downgrades the two most convenient sources
([AI For Mankind](https://github.com/aiformankind/wildfire-smoke-dataset), 2,192 VOC-boxed HPWREN
frames; the raw [HPWREN archive](https://www.hpwren.ucsd.edu/news/20180501/)) to training-only-with-
hard-exclusion.

- **Nemo — the pullable quick win (queued next data add).** COCO-format smoke boxes from **1,073
  ALERTWildfire PTZ videos** (Nevada + CA), ~2,564 train / 250 val images, 3 fine-grained smoke
  classes + a 100-image hard-negative set; incipient-stage framing matches our TTD goal (Yazdi/
  SayBender et al., 2022; [GitHub](https://github.com/SayBender/Nemo),
  [paper](https://www.mdpi.com/2072-4292/14/16/3979)). One `git clone`, no request. ALERTWildfire ≠
  HPWREN → **no FIgLib overlap**, and it broadens beyond both HPWREN and French towers. *Action:* pull,
  inventory against our smoke label schema, gate against any camera/date in our eval, then use as
  extra Phase-C training data. Cheap; do this before the ALERTCalifornia thread.
- **ALERTCalifornia archive — the recent-CA distribution-shift prize (Phase D, by request).** UCSD's
  statewide network: **1,200+ cameras**, and critically the **2023+ era** — the same generation and
  geography as our 6 recent-CA eval fires, so it's the source that could genuinely close the recent-CA
  gap rather than just add volume ([FAQs](https://alertcalifornia.org/faqs/);
  [Wikipedia](https://en.wikipedia.org/wiki/ALERTCalifornia);
  [CA GIS camera layer](https://gis.data.ca.gov/documents/California::alertcalifornia-fire-cameras/explore)).
  Data is open-source but **not a clean API** — request footage as a scientist/partner via
  `alertcalifornianews@ucsd.edu`; it arrives *unlabeled*, so it feeds the **pseudo-label → hand-correct
  in CVAT** loop already built for Phase C. Multi-week thread (email → footage → labeling), so it's a
  Phase D decision, not a today one — but the eval-era match makes it the highest-leverage data play on
  the board. *Caveat:* whole-fire holdout must be preserved (any fire that appears in `EVAL_FIRES`
  stays out of training).

## Recently completed (folded into reports)

- **Combine the two levers — native-res training + hard-negative mining (full scale, @1280).**
  The deployable recipe, tested. Mining with the converged 1280 model (which still false-alarms on
  57.6% of clean training frames — resolution alone doesn't fix confusers) + a gentle ~22% hard-neg
  share. Result: false-alarm burden ~173 → ~133 FP/camera/day (−23%), precision@1% 2.3% → 2.9%,
  best moderate-cost REV — at the cost of ~2.4 pts recall ceiling (0.83 → 0.80). A real but
  *incremental* win; 133 FP/day is still far from < 1. Stopped at epoch 24 (val plateaued). See
  [hard-negative-findings.md](hard-negative-findings.md), [metrics.md](metrics.md).
- **Full-scale native-resolution (1280) training** — confirmed the resolution-findings prediction:
  the proof 1280 run only underperformed because it was undertrained. Full run reaches POD 0.827
  at ~173 FP/camera/day (roughly half the burden of downscaled-inference-only), the best config so
  far. See [resolution-findings.md](resolution-findings.md), [metrics.md](metrics.md).

## Queued

- **D-Fire zero-shot cross-dataset evaluation.** Train on pyro-sdis, evaluate *cold* on
  [D-Fire](https://github.com/gaiasd/DFireDataset) (~21k images, Brazil, YOLO-format smoke+fire
  boxes) with no fine-tuning. Measures cross-dataset distribution shift — the most demanding
  generalization number in the project, extending the leak-safe theme from across-towers to
  across-datasets. Reuse `evaluate.py` almost verbatim; filter to the smoke class and note the
  label-definition mismatch (D-Fire leans closer-range and mixes fire). The FIgLib work already
  gave an *accidental* distribution-shift datapoint (pyro-sdis → FIgLib collapsed to AUC 0.45 before
  the resolution fix); D-Fire turns that into a controlled, box-labelled result.

- **RL for the alarm-timing decision (optimal stopping under asymmetric cost).** RL is a poor
  fit for the *detection* itself (a supervised, single-image problem — no one in the field uses
  it there), but it fits the *when-to-alarm* decision naturally. Frame it as sequential decision-
  making over the frame stream: at each step, alarm or wait, with a reward that heavily penalizes
  a missed or late detection and only lightly penalizes a false alarm — the RL-shaped encoding of
  the [asymmetric-cost / recall-first](resolution-findings.md) objective. Closely related to
  the early-classification-of-time-series problem. Builds directly on the temporal work
  ([temporal-findings.md](temporal-findings.md), [figlib-findings.md](figlib-findings.md)).
  Precedent for RL *around* detection (not the classifier): EcoWild (energy-adaptive sensing),
  ForestProtector (PTZ camera orientation control).

## Motion / change-detection anchored at the horizon

**Observation (author, 2026-07, during CVAT correction).** Scanning consecutive frames, the eye
locks onto the plume by its *motion* — it grows and drifts frame-to-frame — even when it is nearly
invisible in any single still (the dirty-lens `20170722_FIRE_mg-n-iqeye` sequence is the sharp case:
the plume reads only as movement; see [data-quality-flags.md](data-quality-flags.md)). The proposal:
the detector should exploit the **difference from the previous frame** — an object that *moves*
between frames — and privilege change that is **anchored to the horizon**.

**The anchoring is the sharp part (refined 2026-07).** The first cut weighted motion *below* the
horizon, which is wrong: plumes *rise*, so early separability was actually strongest **above** the
skyline (2-fire probe). The correct property is not that the smoke stays low but that it **starts and
stays connected to the ground**: the base sits at/near/below the horizon and the growing column
remains *connected* to the horizon line. Clouds — the dominant confuser (74% of false alarms) —
appear as change that **floats** entirely above the skyline, disconnected from it. So the feature is
**horizon-anchored change** (a contiguous vertical run of change that touches a band around the
horizon and extends up the rising column and down to the base) vs **floating change** (cloud drift).
Explicitly **probabilistic, not a gate**: rare wind-driven fires or fires low in-scene behind high
terrain never clear the ridge, so anchoring is a likelihood bump, not a hard rule.

**Texture cue — tried and REMOVED (author, 2026-07).** A first version added `texture_mix` =
(soft-gradient fraction) × (hard-gradient fraction) inside the anchored region, on the idea that a
plume mixes diffuse haze with a harder-edged column. Dropped: it is not a stable smoke signature. On
a **high-Haines-index** day the column has hard edges and little to no surrounding haze; on a **windy**
day the column is almost entirely diffuse with no hard edge (and is the hardest type to detect). So
the soft × hard product is noise, not signal, and the atmospheric conditions that break it are exactly
the ones that matter. Removed for now.

**How this differs from what we already tested.** The temporal work so far
([temporal-findings.md](temporal-findings.md), [figlib-findings.md](figlib-findings.md)) tested
**persistence** — requiring a *detection* to recur across k frames to suppress flicker. That is the
opposite operation from **motion/change** — surfacing a region *because* it changed. Persistence
suppresses; differencing proposes. On FIgLib (onset data) persistence already *helped* by 12–19 pts,
so the data supports temporal signal there; anchored differencing is the untested, potentially
stronger sibling.

**Preconditions & confounders.** Cameras are fixed (FIgLib, pyro-sdis) so frame differencing needs
no registration in principle — but PTZ moves, wind-shake, exposure/auto-gain shifts, and dirty-lens
artifacts all inject false motion. The implementation strips a uniform exposure/gain shift by
subtracting the per-frame spatial median of the signed difference before rectifying; wind-moved
clouds (the primary false-motion source) are exactly what the anchoring is meant to reject.

**Status — thin probe IMPLEMENTED** ([`figlib_ttd.py`](../src/models/figlib_ttd.py) `--motion`).
Reads the FIgLib onset frames (81/fire, ~60 s apart; no model, no GPU, cached to
`data/figlib/motion_feats.npz`), fixes **one horizon per fire** from the median estimate over its
pre-ignition frames (static camera → constant horizon; pre-ignition-only so the plume can't drag its
own horizon), and emits three features per frame — `anchored_change`, `floating_change`,
`anchored_ratio` — then reports onset-vs-pre-ignition AUC **per fire** (computed within each fire,
averaged with a bootstrap-over-fires 90% CI — the leak-aware read) alongside a pooled AUC for
reference.

**First 24-fire read (2026-07, PRE-rigor-fix — per-frame horizon, pooled AUC, texture included):**
the anchoring hypothesis holds, modestly and — crucially — *complementarily*. Pooled AUC:
`single_frame_conf` 0.578 baseline; `anchored_change` 0.600; `anchored_ratio` 0.604 (best single
motion feature); `floating_change` 0.530 (cloud-drift control, near chance, as predicted);
**`conf + anchored` fusion 0.631 — +5.3 pts over conf alone.** The 2-fire teaser (~0.99) was
easy-fire noise. The load-bearing finding is the fusion lift: anchored motion carries signal the
appearance detector does not, justifying carrying it into Phase C as a temporal input channel. It is
**not** a strong standalone single-frame feature (~0.60 on a genuinely fuzzy near-onset boundary
task, from an unlearned hand-rule).

**⟵ SUPERSEDED by the day-only rerun (2026-07-28).** After excluding the one nocturnal fire (scope;
see [data-quality-flags.md](data-quality-flags.md)) and making the paired stats reproducible from the
probe artifact, the leak-aware read is: **17 day-only fires**, `anchored_change` per-fire AUC
**0.708** [0.606, 0.807] (pooled 0.640) vs conf **0.570**, `floating_change` control 0.566 (null),
`anchored_ratio` 0.708. Paired: anchored beats conf in **11/17 fires**, mean lift **+0.14**, anchored
AUC ≥0.7 in 11, ≥0.8 in 9 — but the win-count sign-test is **p = 0.17, not significant** at n=17. The
old "17/24, p≈0.03" mixed in the held-out eval fires and is not reproducible; the defensible claim is
that anchored motion **separates onset above chance** (CI excludes 0.5) and is **complementary** to the
detector, rescuing the hard imaging cases (`syp-w` 0.23→0.97, faint `so-w` 0.07→0.79, oversaturated
`Dehesa` 0.12→0.80). The two historical blocks below are kept for the record.

**Rigor-fix rerun (2026-07, per-fire horizon + per-fire AUC + texture removed) — anchoring
CONFIRMED and load-bearing.** Per-fire AUC (pooled in parens): `single_frame_conf` 0.568 baseline;
**`anchored_change` 0.715** [0.640, 0.787]; **`anchored_ratio` 0.711**; `floating_change` 0.555
(cloud control, CI straddles 0.5 — null, as predicted); naive `conf+anchored` fusion 0.675. Paired:
anchored beats conf in **17/24 fires** (one-sided sign-test p≈0.03), mean lift **+0.147**, anchored
AUC ≥0.7 in 14 fires, ≥0.8 in 12. *(Superseded — mixed eval fires into the paired count; see above.)*

- **The per-fire *measurement* was the decisive fix, not the horizon change.** Pooled anchored barely
  moved (0.600→0.592); the jump to 0.715 is from scoring AUC *within* each fire. Anchored-change
  magnitude varies by scene, so pooling lets between-fire scale swamp the within-fire onset signal —
  and per-fire is also the operationally correct framing (you watch one camera over time, never
  compare absolute motion across sites). Pooling was hiding the signal.
- **Fusion flipped to hurting** (0.675 < 0.715): naive equal-weight rank-sum dilutes the now-stronger
  motion feature with the weak conf. Not evidence of redundancy — the rescue cases (`syp-w` conf 0.23
  →anchored 0.97; `Bahrman` 0.63→0.96) show orthogonal signal — but the naive sum is the wrong
  combiner. **The learned Phase C temporal head is the proper combiner;** stop citing the naive fusion
  as the combined-value test.
- **Failure cases are diagnostic, and physical:** `mg-n-iqeye` (anchored 0.33) is the flagged
  dirty-lens sequence (false motion everywhere); `wc-s-mobo-c` (anchored 0.00, conf 0.64) is a single
  catastrophic horizon mis-placement; `HighwayFire` (0.55) is a plausibly wind-driven diffuse plume
  that won't anchor. Two of three are a known confound and one fixable horizon miss — not the
  mechanism failing.

**Train-only read (2026-07, 6 recent CA eval fires held out via `--exclude-eval` for Phase C):**
the signal survives the holdout almost unchanged — per-fire `anchored_change` **0.706** [0.609,
0.797], `anchored_ratio` 0.705, vs conf 0.569 and the `floating_change` null control 0.556. So the
18 training fires carry the anchoring signal; removing the eval fires cost ~0.01 AUC. This is the
number Phase C inherits.

**Smoothed-skyline horizon — TRIED and REVERTED (negative result, 2026-07).** The proposed fix for
`wc-s` — a per-column skyline (steepest per-column brightness gradient, median-smoothed across
columns) with a column-warp to flatten the terrain contour — made things **worse**: apples-to-apples
on 24 fires it dropped `anchored_change` 0.715 → 0.616, and `wc-s` stayed broken (0.03). Per-column
gradient estimates are noisy (trees, textured terrain, cloud edges pick wrong rows) and warping by
those noisy shifts scrambles the vertical structure anchoring depends on; the flat median row averages
that noise out and is more robust. Reverted to the flat per-fire horizon. `wc-s` failing under *both*
means its problem was never flat-vs-skyline.

**Luminosity sky/ground mask — TRIED and REVERTED (negative result, 2026-07).** The author's idea
(from photo editing) that absolute luminosity splits sky from ground: per-fire Otsu on the averaged
pre-ignition frame → ground mask, then flood change upward from the ground through connected change
pixels (warp-free A/B vs the flat horizon). It underperformed — per-fire `anchored_lum` **0.537** vs
flat-horizon `anchored_change` **0.706**, with the `floating_lum` control (0.582) *beating* it (the
fingerprint of a mislabeling mask). All 18 masks were non-degenerate (ground fraction 0.34–0.90), so
it is a real feature-quality result: on FIgLib's variable imagery (haze, backlight, dark forest,
bright cloud) *absolute* brightness does not track the horizon, while the flat estimator keys on
*relative* brightness change and is robust. Three horizon variants tried; the simplest wins. See
[motion-findings.md](motion-findings.md).

**Learned combiners — both tested (2026-07-28), neither a validated win; see
[motion-findings.md](motion-findings.md).** (1) *Per-frame logistic fusion* of base conf + the three
motion features ([`figlib_fusion.py`](../src/models/figlib_fusion.py)): leak-free (zero-shot conf +
training-free motion), but appearance-dominated (conf weight +0.77 vs motion ≤0.18), training LOO AUC
0.623 < conf-alone 0.689, eval frame-AUC inconsistent; only an n=6-noisy TTD lift (5/6 vs 4/6). (2)
*Causal LSTM* ([`figlib_lstm.py`](../src/models/figlib_lstm.py)): looked decisive (LOO 0.858, eval fires
at 1.000, TTD 0.97 min) but the ablation controls exposed a **temporal-POSITION leak** — `ABLATE=zero`
(all features zeroed) still scores LOO 0.994 / eval 1.000 (FIgLib's monotonic onset-centered label lets
a sequence model "detect" by counting frames), and `ABLATE=shuffle` (time order permuted) collapses it to
LOO 0.612 ≈ the per-frame LR. **Gating requirement now recorded:** FIgLib's fixed onset-centered
sequences cannot validly evaluate a temporal model for TTD/onset-AUC — a real temporal test needs
**continuous-feed or onset-position-randomized data** (HPWREN archive / ALERTCalifornia; see the
[net-new CA data](#net-new-california-data--evaluated-2026-07-28-next-data-steps) and
[public data sources](#public-data-sources-for-the-temporal--time-to-detection-thread) sections). The
horizon estimator is settled (flat brightness-gradient row); no further tuning warranted.

## Also noted (lower priority)

- **In-distribution tiled detector on FIgLib** (SmokeyNet setup: 224-px tiles, train on FIgLib's own
  bounding boxes). Would raise the FIgLib base AUC above 0.658 and let the *learned* temporal
  model — not just the persistence rule — be tested. The persistence sign-flip predicts it wins.

- **Sharpen the confuser corpus** by cropping to the alarm region before embedding, removing the
  terrain/skyline conflation so clusters become purer weather classes (fog / cumulus / stratus /
  glare) rather than partly per-tower. See [confuser-corpus.md](confuser-corpus.md).

- **Recall-first metric reporting.** Bake a recall-first operating point into `evaluate.py`
  (highest recall subject to a false-alarm budget) and report alarms-per-camera-per-day at a
  fixed high recall, with F1 demoted to context.

## Public data sources for the temporal / time-to-detection thread

Time-to-detection and a stronger temporal model both need onset *sequences* (and, ideally,
continuous camera feeds), which pyro-sdis lacks. The public options, most useful first:

- **HPWREN camera archive** (HPWREN, n.d.) — the richest public source: raw camera images (one
  per minute, fixed cameras; every 10 s, PTZ) *and* compiled MP4 videos, downloadable back to
  ~2000 at `http://c1.hpwren.ucsd.edu/archive/`. FIgLib is curated from this, so it is the path
  to *more* onset sequences and to continuous feeds for time-to-detection.
- **PYRONEAR-2025** (Lostanlen et al., 2024) — images *and videos*, ~640 wildfires from France,
  Spain, Chile and the US; the same lineage as pyro-sdis.
- **ALERTCalifornia / ALERTWildfire** — live feeds from 1,600+ cameras (`alertwest.live`) with
  short in-browser timelapse replay; live/near-real-time rather than a bulk download (the bulk
  archive is HPWREN's).
- **Classic video clip datasets** for smaller temporal experiments: Bilkent VisiFire (40 clips),
  FIRESENSE (49 videos, on Zenodo), MIVIA fire+smoke (180 videos).

Note: Pano AI and most operational vendors do **not** release public feeds or imagery — their
data is proprietary/customer-gated, and they have no peer-reviewed publications (patents and
product pages only). So the public temporal data all traces back to HPWREN and PyroNear.

## Pre-scoping #3 — HPWREN → time-to-detection

The goal is the field's headline metric, which this project lacks: **time-to-detection (TTD)** —
minutes from ignition to the first alarm — plus a *learned* temporal model trained on enough onset
fires to beat the parameter-free persistence rule. Scoped in three phases, cheapest first.

**Why TTD is now cheap to start.** FIgLib is built as onset sequences: **81 frames per fire, ~60 s
apart, spanning −40 to +40 min around ignition**, with the time offset encoded in each frame's
filename (e.g. `..._-05` = 5 min before, `..._+03` = 3 min after). So ground-truth
minutes-from-ignition is *already in the data* — no new labeling. We hold 18 fire sequences
locally and the tiled native-resolution detector ([`figlib_tiled.py`](../src/models/figlib_tiled.py))
already emits a max-confidence score per frame. TTD is a thin new eval on top of that.

**TTD definition (leak-safe, censoring-aware).** Hold out whole fires. Pick an operating threshold
on held-out fires (at a target pre-ignition false-alarm rate — the operator's constraint). For each
test fire, `TTD = smallest t ≥ 0 (minutes) with confidence ≥ threshold`. Report, separately:
- **detection rate** — share of fires ever detected within +40 min (a *missed* fire is right-
  censored, not a small TTD — never average it in);
- **median / mean TTD** over *detected* fires, and **% detected within 5 min** (SmokeyNet's ~3.6
  min and 80%-within-5-min are the comparison points; Dewangan et al., 2022);
- **false-alarm rate on pre-ignition frames** (t < 0) — the trigger-happiness counterpart, so
  "early" is never bought with "cries wolf". TTD and this rate are the operator-relevant *pair*.
- A **persistence-required variant** (alarm only after k consecutive frames cross) and its TTD
  cost — directly extends the persistence sign-flip finding ([figlib-findings.md](figlib-findings.md)).

**Phase A — TTD on the 18 fires we already have (no new data, ~a day).** New `figlib_ttd.py`:
parse the time offset from filenames, reuse the tiled per-frame confidences, compute the metrics
above. Gives a first TTD number and the eval harness. Small n (18 fires), so treat as directional.

**Phase B — more onset fires (data pull).** Two public sources, both traceable to the lineage we
already use:
- **FIgLib full** — 315 fires / ~24,800 images (Dewangan et al., 2022), via the WIFIRE Commons
  Data Catalog (`wifire-data.sdsc.edu/dataset/hpwren-fire-ignition-library`); we curated our 18
  from here, so scaling up is the same path. Tightens TTD stats and supplies training fires.
- **PYRONEAR-2025** — ~50k images / 150k annotations / **640 wildfires**, images *and videos*,
  France/Spain/Chile/US (Lostanlen et al., 2024; arXiv:2402.05349), on Hugging Face under
  `pyronear`. The only public source with sequence *videos* at scale for a learned temporal model.

**Phase C — a learned temporal model that beats persistence.** With enough onset fires, train the
causal GRU / small CNN-LSTM on tiled embeddings (the setup that lost on 18 fires purely for lack of
data; [figlib-findings.md](figlib-findings.md)) and test two claims at matched recall: (1) it beats
the persistence rule on pre-ignition false alarms, and (2) it *lowers TTD* (detects earlier) — the
payoff temporal context is supposed to buy. The persistence sign-flip predicts (1); (2) is the new
question.

**Reuses / new code.** Reuse `figlib_tiled.py` (tiled inference) and the leak-safe hold-out-fires
convention. New: `figlib_ttd.py` (TTD eval), a FIgLib-full fetch helper, and a small extension to
the temporal head for the learned-model test.
