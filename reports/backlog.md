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
  - **Phase C:** train an in-distribution detector on FIgLib boxes → the AUC 0.658 floor rises and
    TTD should drop toward the SmokeyNet range; then test a *learned* temporal head. Best run on a
    cloud GPU ([gcp-plan.md](gcp-plan.md)) rather than the ~92-min/epoch local box.
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
