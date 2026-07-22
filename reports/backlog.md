# Backlog — experiments to try

A running list of ideas worth pursuing, with enough context to pick any one up cold. Ordered
roughly by leverage. Completed threads have their own findings reports (see the [README](../README.md)).

## In progress

- **Full-scale native-resolution (1280) training.** The proof-scale 1280 run was undertrained
  (val metrics still climbing at epoch 15). Retraining on full data at imgsz 1280 with early
  stopping to fairly test whether high-res *training* — not just high-res inference — improves
  the reachable recall ceiling and the false-alarm/recall trade. See
  [resolution-findings.md](resolution-findings.md).

## Queued

- **D-Fire zero-shot cross-dataset evaluation.** Train on pyro-sdis, evaluate *cold* on
  [D-Fire](https://github.com/gaiasd/DFireDataset) (~21k images, Brazil, YOLO-format smoke+fire
  boxes) with no fine-tuning. Measures cross-dataset domain shift — the most honest
  generalization number in the project, extending the leak-safe theme from across-towers to
  across-datasets. Reuse `evaluate.py` almost verbatim; filter to the smoke class and note the
  label-definition mismatch (D-Fire leans closer-range and mixes fire). The FIgLib work already
  gave an *accidental* domain-shift datapoint (pyro-sdis → FIgLib collapsed to AUC 0.45 before
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

- **In-domain tiled detector on FIgLib** (SmokeyNet setup: 224-px tiles, train on FIgLib's own
  bounding boxes). Would raise the FIgLib base AUC above 0.658 and let the *learned* temporal
  model — not just the persistence rule — be tested. The persistence sign-flip predicts it wins.

- **Sharpen the confuser corpus** by cropping to the alarm region before embedding, removing the
  terrain/skyline conflation so clusters become purer weather classes (fog / cumulus / stratus /
  glare) rather than partly per-tower. See [confuser-corpus.md](confuser-corpus.md).

- **Recall-first metric reporting.** Bake a recall-first operating point into `evaluate.py`
  (highest recall subject to a false-alarm budget) and report alarms-per-camera-per-day at a
  fixed high recall, with F1 demoted to context.
