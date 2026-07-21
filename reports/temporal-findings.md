# Temporal model — findings

**Status: proof scale, and a negative result — reported as such.** The hypothesis, taken
straight from the literature (SmokeyNet: +26 precision points from frame-to-frame context),
was that a temporal model would suppress the single-frame detector's false alarms. On
pyro-sdis, over a frozen-detector feature stream, **it does not.** The interesting part is
*why*, and the data answers it cleanly.

## The hypothesis and the mechanism it rests on

Real smoke *persists and grows* across frames; the things a single frame confuses for smoke —
fog banks, glare, terrain haze — were assumed to *flicker*. If that were true, requiring
temporal persistence would drop the false alarms while keeping the real detections. This is
the mechanism behind every temporal smoke result in the literature.

That mechanism has a testable precondition: **the frames the detector false-alarms on must be
surrounded by low-confidence neighbors.** So we tested it directly before building anything.

## The precondition fails on this data

pyro-sdis arrives in short bursts (~29 s between frames, median 19 frames per burst). On the
held-out test towers, of the 202 frames the single-frame detector false-alarms on:

| | count | share |
|---|---|---|
| **persistent** (previous frame also high-confidence) | 153 | **76%** |
| flicker (previous frame low — temporal *could* suppress) | 29 | 14% |
| burst start (no previous frame) | 20 | 10% |

For contrast, true smoke is persistent 91% of the time. **The persistence signal barely
separates the two (91% vs 76%)** — because the false alarms are not flicker. They are stable
structures: a cloud bank sitting behind a ridge, a fixed sun-glare, a hazy valley. They look
the same frame after frame, exactly like real smoke does. Requiring persistence keeps them.

![Temporal context does not beat a single frame on pyro-sdis: at matched recall the three
methods overlap and persistence is worse at tight operating points; the false alarms are 76%
persistent, not flicker.](figures/temporal_result.png)

## The model confirms it, at matched recall

Comparing a temporal model to a single-frame one by each one's own threshold is misleading —
the scores live on different scales. So we hold **recall fixed** and ask the operator's real
question: *at the recall you require, what fraction of clean frames still false-alarm?*

Three score functions, identical held-out towers ([`compare_temporal.py`](../src/models/compare_temporal.py)):

- **single-frame** — the detector's per-frame max confidence (the baseline)
- **persistence** — rolling min of confidence over the last 8 frames (the interpretable
  temporal rule: "only alarm if the evidence persisted")
- **temporal-gru** — a learned GRU head over the confidence sequence

False-alarm rate on clean frames (lower is better):

| recall | single-frame | persistence | temporal-gru |
|---|---|---|---|
| 0.80 | **57.8%** | 60.9% | 60.3% |
| 0.70 | 43.2% | **42.4%** | 44.7% |
| 0.60 | 32.4% | **32.0%** | 32.8% |
| 0.50 | **20.6%** | 27.2% | 21.4% |
| 0.40 | **12.3%** | 18.7% | 12.1% |
| 0.30 | **6.4%** | 10.2% | 6.7% |

The learned GRU tracks the single-frame detector almost exactly — it learned, correctly, that
the most useful thing in the window is the *current* frame's confidence. The persistence rule
is **worse** at the operating points that matter (tighter recall), because it suppresses
borderline true smoke while leaving the persistent confusers untouched. No temporal method
beats single-frame here.

## Why this is not a contradiction of the literature

The literature's temporal gains come from **FIgLib**, which is built specifically as ignition
sequences: 40 frames spanning from before a fire starts to well after, so the model watches a
plume emerge from an empty sky over many minutes. pyro-sdis is a *detection* corpus — its
bursts are short and mostly capture already-developed conditions (39% of test frames are in
bursts that are smoky end-to-end). The onset dynamics that make temporal context powerful are
largely absent, and where a scene *is* ambiguous, it is ambiguous in a temporally stable way.

The temporal advantage is **real but dataset-dependent**: it needs sequences that capture
onset against a clean background, and confusers that are transient. pyro-sdis provides neither
at the scale FIgLib does.

## Honest caveats

- **This is a frozen-detector head, not end-to-end.** We put a temporal head on the baseline
  detector's per-frame evidence rather than training a CNN+LSTM jointly on pixels. A fully
  end-to-end model *might* extract motion cues the per-frame confidence discards. But the
  precondition analysis above is model-independent: it measures the confusers directly, and
  they are persistent. The ceiling is set by the data, not only by this architecture.
- Proof scale throughout (underfit detector as the feature extractor). Direction is
  trustworthy; absolute numbers are not final.

## What this changes

The earlier reports named the temporal model as the expected fix for the false-alarm floor.
It is not, on this dataset. The result that *did* move the false-alarm rate was
[hard-negative mining](hard-negative-findings.md) (42% → 20%) — teaching the detector what a
false alarm looks like, which attacks the persistent confusers head-on rather than hoping they
flicker. On pyro-sdis, **the leverage is in the negatives, not the time axis.** A temporal
model earns its keep on onset-capturing data (FIgLib-style); porting one here would be
cargo-culting the architecture past the conditions that justify it.

## Reproduce

    python src/models/extract_features.py --weights runs/grouped_proof/weights/best.pt
    python src/models/compare_temporal.py          # matched-recall table above
    python src/models/temporal.py --epochs 40       # standalone GRU eval

Comparison table: `results/temporal_comparison.json`. GRU sweep: `results/eval_temporal_test.json`.
