# Wildfire Smoke Detection

A data-science portfolio project on early wildfire smoke detection from fixed-camera
imagery. The theme throughout is **honest evaluation**: measuring what a detector would
actually do in the field, not what it scores on a flattering benchmark.

## The one thing worth knowing

The most-cited academic smoke detector (SmokeyNet) reports ~83% F1 on its benchmark — but
was never field-deployed. The only comparable system with a published field number produced
a **79% false-positive rate** in real deployment. That gap between benchmark and field is the
problem this project is built around. See
[`reports/state-of-smoke-detection.md`](reports/state-of-smoke-detection.md).

## What's here

**A research survey** ([`reports/state-of-smoke-detection.md`](reports/state-of-smoke-detection.md),
backed by [`research/`](research/)) covering public datasets, modeling approaches, how
performance is really measured, and when these tools do and don't work.

**A working pipeline on [pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis)**
(33,636 images, French detection towers) that takes evaluation integrity seriously:

- **Leak-safe splits** ([`src/data/splits.py`](src/data/splits.py)). The 40 "cameras" are
  really 8 physical towers (each camera string is `tower-bearing`), so we hold out whole
  *sites* — a model is only ever tested on terrain it never trained on. Both a naive (leaky)
  and an honest (site-disjoint) split are produced, so the inflation from leakage can be
  *measured*, not just asserted.
- **Operator-framed metrics** ([`src/models/evaluate.py`](src/models/evaluate.py)). Instead of
  mAP (structurally wrong for a boundary-less object), evaluation asks "does this frame raise
  an alarm?" and reports the false-alarm rate on clean frames plus **base-rate-corrected
  precision** — what precision becomes once smoke is as rare as it is in the field.

## Findings so far (proof scale — directional, not final)

| | baseline | + hard-negative mining |
|---|---|---|
| False alarms on clean held-out frames | 42% | **20%** |
| Precision @ 1% deployment base rate | 1.6% | **3.1%** |

The single-frame baseline false-alarms on ~60% of clean training frames — the documented
"smoke is not an object" precision collapse. [Mining those hard negatives](reports/hard-negative-findings.md)
(clouds, fog, glare) and retraining halved the false-alarm rate.

The next step was a **temporal model** — the literature's headline fix (SmokeyNet's +26
precision points from frame-to-frame context). We built it and it **did not transfer to this
dataset**, and [the report explains why](reports/temporal-findings.md): 76% of the false
alarms are *persistent* structures (fixed cloud banks, glare, ridge haze), not the flicker a
persistence model suppresses, and pyro-sdis's short bursts lack the ignition-onset dynamics
that power temporal gains on FIgLib. At matched recall, no temporal method beats the
single-frame detector here. That is reported as a **negative result**, because it is one —
on pyro-sdis the leverage is in the negatives, not the time axis.

As a check on that claim, we ran the same pipeline on [**FIgLib**](reports/figlib-findings.md),
the onset-sequence dataset the temporal literature used. The first run looked like a dead end —
the detector scored AUC 0.45 (worse than random) — until a question about *resolution* found the
real cause: we were downscaling FIgLib's native 3072×2048 frames to 640 px, pooling the tiny
early plumes away. Running the same detector on native-resolution **tiles** lifted AUC to 0.658,
and the positive control then landed: requiring temporal persistence **cuts** false alarms
12–19 pts on FIgLib, where the very same rule **raised** them on pyro-sdis. Same rule, opposite
sign, split by whether the data contains ignition onset — the mechanism, confirmed both ways.

So we went *into* the negatives and built a [**typed confuser corpus**](reports/confuser-corpus.md):
clustering the 2,305 frames the detector false-alarms on into named failure modes. The result
is one clean number — **74% of the false alarms are clouds** (cumulus, backlit stratus, broken
overcast) — the literature's "it screams smoke at every cloud," measured. The report found no
such public corpus exists, so `results/confuser_corpus.csv` is a small original contribution.

## Layout

- [`reports/`](reports/) — the state-of-the-field report + per-stage findings (baseline,
  hard-negative, temporal, confuser corpus, FIgLib, [resolution & recall-first](reports/resolution-findings.md)),
  and a [**research narrative**](reports/research-narrative.md) tracing how the project actually
  unfolded (including the human resolution insight that rescued the FIgLib control). Open threads
  are tracked in the [backlog](reports/backlog.md).
- [`research/`](research/) — detailed source material behind the report
- [`src/data/`](src/data/) — dataset export, leak-safe splits, hard-negative mining, confuser corpus
- [`src/models/`](src/models/) — training, operator-framed evaluation, temporal model + comparison
- [`results/`](results/) — eval sweeps + mined hard-negative list
- `data/` — datasets (gitignored; regenerate with `src/data/export_yolo.py`)

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install datasets ultralytics pillow pyyaml pandas
python src/data/export_yolo.py                                   # download + build splits
python src/models/train.py --split grouped --epochs 40           # baseline
python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped
```

## Data & credit

[pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis) (Apache-2.0) by
[Pyronear](https://pyronear.org/). Image data and model weights are gitignored.

*Proof-scale results throughout: read the direction of the numbers, not the absolute values.
Full-scale runs are one flag change (drop `--fraction`, raise `--epochs`).*
