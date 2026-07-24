# Wildfire Smoke Detection

A data-science portfolio project on early wildfire smoke detection from fixed-camera
imagery. The theme throughout is **field-realistic evaluation**: measuring what a detector
would actually do in the field, not what it scores on a flattering benchmark.

## The one thing worth knowing

The only comparable smoke detector with a published *field* number produced a **79%
false-positive rate** in real deployment (Govil et al., 2020). The most-cited academic model,
SmokeyNet, looks far stronger on paper — but it was never field-deployed, and its headline
score is **F1, a metric that weights a missed fire exactly like a false alarm** (Dewangan et
al., 2022). That is the wrong trade for wildfire, where a missed fire is catastrophic and a
false alarm costs a watchstander a glance. The benchmark-vs-field gap — and the misleading
yardstick behind it — is the problem this project is built around: it measures what a detector
would actually do in the field (**detection rate and false-alarm burden**), not F1. See
[`reports/metrics.md`](reports/metrics.md) and
[`reports/state-of-smoke-detection.md`](reports/state-of-smoke-detection.md).

## What's here

**A research survey** ([`reports/state-of-smoke-detection.md`](reports/state-of-smoke-detection.md),
backed by [`research/`](research/)) covering public datasets, modeling approaches, how
performance is really measured, and when these tools do and don't work.

**A working pipeline on [pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis)**
(33,636 images, French detection towers; Pyronear, 2025) that takes evaluation integrity
seriously:

- **Leak-safe splits** ([`src/data/splits.py`](src/data/splits.py)). The 40 camera IDs in the
  dataset are really 8 physical towers — each ID is a `tower-bearing` view of the same mast — so
  we hold out whole *sites* and a model is only ever tested on terrain it never trained on. Both a naive (leaky)
  and a leak-safe (site-disjoint) split are produced, so the inflation from leakage can be
  *measured*, not just asserted.
- **Recall-first, field-standard metrics** ([`src/models/evaluate.py`](src/models/evaluate.py)).
  A missed fire is catastrophic; a false alarm costs a watchstander a glance, since a human
  reviews every candidate detection before any suppression resources are dispatched. So
  evaluation is **not F1** — which weights the two errors
  equally, wrong for this domain — but what the field actually uses: **probability of detection
  (POD)**, the false-alarm burden as **false-positives-per-camera-per-day** (Pano's operational
  target is < 1), and **relative economic value across cost-loss ratios** (meteorology's score
  for asymmetric costs). mAP and F1 are computed but demoted to context.

## Findings so far (mostly proof scale — directional; the native-resolution training row is a converged full-scale run)

The right question isn't the F1 score — it's **how high can detection (POD) go, and at what
false-alarm burden.** On the held-out towers:

| configuration | max detection rate (POD) | false-alarm burden* |
|---|---|---|
| single-frame baseline (infer @640) | 0.68 | ~208 FP/camera/day |
| + native-resolution inference (@1280) | **0.86** | ~388 FP/camera/day |
| + native-resolution **training** (@1280, full-scale) | 0.83 | **~173 FP/camera/day** |
| + hard-negative mining (@640) | 0.68 | ~half the burden |

<sub>*at an assumed 1% base rate and 500 frames/camera/day — an extrapolation, not a measured
rate. Pano AI's operational target is < 1 FP/camera/day (Pano AI, 2024); the gap is the work
that remains.</sub>

Two levers, two axes. **Resolution raises the detection *ceiling*** — the 640 model structurally
caps at POD 0.68 (it never sees the small plumes), while native-resolution inference reaches 0.86,
and native-resolution *training* to convergence holds 0.83 at **roughly half the false-alarm
burden** (~173 vs ~388 FP/camera/day) — the best config so far
([resolution](reports/resolution-findings.md)). **[Hard-negative mining](reports/hard-negative-findings.md)
and the [confuser corpus](reports/confuser-corpus.md) lower the false-alarm *burden*** (the
baseline false-alarms on ~60% of clean frames — 74% of them clouds — and mining halves that).
The two levers have **not yet been combined**, and that is the open work: at the cost-loss ratios
where misses dominate, even the best config still adds only marginal value. Native-resolution
training *plus* confuser-targeted mining — the deployable recipe — is what should close the gap.

The next step was a **temporal model** — the literature's headline fix (SmokeyNet's +26
precision points from frame-to-frame context; Dewangan et al., 2022). We built it and it **did
not transfer to this dataset**, and [the report explains why](reports/temporal-findings.md): 76% of the false
alarms are *persistent* structures (fixed cloud banks, glare, ridge haze), not the flicker a
persistence model suppresses, and pyro-sdis's short bursts lack the ignition-onset dynamics
that power temporal gains on FIgLib. At matched recall, no temporal method beats the
single-frame detector here. That is reported as a **negative result**, because it is one —
on pyro-sdis the leverage is in the negatives, not the time axis.

As a check on that claim, we ran the same pipeline on [**FIgLib**](reports/figlib-findings.md),
the onset-sequence dataset the temporal literature used (Dewangan et al., 2022). The first run looked like a dead end —
the detector scored AUC 0.45 (worse than random) — until a question about *resolution* found the
real cause: we were downscaling FIgLib's native 3072×2048 frames to 640 px, pooling the tiny
early plumes away. Running the same detector on native-resolution **tiles** lifted AUC to 0.658,
and the positive control then landed: requiring temporal persistence **cuts** false alarms
12–19 pts on FIgLib, where the very same rule **raised** them on pyro-sdis. Same rule, opposite
sign, split by whether the data contains ignition onset — the mechanism, confirmed both ways.

So we went *into* the negatives and built a [**typed confuser corpus**](reports/confuser-corpus.md):
clustering the 2,305 frames the detector false-alarms on into named failure modes. The result
is one clean number — **74% of the false alarms are clouds** (cumulus, backlit stratus, broken
overcast) — a measured version of the documented single-frame failure mode, where the detector
fires on nearly every cloud. The report found no such public corpus exists, so
`results/confuser_corpus.csv` is a small original contribution.

## Layout

- [`reports/`](reports/) — the state-of-the-field report, a [metrics rationale](reports/metrics.md)
  (why recall-first, not F1), and per-stage findings (baseline, hard-negative, temporal, confuser
  corpus, FIgLib, [resolution & recall-first](reports/resolution-findings.md),
  [time-to-detection](reports/ttd-findings.md)),
  and a [**research narrative**](reports/research-narrative.md) tracing how the project actually
  unfolded (including the human resolution insight that rescued the FIgLib control). Open threads
  are tracked in the [backlog](reports/backlog.md), and background primers in
  [background-topics](reports/background-topics.md).
- [`research/`](research/) — detailed source material behind the report
- [`src/data/`](src/data/) — dataset export, leak-safe splits, hard-negative mining, confuser corpus
- [`src/models/`](src/models/) — training, operator-framed evaluation, temporal model + comparison
- [`results/`](results/) — eval sweeps + mined hard-negative list
- `data/` — datasets (gitignored; regenerate with `src/data/export_yolo.py`). The measured
  data profile is in [`data/data-profile.md`](data/data-profile.md).

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install datasets ultralytics pillow pyyaml pandas
python src/data/export_yolo.py                                   # download + build splits
python src/models/train.py --split grouped --epochs 40           # baseline
python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped
```

## Data & credit

[pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis) (Apache-2.0) by
[Pyronear](https://pyronear.org/) (Pyronear, 2025). FIgLib imagery courtesy of HPWREN
(Dewangan et al., 2022; credit `http://hpwren.ucsd.edu/`). Image data and model weights are
gitignored.

*Mostly proof-scale results: read the direction of the numbers, not the absolute values. The
native-resolution training row is a converged full-scale run (40 epochs, full data); the rest are
one flag change away (drop `--fraction`, raise `--epochs`).*

## Works Cited

- Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., &
  Nguyen, M. H. (2022). FIgLib & SmokeyNet: Dataset and deep learning model for real-time
  wildland fire smoke detection. *Remote Sensing, 14*(4), 1007.
  https://doi.org/10.3390/rs14041007
- Govil, K., Welch, M. L., Ball, J. T., & Pennypacker, C. R. (2020). Preliminary results from a
  wildfire detection system using deep learning on remote camera images. *Remote Sensing,
  12*(1), 166. https://doi.org/10.3390/rs12010166
- Pano AI. (2024). *Pano Rapid Detect: solution overview* [Product page]. https://www.pano.ai/solution
  (dual-camera 360° stations, patented triangulation for GPS geolocation; the < 1 false-positive-
  per-camera-per-day figure is Pano's own operational claim, not a peer-reviewed result).
- Pyronear. (2025). *pyro-sdis* [Dataset]. Hugging Face.
  https://huggingface.co/datasets/pyronear/pyro-sdis

<sub>Detailed per-topic sources — operational networks (Pano, ALERTCalifornia), NOAA GOES/NGFS,
and meteorological verification (cost-loss / relative economic value) — are cited inline in
[`reports/metrics.md`](reports/metrics.md) and [`research/`](research/).</sub>
