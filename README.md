# Wildfire Smoke Detection

Teaching a camera to spot a wildfire in its first minutes — and, just as importantly, measuring
honestly whether it actually works in the real world.

## Bottom line

Most published smoke detectors look great on their test sets and then struggle in the field. This
project is built around closing that gap: it judges a detector the way a fire lookout would — **how
many real fires does it catch, and how many false alarms does it raise** — instead of a lab score that
treats a missed fire and a false alarm as equally bad.

The headline results:

- **Training on local smoke works.** After some extra training on California fires it had never seen,
  the detector caught **more fires (5 of 6, up from 4 of 6)** and **cut its false alarms by more than
  half**. This is the clearest evidence in the project that a detector can be adapted to a new region.
- **Sharper images matter most for early smoke.** Small, faint plumes disappear when a photo is shrunk
  down. Keeping the full image resolution roughly **doubled** the share of fires the detector could
  catch.
- **Most false alarms are clouds** — about **74%** of them. Knowing exactly what fools the detector is
  the first step to fixing it.
- **Two popular "smart" ideas did not pan out here — and that's reported plainly.** A method that uses
  motion across video frames failed to help on this data, and a more advanced version looked like a big
  win until a careful check revealed it was quietly cheating (learning *when* in a clip a fire usually
  appears rather than *what* smoke looks like). Catching that was one of the most valuable moments in
  the project.
- **The honest gap.** Even the best setup here still raises far more false alarms than a real operator
  would tolerate. What it would take to close that gap is spelled out, not glossed over.

## Why this is hard

A wildfire caught in its first minutes can be stopped; one caught an hour later can become a disaster.
So a smoke detector's mistakes are not equal: **missing a real fire is catastrophic, while a false
alarm just costs a person a few seconds to glance at a camera** (a human always checks before any
crews are sent). Yet the most-cited research scores its models with a measure that treats those two
mistakes as equally bad — and the one comparable system with a real field report raised a false alarm
**79% of the time**. This project measures the things that actually matter to the people watching the
cameras.

## About the project author

<!-- AUTHOR NARRATIVE GOES HERE — Joel to provide; paste it and I'll format it in. -->

_Author narrative to be added._

## Find what you're looking for

This project has three layers: this README (the plain-language overview), a **[Findings
Report](reports/smoke-detection-report.md)** (the technical story, with a table of contents and a
plain-language [glossary](reports/smoke-detection-report.md#11-glossary) of every term and
abbreviation), and a **[`research/`](research/)** folder holding the detailed notes behind every result.

**Two common paths:**

- **"How does the detector actually work?"** — the machine-learning and object-detection side: how
  performance is measured, how the model was built and improved, and how evaluation was kept honest.
  Start with the Findings Report, sections
  [2 (how we measure)](reports/smoke-detection-report.md#2-how-performance-is-measured--and-why-not-f1),
  [4 (the baseline)](reports/smoke-detection-report.md#4-baseline-the-precision-collapse),
  [5 (image resolution)](reports/smoke-detection-report.md#5-resolution-one-lever-both-axes), and
  [6 (learning from mistakes)](reports/smoke-detection-report.md#6-into-the-negatives-hard-negative-mining-and-the-confuser-corpus).
  The code lives in [`src/`](src/).
- **"How well did it perform, and what was learned?"** — the results and the story behind them: the
  region-adaptation win, the motion experiments, the honest negatives, and how far this is from being
  deployable. See Findings Report sections
  [8 (speed to detection)](reports/smoke-detection-report.md#8-time-to-detection),
  [9 (motion)](reports/smoke-detection-report.md#9-motion-horizon-anchored-change), and
  [10 (where the gap stands)](reports/smoke-detection-report.md#10-where-the-gap-stands-and-whats-next),
  plus the plain-English **[research narrative](reports/research-narrative.md)** of how the work
  unfolded, decision by decision.

**By who you are:**

| If you are a… | Start here |
|---|---|
| **Recruiter / hiring manager** | This page, then the Findings Report [section 1](reports/smoke-detection-report.md#1-the-problem-and-the-frame) and [section 10](reports/smoke-detection-report.md#10-where-the-gap-stands-and-whats-next) — the framing and the outcomes. |
| **ML / data scientist** | The "how does it work" path above, the [research narrative](reports/research-narrative.md), and the code in [`src/`](src/). |
| **Fire-detection researcher** | The [Findings Report](reports/smoke-detection-report.md) in full and the [field survey](research/state-of-smoke-detection.md) of datasets, methods, and how the field measures success. |
| **Agency / wildfire manager** | This page, plus [section 10](reports/smoke-detection-report.md#10-where-the-gap-stands-and-whats-next) — what works today, what doesn't yet, and why. |

## What's in this project

The detector is built and tested on [pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis)
(33,636 images from French fire-detection towers; Pyronear, 2025), with additional tests on California
onset fires from the HPWREN camera archive. Two ideas run through all of it:

- **No cheating in the test.** The detector is only ever tested on camera sites it never trained on, so
  its scores reflect genuinely new terrain — not memorized backgrounds.
- **Field-realistic scoring.** Results are reported as detection rate and false alarms per camera per
  day (the numbers an operator cares about), not a single lab score.

Folder map:

- **[`reports/`](reports/)** — the reader-facing write-ups: the
  [Findings Report](reports/smoke-detection-report.md) and the
  [research narrative](reports/research-narrative.md).
- **[`research/`](research/)** — the detailed notes behind every result: the
  [field survey](research/state-of-smoke-detection.md), the
  [measurement rationale](research/metrics.md), and per-topic findings for
  [baseline](research/baseline-findings.md), [resolution](research/resolution-findings.md),
  [learning from mistakes](research/hard-negative-findings.md),
  [what fools the detector](research/confuser-corpus.md), [motion over time](research/temporal-findings.md),
  [the California test](research/figlib-findings.md), [speed to detection](research/ttd-findings.md),
  [the motion cue](research/motion-findings.md), and [region adaptation](research/phase-c-findings.md).
  Working notes live alongside them (the [backlog](research/backlog.md),
  [data-quality flags](research/data-quality-flags.md)).
- **[`src/`](src/)** — the pipeline: dataset preparation and honest splits
  ([`src/data/`](src/data/)), plus training, operator-focused evaluation, and the experiments
  ([`src/models/`](src/models/)).
- **[`results/`](results/)** — evaluation outputs and the list of frames that fool the detector.
- `data/` — datasets (not stored in git; rebuilt with `src/data/export_yolo.py`). A measured summary is
  in [`data/data-profile.md`](data/data-profile.md).

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install datasets ultralytics pillow pyyaml pandas
python src/data/export_yolo.py                                   # download + build honest splits
python src/models/train.py --split grouped --epochs 40           # train the baseline detector
python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped
```

## A note on maturity

Many results here are *proof-scale* — run on a fraction of the data to test an idea quickly, so read
the **direction** of the numbers, not the last decimal. The full-resolution training result is a
complete run. False-alarm-per-day figures are informed estimates, not measured field rates. Where a
result is preliminary or based on a small sample, it says so.

## Data & credit

[pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis) (Apache-2.0) by
[Pyronear](https://pyronear.org/) (Pyronear, 2025). California fire imagery courtesy of HPWREN
(Dewangan et al., 2022; credit `http://hpwren.ucsd.edu/`). Image data and model weights are not stored
in git.

## Works Cited

- Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., &
  Nguyen, M. H. (2022). FIgLib & SmokeyNet: Dataset and deep learning model for real-time
  wildland fire smoke detection. *Remote Sensing, 14*(4), 1007. https://doi.org/10.3390/rs14041007
- Govil, K., Welch, M. L., Ball, J. T., & Pennypacker, C. R. (2020). Preliminary results from a
  wildfire detection system using deep learning on remote camera images. *Remote Sensing,
  12*(1), 166. https://doi.org/10.3390/rs12010166
- Pano AI. (2024). *Pano Rapid Detect: solution overview* [Product page]. https://www.pano.ai/solution
- Pyronear. (2025). *pyro-sdis* [Dataset]. Hugging Face.
  https://huggingface.co/datasets/pyronear/pyro-sdis

<sub>Detailed per-topic sources — operational networks (Pano, ALERTCalifornia), NOAA GOES/NGFS, and
meteorological verification — are cited inline in [`research/metrics.md`](research/metrics.md) and
[`research/`](research/).</sub>
