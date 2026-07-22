# pyro-sdis — what the data actually looks like

Measured from the corpus itself (`src/data/export_yolo.py`, `src/data/splits.py`), not
from the dataset card. Images and labels are gitignored; regenerate with:

    python src/data/export_yolo.py

## Corpus

| | |
|---|---|
| Images | 33,636 |
| With smoke | 28,137 (83.7%) |
| True negatives | 5,499 (16.3%) |
| Boxes | 32,109 |
| Resolution | 1280×720 (uniform; **the dataset card does not state this**) |
| Date range | 2024-01-01 → 2024-09-22 |
| License | Apache-2.0 |

## Site structure — the constraint that shapes evaluation

The `camera` field has **40 distinct values, but only 8 physical sites.** Camera strings
encode *tower + bearing*: `brison-110` and `brison-200` are two directions from one mast,
sharing terrain, hardware, and sky.

**Grouping by `camera` would leak.** We strip the trailing bearing and group by site.

| Site | Images | Cameras |
|---|---|---|
| brison | 12,803 | 7 |
| courmettes | 7,327 | 4 |
| marguerite | 4,634 | 4 |
| cabanelle | 2,701 | 4 |
| croix-augas | 2,102 | 8 |
| ferion | 2,022 | 3 |
| valbonne | 1,718 | 4 |
| serre-de-barre | 329 | 6 |

Sizes are badly skewed — `brison` alone is 38% of the corpus.

**Consequence:** a single held-out-site test set is dominated by whichever tower lands in
it, so its score measures that tower, not generalization. We therefore make
**leave-one-site-out CV (8 folds)** the headline evaluation and report mean *and spread*.
The spread is the finding, not noise: a model scoring 0.85 on one tower and 0.55 on
another has learned terrain, not smoke.

Two folds need caveats when reported: `brison` puts 38% of the data in test (training on
only 16.7k images), and `serre-de-barre` has just 329 test images (noisy).

## Smoke is tiny

| Box area (% of image) | |
|---|---|
| mean | 0.402% |
| median | **0.138%** |
| p90 | 0.715% |
| p99 | 3.98% |
| under 1% of area | **93.3% of boxes** |

**Median box is ~41×31 px on a 1280×720 frame.** At YOLO's default 640px input that
becomes ~20×15 px. This is tighter than PyroNear2024's published ~1.04% mean area, and it
is the argument for tiling rather than naive downscaling.

## Two known traps, recorded

**1. Class index.** pyro-sdis ships its single smoke class with class id **`1`**, but
Ultralytics expects zero-based indices — with `nc: 1` the only legal id is `0`. Left
unfixed, every one of the 32,109 boxes is out-of-range and the model trains on zero
positives. `to_yolo()` remaps this. Do not remove it.

**2. Base rate.** The corpus is **83.7% positive.** Deployment is nowhere near this — a
tower sees no smoke on the overwhelming majority of frames. Any precision measured on this
split is therefore *optimistic* relative to the field. Report false-positives-per-image on
the negative subset alongside precision, and state the base-rate caveat.

## Splits produced

`data/processed/<name>.yaml` + `data/processed/splits/<name>_{train,val,test}.txt`

- `random` — image-level shuffle. **Leaky by construction** (all 8 sites appear in both
  train and test). Kept deliberately so the inflation can be *quantified*, not just asserted.
- `grouped` — site-disjoint. train: brison/courmettes/croix-augas/valbonne · val:
  cabanelle/ferion · test: marguerite/serre-de-barre.
- `loso_<site>` × 8 — leave-one-site-out folds.

Images are materialized once; each split is only a manifest of paths, so all 10 splits
share one 3.3 GB copy of the corpus instead of 26 GB of duplicates.
