# Public Wildfire / Smoke Datasets — Survey and Ratings

Each entry was checked against a live landing page, repo, LICENSE file, or S3 listing.
Sizes marked **estimated** are not published by the dataset authors — do not plan around them.

## The finding that matters most: leakage

This applies to **FIgLib, Nemo, pyro-sdis, AI-For-Mankind, and SmokeViz** — every dataset
drawn from a continuous camera or satellite feed.

FIgLib frames are spaced ~60 seconds apart from a **fixed** camera. Frames at t=+300s and
t=+360s of the same fire are effectively the same photograph: same ridge, same vegetation,
same camera pose, plume a few percent larger. Shuffle the ~25,000 images into a random
80/20 split and **a near-twin of every test image is in the training set.** You get a
spectacular score that measures memorization of 101 camera backgrounds, and the model
collapses on any camera it hasn't seen.

Because the cameras are fixed, a model can learn "this ridge + these JPEG artifacts → fire
sequence" without ever looking at the smoke.

**What the FIgLib authors did — copy it.** Split by fire, never by image:

| Split | Fires | Images |
|---|---|---|
| Train | 144 | 11.3K |
| Val | 64 | 4.9K |
| Test | 62 | 4.9K |
| Omitted | 45 | 3.7K |

**Go further: split by camera/station too.** Two different fires from the same camera still
share the whole background. Reporting both numbers side by side — "94% on a random split,
71% when I hold out cameras" — is exactly the result that signals you know what you're doing.

**If you see >95% validation accuracy on FIgLib, you have leaked.**

### Other FIgLib quality problems to name explicitly
- **Label noise at onset.** Labels derive from timestamp offset from visible plume
  appearance. Frames just after t=0 are labeled positive but the plume may be a few pixels
  or not yet visible. The positive class near t=0 is partly mislabeled by construction.
  SmokeyNet's preprocessing dropped 16 fires with "questionable smoke presence."
- **The canonical benchmark is day-only** — the authors dropped 19 night fires and 10 B&W
  images. Keep night frames and your numbers fall. Say so if you do.
- 114 of 315 fires are missing an average of 6.6 images each.
- **The 50/50 class balance is artificial.** In deployment smoke is rare, so a model tuned
  on this balance produces a far worse false-positive rate than its PR curve suggests.
- **Cross-dataset contamination:** FIgLib and the AI-For-Mankind Roboflow set come from the
  same HPWREN cameras and overlapping fires. Do not use one to "validate" the other.

## Camera datasets

### pyro-sdis (PyroNear) — rating 5/5, best starting point
- https://huggingface.co/datasets/pyronear/pyro-sdis
- **Apache-2.0** (most permissive license in the space), free, no gate.
- 33,636 images (28,103 with smoke, ~5,533 negatives), 31,975 smoke instances. **3.28 GB.**
- Parquet-backed, YOLO bboxes, Ultralytics-compatible out of the box.
- Wildland/forest, France, fixed detection towers. Smoke is typically **far, small,
  low-contrast** — genuinely hard.
- Resolution not stated on the card (unverified).
- Fixed cameras → same site-memorization risk. Heavily positive-skewed.

### D-Fire — rating 4/5, best negatives and best license
- https://github.com/gaia-solutions-on-demand/DFireDataset
- **CC0 1.0** per the LICENSE text (GitHub's API reports NOASSERTION because it's a custom
  file). Caveat: maintainers state they do not hold copyright over the scraped images.
- 21,527 images: 1,164 fire-only, 5,867 smoke-only, 4,658 both, **9,838 negatives**
  including deliberate fire-like distractors (sunsets, lights). 14,692 fire + 11,865 smoke boxes.
- **Estimated ~2–4 GB.** JPEG + YOLO txt, pre-split.
- Mixed context: urban, indoor, forest, vehicles. Web-sourced, so **not** wildland early
  detection.
- Varied cameras — a strength, no background memorization.
- Weakness: the fire class is too easy; headline mAP looks great for uninteresting reasons.

### FIgLib (HPWREN) — rating 4/5, where the real problem lives
- http://hpwren.ucsd.edu/HPWREN-FIgLib/ · index: https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html
- Free, no registration. **No formal license** — "provided as-is," asks for credit. Legally
  ambiguous, which matters if you want to publish images in a portfolio.
- 315 sequences, 101 cameras, ~24,800 images (SmokeyNet snapshot); archive now 400+ sequences.
- **~30 GB.** JPEG + MP4 timelapses. Filenames encode the label:
  `origin_timestamp_offset(sec)_from_visible_plume_appearance.jpg` — negative offset = pre-ignition.
- 1536×2048 or 2048×3072. Wildland/WUI, SoCal, fixed mountaintop cameras.
- ~81 images/sequence spanning −40/+40 min from ignition → ~50/50 balance by construction.
- See the leakage section above. The leakage story *is* the portfolio value.

### Nemo — rating 4/5, the only fine-grained density labels
- https://github.com/SayBender/Nemo · stats: https://datasetninja.com/nemo
- **Apache-2.0.** 2,934 images, 4,522 boxes, **1.12 GB**, COCO JSON.
- Frames from 1,073 AlertWildfire videos. 3072×2048 and 1920×1080 both appear.
- **Density ordinal: low (2,726) / mid (1,403) / high (393)** — low is the early incipient
  stage. Almost no portfolio project attempts ordinal early-stage detection.
- Near-duplicate risk within a fire; 7:1 class imbalance.
- Note: the *repo's* dataset download link is a placeholder, but Dataset Ninja mirrors it.

### AI For Mankind / Wildfire Smoke Dataset — rating 4/5, fastest on-ramp
- https://public.roboflow.com/object-detection/wildfire-smoke · https://github.com/aiformankind/wildfire-smoke-dataset
- **CC BY-NC-SA 4.0** (non-commercial). Roboflow mirror needs a free account.
- Roboflow mirror: 737 images, all positive, one class. Repo: v1.0 = 744, v2.0 = 2,192
  (Pascal VOC). Separate cloud/fog hard-negative set exists — use it.
- Under 1 GB (estimated). Same HPWREN cameras as FIgLib → **do not cross-validate between them.**

### Lower-value / avoid
- **FASDD** — rating 2/5. ~95k CV + 5.7k RS images, but the **ESSD preprint was withdrawn by
  its own authors** after review criticism; size/license unverified from the live SciDB page;
  it aggregates other public sets so it's contaminated w.r.t. D-Fire. Impressive scale, poor
  foundation. https://doi.org/10.57760/sciencedb.j00104.00103
- **FLAME / FLAME2 / FLAME3** — rating 2/5. Gated behind IEEE login. **41.7 GB.** It is a
  *single prescribed pile burn* shot continuously — the Fire/No-Fire task is near-degenerate
  (~99% means nothing). It's a *flame* dataset, not smoke; smoke masks exist only in the
  small FLAME2-DT subset (1,280 RGB-thermal pairs, rating 3/5).
- **Smoke100k** — rating 2/5. Synthetic composited smoke, perfect masks. Pretraining aid
  only; fatal if used for evaluation. https://bigmms.github.io/cheng_gcce19_smoke100k/
- **SMOKE5K** — rating 2/5. Real benchmark (AAAI'22) but **no authoritative first-party
  download page found**; circulates via community mirrors. 1,400 real + 4,000 synthetic.
- **Corsican Fire DB** — rating 1/5. Gated, requires signed license agreement that restricts
  redistribution (awkward for a public portfolio). ~500 images, near-field flame not smoke.
- **MIVIA/Foggia** — rating 1/5. Gated, 31 videos, 2015-era.
- **BoWFire** — rating 1/5. 226 images. Extra test set at most.
- **FIRE-SMOKE-DATASET (DeepQuestAI)** — rating 2/5. ~3,000 obvious images, classification
  only. 97% in an afternoon, impresses nobody.

## Satellite

### SmokeViz (NOAA) — rating 3/5, novel but huge
- https://noaa-gsl-experimental-pds.s3.amazonaws.com/index.html#SmokeViz/ · paper: https://openreview.net/forum?id=NheuvQEWDt
- Free, anonymous S3. GeoTIFF. **>160,000 annotations** from NOAA HMS analyst polygons over
  GOES-16 ABI. Pixel masks, 3 density classes.
- **Size unpublished; estimated several hundred GB.** Verify with
  `aws s3 ls --summarize --human-readable --no-sign-request` before downloading. Subset one year.
- Labels are human analyst polygons — subjective and coarse. Consecutive 5-min GOES frames
  are extremely near-duplicate; split by fire event and year.

### NASA FIRMS — rating 2/5 as data, 4/5 as a *label source*
- https://firms.modaps.eosdis.nasa.gov/
- Vector points (SHP/KML/CSV), **not imagery**. Thermal hotspots, not smoke.
- The smart use: join FIRMS hotspots to camera or Sentinel-2 scenes to generate weak
  supervision or confirm ground-truth ignition times.

### Raw Sentinel-2 / Landsat / GOES
Free and open, but **no smoke labels**. Building them is a project unto itself. Scope trap.

## Could not verify

- **"RIS-Fire"** — no dataset by this name found. The closest real thing is **FireRisk**
  (https://arxiv.org/abs/2303.07035), 91,872 images across 7 fire-*risk* classes — an aerial
  land-cover risk task, **not** smoke detection. Do not cite RIS-Fire as existing.

## Recommended project shape

Train on **pyro-sdis** → evaluate on a **held-out-camera** split → evaluate zero-shot on
**FIgLib** and **D-Fire** to quantify domain shift → add a temporal head on **FIgLib** and
report **time-to-detection** instead of accuracy.

## Works Cited

- AI For Mankind. (n.d.). *Wildfire smoke dataset* [Dataset]. GitHub / Roboflow.
  https://github.com/aiformankind/wildfire-smoke-dataset
- ALERTCalifornia / ALERTWildfire. (n.d.). *Wildfire camera network* [Live feeds]. University of
  California San Diego. https://alertcalifornia.org/ · https://www.alertwest.live/
- Bilkent University. (n.d.). *VisiFire sample clips* [Video dataset].
  http://signal.ee.bilkent.edu.tr/VisiFire/
- de Venâncio, P. V. A. B., Lisboa, A. C., & Barbosa, A. V. (2022). An automatic fire detection
  system based on deep convolutional neural networks for low-power, resource-constrained devices
  [D-Fire dataset]. *Neural Computing and Applications, 34*, 15349–15368.
  https://doi.org/10.1007/s00521-022-07467-z
- Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., &
  Nguyen, M. H. (2022). FIgLib & SmokeyNet: Dataset and deep learning model for real-time
  wildland fire smoke detection. *Remote Sensing, 14*(4), 1007. https://doi.org/10.3390/rs14041007
- FIRESENSE. (n.d.). *Database of videos for flame and smoke detection* [Dataset]. Zenodo.
  https://zenodo.org/record/836749
- HPWREN. (n.d.). *HPWREN Fire Ignition Library (FIgLib) and camera image archive* [Dataset].
  University of California San Diego. http://hpwren.ucsd.edu/HPWREN-FIgLib/ ·
  http://c1.hpwren.ucsd.edu/archive/
- Lostanlen, M., Isla, N., Guillen, J., Zanca, R., Veith, F., Buc, C., & Barriere, V. (2024).
  Constructing a real-world benchmark for early wildfire detection with the new PYRONEAR-2025
  dataset. *arXiv:2402.05349.* https://arxiv.org/abs/2402.05349
- MIVIA Lab, University of Salerno. (n.d.). *Fire and smoke detection datasets* [Video dataset].
  https://mivia.unisa.it/
- NASA. (n.d.). *Fire Information for Resource Management System (FIRMS)* [Data product].
  https://firms.modaps.eosdis.nasa.gov/
- NOAA Global Systems Laboratory. (n.d.). *SmokeViz* [Dataset].
  https://openreview.net/forum?id=NheuvQEWDt
- Pyronear. (2025). *pyro-sdis* [Dataset]. Hugging Face.
  https://huggingface.co/datasets/pyronear/pyro-sdis
- Yazdi, A., Qin, H., Jordan, C. B., Yang, L., & Yan, F. (2022). Nemo: An open-source
  transformer-supercharged benchmark for fine-grained wildfire smoke detection. *Remote Sensing,
  14*(16), 3979. https://doi.org/10.3390/rs14163979

<sub>Also referenced above (not smoke-detection datasets we recommend, cited for completeness):
FASDD (withdrawn preprint, https://doi.org/10.57760/sciencedb.j00104.00103), FireRisk
(arXiv:2303.07035), FLAME/FLAME2 (IEEE DataPort), Smoke100k
(https://bigmms.github.io/cheng_gcce19_smoke100k/), Corsican Fire DB, BoWFire.</sub>

Four honest numbers and one uncomfortable, well-explained drop. Worth more than any 0.97 mAP.
