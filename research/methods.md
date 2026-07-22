# Modeling Methods for Smoke Detection — Compare and Contrast

## The framing problem that drives every architectural choice

**Smoke is not an object.** Three structural facts:

1. **No crisp boundary.** Smoke is semi-transparent and fluid; a pixel is *partly* smoke.
   This is closer to alpha matting than object detection. Yuan et al. note it is "extremely
   time-consuming and difficult to manually segment smoke objects... since smoke has fuzzy
   edges and translucent property" — which is why they generated 8,162 *synthetic*
   composited images instead of hand-labeling. (https://arxiv.org/pdf/1809.00774)
2. **Early smoke is tiny.** In PyroNear2024, smoke boxes average **1.043% of image area**.
   On a 2048×3072 HPWREN frame an incipient plume is a handful of pixels — it does not
   survive downsampling to 224×224. Hence **tiling**.
3. **A single hazy frame is ambiguous even to humans.** Cloud, fog, dust, haze, and glare all
   look like smoke in a still image. What distinguishes smoke is that it *emerges from a
   fixed point and grows*. This is the entire argument for temporal models.

## The families

### Classical CV (pre-2015)
Background subtraction / frame differencing → color filtering (RGB/HSI/YUV/YCbCr) → dynamic
texture features (wavelet energy decay, edge-orientation histograms, motion vectors) → SVM.

Genuinely exploitable physics — smoke really does attenuate high-frequency background detail.
But every cue is confounded: background subtraction assumes a static camera (PTZ cameras pan,
trees sway, clouds move); wavelet-energy decay fires on fog and defocus; color thresholds on
grey smoke collide with cloud and dust. Thresholds are hand-tuned per camera and per time of
day, so nothing transfers.

Still worth implementing **as a baseline you then beat**, and frame-differencing is a useful
*input channel* to a modern network.

### CNN classification (ResNet / EfficientNet / MobileNet, tiled)
SmokeyNet's own single-frame baseline on FIgLib: **ResNet50 = 68.51% acc, F1 74.30,
precision 63.35, recall 89.89.**

Look at the *shape* of that error: recall high, **precision collapsed**. The single-frame CNN
screams "smoke" at every cloud. That is the signature failure of this family and reproducing
it is more instructive than beating it.

Cheapest to train, best edge deployability (MobileNet/EfficientNet-lite on a Raspberry Pi).

### Object detection (YOLO / Faster R-CNN / RetinaNet / DETR)
Most-published family, largely because YOLO is easy and datasets ship in COCO format.

- YOLOv8 fine-tuned: mAP@50 92.6% — **but on curated data with large obvious plumes**, not
  incipient smoke. [abstract-only]
- PyroNear2024 on realistic data: **F1 0.60–0.70**, vs 0.87–0.91 on prior easier datasets.
- FIgLib detector baselines are weak: Faster R-CNN 71.56% acc / F1 66.92; Mask R-CNN 73.24%
  / F1 69.94 — both precision-heavy, recall-starved.
- **Nemo** (DETR): on 95 HPWREN sequences, **97.9% of incipient fires detected, 80% within 5
  minutes, mean detection time 3.6 min.** DETR wins by dropping anchors/region proposals.

**How papers handle the ill-defined box** — this is the interesting part:
- **Relaxed IoU.** SKLFS evaluates at **AP@0.1**, not AP@0.5, explicitly "accounting for
  boundary ambiguity." (https://arxiv.org/html/2311.10116v3)
- **Abandon mAP entirely.** PyroNear2024 reports precision/recall/F1 on smoke *presence*,
  "focusing on binary classification of smoke presence rather than precise boundary
  delineation."
- **Convert transparency into label structure.** Nemo adds density classes keyed to perceived
  pixel density.
- **Separable negative sampling.** SKLFS samples negatives from *positive* images at 10×, but
  from *negative* images at **190×** with OHEM — so ambiguous smoke edges aren't mislabeled
  background while genuinely hard false positives get hammered.

Failure modes: cloud false positives ("clouds are the biggest interference factor" — SKLFS);
duplicate/unstable boxes because extent is undefined; brutal cross-dataset degradation;
threshold fragility (optimal confidence thresholds ranged **0.04–0.19**).

Pyronear ships YOLOv8 on Raspberry Pi and found **YOLOv9/v10 did not beat v8.**

### Segmentation (U-Net / DeepLab / SegFormer) — the alpha-matte view
Conceptually the most faithful framing: predict per-pixel smoke *density*, not a box.

The field's most interesting trick and its biggest validity risk are the same thing:
**synthetic compositing** (SYN70K). Linear alpha compositing is *not* how real smoke scatters
light, so models trained on it may be fitting the compositing operator. MIFNet reports mIoU
81.6% on SYN70K [abstract-only] — on synthetic data.

Failure modes: synthetic→real distribution gap; boundary mIoU is near-meaningless on a gradient;
whole-sky fog becomes a whole-image mask.

### Temporal / spatiotemporal — **the key family**

**Why temporal context matters more here than in almost any vision task:** a static grey
diffuse blob is genuinely ambiguous — cloud, fog, dust, or smoke, at roughly equal prior.
What is *not* ambiguous is the dynamics. Smoke **originates at a fixed ground point, grows
monotonically, and advects.** Clouds translate rigidly and don't emanate from a point; fog
doesn't grow from a source. **The evidence for ignition is in the difference between frames,
not in any frame.**

**SmokeyNet** (https://ar5iv.labs.arxiv.org/html/2112.08598):
- Input: current frame + previous frame, each cut into **45 overlapping 224×224 tiles** (20px overlap)
- **ResNet34** → tile embedding per tile per frame
- **LSTM** fuses the two frames *per tile* → temporal context at tile granularity
- **ViT** attends *across* the 45 tiles → global context → image-level prediction
- 56.9M params, **51.6 ms/image on a 2080 Ti**

| Model | Acc | F1 | P | R |
|---|---|---|---|---|
| ResNet50 (single frame) | 68.51 | 74.30 | 63.35 | 89.89 |
| Faster R-CNN | 71.56 | 66.92 | 81.34 | 56.88 |
| Mask R-CNN | 73.24 | 69.94 | 81.08 | 61.51 |
| ResNet34 + LSTM | 79.35 | 79.21 | 82.00 | 76.74 |
| ResNet34 + ViT (1 frame) | 82.53 | 81.30 | 88.58 | 75.19 |
| **SmokeyNet (CNN+LSTM+ViT)** | **83.49** | **82.59** | **89.84** | **76.45** |
| Human experts (3 lab members) | 78.5 | 82.8 | 93.5 | 74.4 |

Mean time-to-detection **3.12 minutes**.

**Read the ablation carefully.** Adding temporal context to a plain CNN buys ~11 points of
accuracy and **+19 points of precision**. The gain is almost entirely *false-positive
suppression* — exactly what the ambiguity argument predicts. That is a causal story about the
data, not a leaderboard climb.

Corroborating: PyroNear2024's CNN-LSTM over a YOLOv8s single-frame baseline gives **+9.3%
recall at equal precision**, and detection time drops from 1:46 → 1:05.

Failure modes (SmokeyNet's own): low cloud and haze still cause false positives; faint smoke
in the first minutes still missed; smoke occluded by infrastructure; and **camera motion
breaks the frame-pair assumption** — the approach assumes a fixed view, which is why fixed
HPWREN cameras are the natural substrate and PTZ/drone footage is not.

### Transformer / foundation models
Nemo's DETR beats anchor-based detectors on early smoke. SKLFS replaces standard patch
embedding with **Cross Contrast Patch Embedding** because plain Transformers are weak at the
*low-level* contrast/transparency cues that define smoke — a concrete statement that ViTs
alone under-model smoke.

**CLIP/SAM zero-shot for smoke is an unvalidated gap.** No peer-reviewed wildfire-smoke paper
doing this was found. The nearest prior art is **LangGas** (language-guided zero-shot
detection of semi-transparent *gas leaks*, https://arxiv.org/pdf/2503.02910) — same physics,
plausible to port. Interesting as a stretch section; do **not** stake the project on it. (SAM
is known to degrade badly out-of-distribution: https://arxiv.org/pdf/2401.08787)

### Satellite / multispectral — a *different problem*, not a harder one
Physics, not appearance: MIR/TIR brightness-temperature contrast, reflectance ratios, aerosol
optical depth.

The resolution/cadence tradeoff is the whole game: GOES ABI = ~5-min cadence but **~2 km
pixels**; VIIRS = 375 m but ~twice daily. At those resolutions **an incipient plume is
sub-pixel.** You are not doing early detection — you are doing plume *mapping* after the fire
is already large.

Ground cameras answer "is a fire starting now." Satellites answer "where is the smoke going."

## Comparison

| Family | Reported perf | Compute | Data hunger | Edge | Signature failure |
|---|---|---|---|---|---|
| Classical CV | High on tiny curated sets; unusable field FPR | Trivial | ~none | Excellent | Camera motion, fog; zero transfer |
| CNN classification | FIgLib ResNet50: 68.5% acc, **P 63.4 / R 89.9** | Low | Moderate | **Best** | **Precision collapse on clouds** |
| Object detection | Curated mAP@50 92.6%; realistic **F1 0.60–0.70** | Medium | High | Very good | mAP is the wrong metric; cross-dataset collapse |
| Segmentation | mIoU 81.6% on **synthetic** SYN70K | Med-high | Very high, or synthetic | Moderate | Synthetic→real gap |
| **Temporal** | **SmokeyNet 83.5% acc, F1 82.6, TTD 3.12 min** — beats human F1 (82.8) | Med-high (~2× CNN) | Needs *sequences* | Good | Requires fixed camera |
| Transformer | Nemo: 97.9% of incipient fires, 3.6 min TTD | High | Highest | Poor | ViTs under-model transparency cues |
| Satellite | Not comparable | Low-med | Weak labels | N/A | **Sub-pixel incipient plumes — it's plume mapping, not detection** |

## Practical tricks that decide the outcome

1. **Tiling.** 45 tiles of 224×224 with 20px overlap. Overlap matters — a plume straddling a
   boundary gets halved. Creates a weak-supervision problem: how do you get tile labels?
   (Derive from boxes; or MIL — image positive if any tile positive.)
2. **Hard-negative mining.** Where projects are won. Nuisance classes documented across the
   literature: **clouds, fog, dust, haze, sun glare, headlights, street lights, flags.**
3. **Class imbalance.** FIgLib is ~50/50 at *image* level but tile-level positives are **1–2%
   of tiles.** Your loss must reflect that (focal loss, OHEM); your metric must not be tile accuracy.
4. **Metrics.** Not mAP@0.5. Use precision/recall/F1 + AUC, relaxed AP@0.1 if localizing, and
   **time-to-detection** + **false-positives-per-camera-per-day**.
5. **Splits.** By fire, and ideally by camera. Random image splits leak 10+ points.

## Recommendation

**Baseline: tiled CNN classifier on FIgLib, evaluated rigorously.** ResNet34/EfficientNet-B0
over 224×224 tiles, split by fire, reported as P/R/F1 + time-to-detection + false-alarms-per-day.
**The point is not the number — it is to reproduce the precision collapse.** Showing 90%
recall at 63% precision and diagnosing that every false positive is a cloud is a better
artifact than 92% mAP on a scraped Kaggle set.

**Differentiator: add temporal context — SmokeyNet-style CNN→LSTM→attention over tiles.**
Because:
1. It demonstrates the insight the field converged on, and the published ablation is
   unambiguous: **+19 points of precision, from false-positive suppression.** You can
   reproduce that delta and explain *why*. That is a causal story about data, which is what
   interviews actually test.
2. It forces every hard engineering decision to be visible: tile grid, tile-label derivation,
   sequence-aware splits, class imbalance.
3. It gives you a real hard-negative-mining chapter: mine the baseline's false positives
   (they'll be clouds), retrain with OHEM, show the precision curve move.
4. It's cheap: 56.9M params, 51.6 ms/frame, public data.

**Ablation ladder:** single-frame CNN → + frame-differencing channel → + LSTM → + ViT.
Four bars, one clean story.

**Avoid:** plain YOLO on a scraped fire dataset (highest headline number, lowest signal, and
everyone has done it); satellite as the main project (different job, weak labels); CLIP/SAM
zero-shot as the core bet (unvalidated).

## Works Cited

- de Venâncio, P. V. A. B., Lisboa, A. C., & Barbosa, A. V. (2022). An automatic fire detection
  system based on deep convolutional neural networks for low-power, resource-constrained devices
  [D-Fire dataset]. *Neural Computing and Applications, 34*, 15349–15368.
  https://doi.org/10.1007/s00521-022-07467-z
- Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., &
  Nguyen, M. H. (2022). FIgLib & SmokeyNet: Dataset and deep learning model for real-time
  wildland fire smoke detection. *Remote Sensing, 14*(4), 1007. https://doi.org/10.3390/rs14041007
- Lostanlen, M., Isla, N., Guillen, J., Zanca, R., Veith, F., Buc, C., & Barriere, V. (2024).
  Constructing a real-world benchmark for early wildfire detection with the new PYRONEAR-2025
  dataset. *arXiv:2402.05349.* https://arxiv.org/abs/2402.05349
- *Multimodal wildland fire smoke detection.* (2023). *Remote Sensing, 15*(11), 2790.
  https://arxiv.org/abs/2212.14143 (author list not individually verified)
- Pyronear. (n.d.). *pyro-vision* [Software]. GitHub. https://github.com/pyronear/pyro-vision
- *SAM out-of-distribution limits for smoke.* (2024). *arXiv:2401.08787.*
  https://arxiv.org/abs/2401.08787 (author list not individually verified)
- *Semi-transparent object zero-shot detection (LangGas).* (2025). *arXiv:2503.02910.*
  https://arxiv.org/abs/2503.02910 (author list not individually verified)
- Wang, C., Xu, C., & Akram, A. (2023). Wildfire smoke detection system: Model architecture,
  training mechanism, and dataset [SKLFS-WildFire; Cross Contrast Patch Embedding].
  *arXiv:2311.10116.* https://arxiv.org/abs/2311.10116
- Yazdi, A., Qin, H., Jordan, C. B., Yang, L., & Yan, F. (2022). Nemo: An open-source
  transformer-supercharged benchmark for fine-grained wildfire smoke detection. *Remote Sensing,
  14*(16), 3979. https://doi.org/10.3390/rs14163979
- Yuan, F., Zhang, L., Xia, X., Wan, B., Huang, Q., & Li, X. (2019). Deep smoke segmentation.
  *Neurocomputing, 357*, 248–260. https://arxiv.org/abs/1809.00774
