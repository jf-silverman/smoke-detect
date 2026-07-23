# Background Topics

A growing reference of background primers for this project. Each section is a self-contained
explainer on a topic that context for the smoke-detection work draws on. Sections are added as
the need comes up.

**Contents**
1. [Object detection — a primer (with video and cross-domain methods)](#1-object-detection--a-primer)

---

## 1. Object detection — a primer

### 1.1 What object detection is

Object detection answers two questions at once for every object in an image: **where is it**
(localization) and **what is it** (classification). It sits between two neighbors:

- **Image classification** — one label for the whole image ("this image contains a dog"). No
  location.
- **Semantic / instance segmentation** — a class for every *pixel* (semantic) or a per-object
  pixel mask (instance). Finer than a box, more expensive to label and compute.

Detection's usual output is a set of **bounding boxes**, each with a class label and a confidence
score. Variants output rotated boxes (remote sensing), keypoints (pose), or masks (Mask R-CNN
bridges detection and instance segmentation).

Two facts shape everything downstream. First, the number of objects is not fixed, so the model
must emit a variable-length set and then **suppress duplicates** — classically with
non-maximum suppression (<abbr title="Non-Maximum Suppression — removes duplicate, overlapping detections of the same object, keeping the highest-confidence box">NMS</abbr>). Second, most of an image is background, so detectors fight a
severe **foreground/background imbalance** (addressed by <abbr title="A classification loss that down-weights easy, well-classified examples so training focuses on hard ones — the fix for dense detectors' background/foreground imbalance">focal loss</abbr> and <abbr title="Training on the negatives a model gets wrong (clouds, fog, glare it false-alarms on) rather than random negatives, to cut the dominant false-positive class">hard-negative mining</abbr>;
Lin et al., 2017b).

### 1.2 The architectural families

**Two-stage detectors (region-based).** First propose candidate regions, then classify and refine
each. The <abbr title="Region-based Convolutional Neural Network — the two-stage detector lineage (R-CNN → Fast → Faster → Mask R-CNN): propose candidate regions, then classify and refine each">R-CNN</abbr> lineage — R-CNN (Girshick et al., 2014) → Fast R-CNN (Girshick, 2015) → Faster
R-CNN (Ren et al., 2015), which made region proposals learnable — set the accuracy standard for
years. Mask R-CNN (He et al., 2017) adds a mask head. Strengths: accuracy, small objects.
Weakness: slower, harder to run in real time.

**One-stage detectors (dense prediction).** Predict boxes and classes directly over a grid in a
single pass — no proposal stage. YOLO (Redmon et al., 2016) and SSD (Liu et al., 2016) traded a
little accuracy for a large speed gain; RetinaNet (Lin et al., 2017b) closed much of the accuracy
gap with **focal loss**, which down-weights easy background. This family dominates real-time use;
the modern YOLO releases (v8/v11 from Ultralytics, plus v12) are one-stage. **This project's
detector, yolo11n, is a one-stage model.**

**Anchor-based vs anchor-free.** Early one-stage detectors tiled each grid cell with preset
"anchor" boxes of assorted shapes and predicted offsets from them. Anchors add hyperparameters
and a scale/aspect prior. **Anchor-free** detectors — FCOS (Tian et al., 2019), CenterNet (Zhou
et al., 2019) — predict object centers and sizes directly, simplifying the design. Recent YOLOs
are largely anchor-free.

**Transformer / set-prediction detectors.** <abbr title="DEtection TRansformer — reframes detection as direct set prediction with bipartite (Hungarian) matching; no anchors and no NMS">DETR</abbr> (Carion et al., 2020) reframed detection as
direct **set prediction**: a transformer emits a fixed set of predictions matched to ground truth
by <abbr title="A one-to-one assignment between predicted and ground-truth boxes, computed by the Hungarian algorithm; lets DETR train as direct set prediction with each object matched to exactly one prediction">bipartite (Hungarian) matching</abbr> — **no anchors and no NMS**. It was elegant but slow to train;
Deformable DETR (Zhu et al., 2021) and DINO (Zhang et al., 2022) fixed convergence and pushed
accuracy to the top of the benchmarks, and RT-DETR (Zhao et al., 2023) brought the approach into
real-time territory, making it a genuine competitor to YOLO.

**Backbones** (the feature extractor under any of the above) evolved in parallel: ResNet (He et
al., 2016), EfficientNet (Tan & Le, 2019), and the transformer backbones <abbr title="Vision Transformer — applies the transformer architecture to image patches instead of convolutions">ViT</abbr> (Dosovitskiy et al.,
2021) and Swin (Liu et al., 2021). Feature Pyramid Networks (<abbr title="Feature Pyramid Network — fuses features across resolution scales so a detector can find both large and small objects">FPN</abbr>; Lin et al., 2017a) let a detector
use multiple scales at once — important for small objects like distant smoke.

### 1.3 How performance is measured (and a caveat)

The standard metric is **mean Average Precision (<abbr title="mean Average Precision — area under the precision–recall curve per class, averaged over classes and IoU thresholds; the standard object-detection score">mAP</abbr>)**: average precision (area under the
precision-recall curve) per class, meaned over classes and over IoU thresholds (COCO averages IoU
0.50–0.95; Lin et al., 2014). **Intersection-over-Union (<abbr title="Intersection over Union — area of overlap ÷ area of union between a predicted and a ground-truth box; measures localization quality">IoU</abbr>)** measures box overlap.

The caveat this project rests on: mAP assumes objects have crisp boundaries against which IoU is
meaningful. For an amorphous, boundary-less object like smoke, IoU is ill-defined and mAP is a
poor proxy for field usefulness — which is why the [metrics report](metrics.md) reframes
evaluation around detection rate and false-alarm burden instead. mAP is a general-purpose metric,
not a universal one.

### 1.4 Current best methods (2024–2026)

- **Real-time:** the recent **YOLO** releases (v8/v11/v12) and **RT-DETR** are the practical
  front-runners; the choice is often ecosystem and latency, not raw accuracy.
- **Highest accuracy:** DETR descendants — **DINO**, **Co-DETR**, and ensembles with large
  transformer backbones — top the COCO leaderboards.
- **Open-vocabulary / promptable detection** is the newer frontier: detect classes named at
  inference by text rather than fixed at training. **Grounding DINO** (Liu et al., 2023),
  **YOLO-World** (Cheng et al., 2024), and **OWL-ViT** (Minderer et al., 2022) detect arbitrary
  described objects; **SAM** (Kirillov et al., 2023) segments anything given a point or box prompt.
  These matter when labels are scarce — though, as the [research notes](../research/methods.md)
  record, they degrade on out-of-distribution, semi-transparent targets like smoke.

### 1.5 Detection in *video*

Video is not merely a stack of independent images. It brings both a problem and a gift.

**Why video differs from image-by-image detection.**
- **Temporal redundancy and cost.** Consecutive frames are near-duplicates. Running a full
  detector on all of them is wasteful, and it also means benchmarks leak badly if consecutive
  frames land in both train and test (the same leakage this project guards against by splitting on
  fire/site, not frame).
- **Per-frame degradation.** Motion blur, defocus, and partial occlusion make individual frames
  ambiguous in ways a single-image detector cannot resolve.
- **Coherence to exploit.** An object present now was very likely present a moment ago and moved a
  little — a strong prior a per-frame model throws away.
- **Streaming constraints.** Many uses (driving, surveillance, fire watch) are online: the model
  must decide on frame *t* using only frames up to *t*, under a latency budget. This project's
  temporal head uses exactly such a **causal window** ([temporal findings](temporal-findings.md)).

**The main approaches, roughly in order of integration:**
1. **Post-processing the per-frame outputs.** Run an image detector, then reconcile boxes across
   time — Seq-NMS (Han et al., 2016) links detections into tracks and rescores them. Cheap, no
   retraining.
2. **Tracking-by-detection.** Detect per frame, then associate identities across frames with a
   tracker — SORT (Bewley et al., 2016), DeepSORT (Wojke et al., 2017), ByteTrack (Zhang et al.,
   2022). This is the workhorse of surveillance and driving: the detector finds, the tracker
   persists and smooths.
3. **Feature-level temporal aggregation.** Fuse *features* across frames before predicting —
   Flow-Guided Feature Aggregation (Zhu et al., 2017) warps neighboring features along optical
   flow; MEGA (Chen et al., 2020) and TransVOD (He et al., 2021) aggregate with attention over a
   window. This recovers the plume-emerges-over-time signal a single frame lacks.
4. **Native video models.** 3D <abbr title="Convolutional Neural Network — the standard image-feature backbone built from learned convolutional filters">CNN</abbr>s, Conv<abbr title="Long Short-Term Memory — a recurrent neural network that carries state across a sequence of frames">LSTM</abbr>s, and video transformers — TimeSformer (Bertasius
   et al., 2021), ViViT (Arnab et al., 2021) — treat time as a first-class axis. Most powerful,
   most data- and compute-hungry. **SmokeyNet** (Dewangan et al., 2022), the reference smoke
   model, is of this family: a CNN per tile, an LSTM across two frames, and attention across tiles.

The project's own temporal experiments are a lightweight instance of category 3/4 (a <abbr title="Gated Recurrent Unit — a recurrent neural network for sequences, simpler than an LSTM">GRU</abbr> over a
frozen detector's per-frame evidence), and its central finding is that the temporal gain is
**dataset-dependent** — it needs onset sequences to pay off ([temporal](temporal-findings.md),
[figlib](figlib-findings.md)).

### 1.6 What different domains actually use

The same detection toolbox is specialized very differently across domains, driven by the data and
the cost of errors:

- **Autonomous driving.** Real-time one-stage detection *plus* 3D: LiDAR point-cloud detectors
  (PointPillars, Lang et al., 2019; CenterPoint, Yin et al., 2021) and multi-camera bird's-eye-view
  transformers (BEVFormer, Li et al., 2022), fused across sensors and coupled to tracking and
  motion forecasting. Latency and 3D localization dominate; a missed pedestrian is the asymmetric
  cost.
- **Medical imaging / radiology / biology.** Often **segmentation-first** — U-Net (Ronneberger et
  al., 2015) and nnU-Net (Isensee et al., 2021) — because clinicians need precise extent, not a
  box; detection is used for lesions, nodules, and cell counting. Data is 3D (CT/MRI volumes),
  small-object-heavy, class-imbalanced, and label-scarce, so heavy augmentation and cross-
  validation matter.
- **Remote sensing / satellite.** Huge images tiled into patches, **oriented** bounding boxes
  (objects at any rotation), and extreme small-object scales — the same tiling logic this project
  needed for high-resolution FIgLib frames ([resolution](resolution-findings.md)).
- **Manufacturing / retail.** Defect and anomaly detection, often with few or no positive
  examples, pushing toward one-class and self-supervised methods rather than supervised boxes.
- **Ecology / wildlife.** Camera-trap detection paired with **re-identification** to count and
  track individuals; long-tailed species distributions.
- **Security / surveillance.** Detection + multi-object tracking + re-ID over continuous video —
  the tracking-by-detection stack above.
- **Wildfire smoke (this project).** An amorphous target with no crisp boundary, frequently small
  and distant, developing over minutes, imaged at high resolution, under a steeply asymmetric cost
  (a missed fire ≫ a false alarm). That combination is why standard detection defaults (mAP, box
  IoU, F1, whole-frame downscaling, frame-level splits) transfer poorly, and why the project
  rebuilds evaluation around detection rate, false-alarm burden, tiling, and leak-safe splits.

### 1.7 The through-line for this project

The takeaways that recur in the rest of the reports:
- **One-stage detector (yolo11n)** for a real-time smoke task.
- **Tiling / native resolution** because the target is small (§1.6 remote sensing; §1.5 video).
- **Temporal modeling helps only when the data carries the signal** — motion and onset — which is
  the whole subject of the temporal and FIgLib findings.
- **Metrics and splits must be built for the domain**, not inherited from COCO.

### Works Cited

- Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C. (2021). ViViT: A video
  vision transformer. *ICCV.* https://arxiv.org/abs/2103.15691
- Bertasius, G., Wang, H., & Torresani, L. (2021). Is space-time attention all you need for video
  understanding? [TimeSformer]. *ICML.* https://arxiv.org/abs/2102.05095
- Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking
  [SORT]. *ICIP.* https://arxiv.org/abs/1602.00763
- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end
  object detection with transformers [DETR]. *ECCV.* https://arxiv.org/abs/2005.12872
- Chen, Y., Cao, Y., Hu, H., & Wang, L. (2020). Memory enhanced global-local aggregation for video
  object detection [MEGA]. *CVPR.* https://arxiv.org/abs/2003.12063
- Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., & Shan, Y. (2024). YOLO-World: Real-time
  open-vocabulary object detection. *CVPR.* https://arxiv.org/abs/2401.17270
- Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., &
  Nguyen, M. H. (2022). FIgLib & SmokeyNet: Dataset and deep learning model for real-time wildland
  fire smoke detection. *Remote Sensing, 14*(4), 1007. https://doi.org/10.3390/rs14041007
- Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition
  at scale [ViT]. *ICLR.* https://arxiv.org/abs/2010.11929
- Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate
  object detection and semantic segmentation [R-CNN]. *CVPR.* https://arxiv.org/abs/1311.2524
- Girshick, R. (2015). Fast R-CNN. *ICCV.* https://arxiv.org/abs/1504.08083
- Han, W., et al. (2016). Seq-NMS for video object detection. *arXiv:1602.08465.*
  https://arxiv.org/abs/1602.08465
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition
  [ResNet]. *CVPR.* https://arxiv.org/abs/1512.03385
- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *ICCV.*
  https://arxiv.org/abs/1703.06870
- He, L., et al. (2021). TransVOD: End-to-end video object detection with spatial-temporal
  transformers. *arXiv:2201.05047.* https://arxiv.org/abs/2201.05047
- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A
  self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods,
  18*, 203–211. https://doi.org/10.1038/s41592-020-01008-z
- Kirillov, A., et al. (2023). Segment Anything [SAM]. *ICCV.* https://arxiv.org/abs/2304.02643
- Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2019). PointPillars: Fast
  encoders for object detection from point clouds. *CVPR.* https://arxiv.org/abs/1812.05784
- Li, Z., et al. (2022). BEVFormer: Learning bird's-eye-view representation from multi-camera images
  via spatiotemporal transformers. *ECCV.* https://arxiv.org/abs/2203.17270
- Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). Microsoft COCO: Common objects in context.
  *ECCV.* https://arxiv.org/abs/1405.0312
- Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017a). Feature
  pyramid networks for object detection. *CVPR.* https://arxiv.org/abs/1612.03144
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017b). Focal loss for dense object
  detection [RetinaNet]. *ICCV.* https://arxiv.org/abs/1708.02002
- Liu, W., Anguelov, D., Erhan, D., et al. (2016). SSD: Single shot multibox detector. *ECCV.*
  https://arxiv.org/abs/1512.02325
- Liu, S., et al. (2023). Grounding DINO: Marrying DINO with grounded pre-training for open-set
  object detection. *arXiv:2303.05499.* https://arxiv.org/abs/2303.05499
- Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin Transformer: Hierarchical vision transformer using
  shifted windows. *ICCV.* https://arxiv.org/abs/2103.14030
- Minderer, M., et al. (2022). Simple open-vocabulary object detection with vision transformers
  [OWL-ViT]. *ECCV.* https://arxiv.org/abs/2205.06230
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified,
  real-time object detection [YOLO]. *CVPR.* https://arxiv.org/abs/1506.02640
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection
  with region proposal networks. *NeurIPS.* https://arxiv.org/abs/1506.01497
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical
  image segmentation. *MICCAI.* https://arxiv.org/abs/1505.04597
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural
  networks. *ICML.* https://arxiv.org/abs/1905.11946
- Tian, Z., Shen, C., Chen, H., & He, T. (2019). FCOS: Fully convolutional one-stage object
  detection. *ICCV.* https://arxiv.org/abs/1904.01355
- Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep
  association metric [DeepSORT]. *ICIP.* https://arxiv.org/abs/1703.07402
- Yin, T., Zhou, X., & Krähenbühl, P. (2021). Center-based 3D object detection and tracking
  [CenterPoint]. *CVPR.* https://arxiv.org/abs/2006.11275
- Zhang, H., Li, F., Liu, S., et al. (2022). DINO: DETR with improved denoising anchor boxes for
  end-to-end object detection. *arXiv:2203.03605.* https://arxiv.org/abs/2203.03605
- Zhang, Y., Sun, P., Jiang, Y., et al. (2022). ByteTrack: Multi-object tracking by associating
  every detection box. *ECCV.* https://arxiv.org/abs/2110.06864
- Zhao, Y., Lv, W., Xu, S., et al. (2023). DETRs beat YOLOs on real-time object detection [RT-DETR].
  *arXiv:2304.08069.* https://arxiv.org/abs/2304.08069
- Zhou, X., Wang, D., & Krähenbühl, P. (2019). Objects as points [CenterNet]. *arXiv:1904.07850.*
  https://arxiv.org/abs/1904.07850
- Zhu, X., Wang, Y., Dai, J., Yuan, L., & Wei, Y. (2017). Flow-guided feature aggregation for video
  object detection. *ICCV.* https://arxiv.org/abs/1703.10025
- Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2021). Deformable DETR: Deformable
  transformers for end-to-end object detection. *ICLR.* https://arxiv.org/abs/2010.04159

---

## Glossary

Acronyms above are also given as hover tooltips on first use (`<abbr>`); the definitions live here
in text so they are reachable on touch devices and by screen readers. If a tooltip does not appear
on GitHub, its HTML sanitizer has stripped the `title` attribute — this table is the source of truth.

| Term | Meaning |
|---|---|
| **mAP** | mean Average Precision — area under the precision–recall curve per class, averaged over classes and IoU thresholds. The standard object-detection score (COCO averages IoU 0.50–0.95). |
| **IoU** | Intersection over Union — area of overlap ÷ area of union between a predicted and a ground-truth box. Measures localization quality; ill-defined for boundary-less smoke. |
| **NMS** | Non-Maximum Suppression — post-processing that removes duplicate overlapping detections of the same object, keeping the highest-confidence box. DETR-family detectors avoid it. |
| **FPN** | Feature Pyramid Network — fuses features across resolution scales so one detector handles both large and small objects. |
| **CNN** | Convolutional Neural Network — the standard image-feature backbone built from learned convolutional filters. |
| **LSTM** | Long Short-Term Memory — a recurrent neural network that carries state across a sequence (e.g. frames of video). |
| **GRU** | Gated Recurrent Unit — a recurrent neural network for sequences, simpler and lighter than an LSTM. |
| **ViT** | Vision Transformer — applies the transformer (attention) architecture to image patches instead of convolutions. |
| **R-CNN** | Region-based CNN — the two-stage detector lineage (R-CNN → Fast → Faster → Mask R-CNN): propose regions, then classify and refine. |
| **YOLO** | You Only Look Once — the one-stage, real-time detector family; this project uses yolo11n. |
| **SSD** | Single Shot MultiBox Detector — an early one-stage dense detector. |
| **DETR** | DEtection TRansformer — reframes detection as direct set prediction with bipartite matching; no anchors, no NMS. |
| **RT-DETR** | Real-Time DETR — a DETR variant fast enough to compete with YOLO. |
| **FCOS / CenterNet** | Anchor-free one-stage detectors that predict object centers and sizes directly. |
| **SAM** | Segment Anything Model — a promptable segmentation model (point/box prompt → mask). |
| **BEV** | Bird's-Eye-View — the top-down representation used by multi-camera 3D detectors in autonomous driving. |
| **re-ID** | Re-identification — matching the same individual/object across cameras or time, paired with detection in tracking and wildlife counting. |
| **focal loss** | A modified classification loss (Lin et al., 2017b) that down-weights easy, well-classified examples so training focuses on hard ones — the fix for the extreme background/foreground imbalance in dense one-stage detectors. |
| **hard-negative mining** | Deliberately training on the negatives a model gets *wrong* (here: clouds, fog, glare it false-alarms on) rather than random negatives, to directly cut the dominant false-positive class. This project's main lever on the false-alarm burden. |
| **bipartite (Hungarian) matching** | A one-to-one assignment between a model's predicted boxes and the ground-truth boxes, computed by the Hungarian algorithm. Lets DETR train as direct set prediction — each object matched to exactly one prediction — so no anchors or NMS are needed. |
