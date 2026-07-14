# Existing Smoke Detection Projects — Survey

Research phase. Every link below was checked by the researching agent; caveats are noted
inline where a resource turned out to be stale, gated, or unverifiable.

## Production / research systems (no public code — cite as motivation, don't fork)

### ALERTWildfire (UCSD / UNR / U Oregon)
- https://www.alertwildfire.org/
- 1,200+ PTZ mountaintop cameras across CA/NV/OR/WA/ID. AI layer flags smoke, humans at
  fire dispatch confirm. This is the camera network that FIgLib and Nemo draw frames from.
- Docs quality 2/5 — program page, not an engineering artifact. No repo, no model card.

### HPWREN + FIgLib + SmokeyNet (WIFIRE Lab, San Diego Supercomputer Center / UCSD)
- Paper: https://www.mdpi.com/2072-4292/14/4/1007
- Dataset: http://hpwren.ucsd.edu/HPWREN-FIgLib/
- Closest runnable code: https://github.com/iperezx/sage-smoke-detection
- FIgLib: ~25,000 labeled smoke images (1536x2048 / 2048x3072) from fixed mountaintop
  cameras, ~30GB, 400+ fire sequences. SmokeyNet is a spatiotemporal architecture
  (CNN + LSTM/ViT temporal fusion over image tiles) reported to rival human performance.
- **Caveat:** no official standalone SmokeyNet repo appears to exist. The paper's training
  code is not public; `sage-smoke-detection` is a downstream SAGE-platform edge integration.
- Bounding boxes exist for only a subset of fires; the provided labeling tool is
  explicitly unfinished.
- Docs quality 3.5/5.

### Nemo (Nevada Smoke Detection Benchmark)
- Paper: https://www.mdpi.com/2072-4292/14/16/3979
- Repo: https://github.com/SayBender/Nemo
- Fine-grained smoke *density* detection + bboxes, from 1,073 ALERTWildfire videos, COCO
  format. Model is DETR — transformer detection, no anchors, no NMS.
- **Caveat:** the dataset download link in the repo is a literal placeholder
  ("ADD A URL ONCE DATASET IS PUBLIC"). Data access is unverified despite open-source framing.
- Docs quality 3.5/5.

### Google FireSat (Google Research / Earth Fire Alliance / Muon Space)
- https://sites.research.google/gr/wildfires/firesat/
- Purpose-built IR satellite constellation, detects fires down to ~5x5m, ~20 min global
  refresh. Detection works by comparing against ~1,000 prior images of the same location
  plus weather context — temporal-anomaly detection, not frame classification.
- Docs quality 1.5/5 for engineering purposes. Proprietary hardware and models.

### Descartes Labs Wildfire Detector
- https://medium.com/descarteslabs-team/the-satellites-hunting-for-megafires-afa1305fdc2c
- Thermal-IR satellite hotspot anomaly detection vs historical baseline, ~9 min latency.
  Notably alerted on the Kincade Fire before 911 calls.
- Docs quality 2/5 — good narrative, no code. Useful as an example of a genuinely
  different paradigm (anomaly-vs-baseline rather than CNN classification).

## Open source, between production and portfolio scale

### Pyronear (French nonprofit) — best-documented project found
- Org: https://github.com/pyronear
- Edge inference: https://github.com/pyronear/pyro-engine
- Models: https://github.com/pyronear/pyro-vision
- Alert API: https://github.com/pyronear/pyro-api
- Dashboard: https://github.com/pyronear/pyro-platform
- Full open pipeline: camera -> edge inference -> alert API -> dashboard. PyTorch models
  exported to ONNX, Raspberry Pi-class targets, Docker Compose.
- Docs quality 4.5/5. **The only project confirmed actively maintained** — v1.0.13
  released July 2026, 16 releases, live CI/CD and docs site.
- Study this for how a real deployed system decomposes into separate concerns.

### AI For Mankind — Open Wildfire Smoke Detection
- Org: https://github.com/aiformankind
- Dataset: https://github.com/aiformankind/wildfire-smoke-dataset
- Tutorial: https://github.com/aiformankind/wildfire-smoke-detection-camera
- TensorFlow Faster R-CNN / ResNet101 on 744 hand-annotated HPWREN images (XML bboxes).
  Published variants: SuperDuper-v1 (0.75 AP@0.5), v2 (0.87 AP), edge (0.68 AP).
- Reports honest failure modes: 8.6% average false-positive rate, **39.8% in fog**.
- Docs quality 4/5. Closest analog to a well-scoped portfolio project — small enough to
  train in Colab, structured as an explicit tutorial.
- **Caveat:** TF 1.x-era Faster R-CNN code; check version pinning before depending on it.

### Fuego / Open Climate Tech — Firecam
- https://github.com/fuego-dev/firecam (**archived read-only since May 2020**)
- Successor org: https://github.com/open-climate-tech
- HPWREN images -> classifier -> notify, with GCP notification layer.
- Docs quality 2/5. Stale — check the successor org before using.

## Portfolio-scale references (realistic targets)

### RihabFekii/wildfire-smoke-detector — best structural template
- https://github.com/RihabFekii/wildfire-smoke-detector
- YOLOv8 + Roboflow data + DVC (data/pipeline versioning) + MLflow (experiment tracking),
  Makefile-driven, `dvc exp run` workflow.
- Docs quality 3/5 — clean structure, but no reported metrics or dataset detail.
- Copy the repo layout and MLOps hygiene, not the model.

### Roboflow Universe — Wildfire Smoke object detection dataset
- https://public.roboflow.com/object-detection/wildfire-smoke/1
- Pre-labeled bboxes, one-click export to YOLO/COCO. This is what most YOLOv8 smoke repos
  actually train on.
- The pragmatic MVP dataset: much easier than FIgLib, at the cost of being less "real"
  (static labeled frames, not full fire sequences).

### Abonia1/YOLOv8-Fire-and-Smoke-Detection
- https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection
- Companion writeup: https://medium.com/@abonia/yolov8-fire-and-smoke-detection-d9b74b9d0d97
- Runnable training notebook + paired blog post. Low dependency-rot risk (ultralytics is
  still current).
- Docs quality 3.5/5. The repo + writeup pairing is a good presentation pattern to copy.

## Bottom line

- **Data:** FIgLib for authenticity, Roboflow's public set for a fast YOLO-formatted MVP.
- **Structure:** RihabFekii for MLOps hygiene, Abonia1 for presentation, AI For Mankind for
  how to scope a small detector and report failure modes honestly.
- **Systems thinking:** Pyronear.
- **Don't try to reproduce:** ALERTWildfire, FireSat, Descartes Labs.
