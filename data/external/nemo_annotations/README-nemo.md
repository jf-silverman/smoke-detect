# Nemo annotations (archived 2026-07-28)

COCO-format smoke bounding-box annotations from the Nemo benchmark (SayBender/Yazdi et al., 2022;
https://github.com/SayBender/Nemo, https://www.mdpi.com/2072-4292/14/16/3979), over ALERTWildfire
(Nevada + CA) video frames — leak-clean vs our HPWREN/FIgLib eval.

**Annotations only — the image frames are NOT publicly hosted** (the repo's download URL is the
placeholder `[ADD A URL ONCE DATASET IS PUBLIC]`; the JSONs carry only `file_name`, no `coco_url`).
Files here:
- `nemo_sc_{train,val}.json` — single-class (`smoke`): 2,342 train / 237 val images.
- `nemo_dg_{train,val}.json` — density (low/mid/high): 2,680 train images.

To actually use these for training we'd need the frames — see research/backlog.md (email authors, or
reconstruct from the source ALERTWildfire timelapse videos named in each `file_name`). The Fuego
fallback the Nemo README offers is HPWREN-sourced and therefore leaky against our eval.
