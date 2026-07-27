# GCP plan — cheap cloud storage + optional GPU compute

Purpose: two problems this could solve, both cheaply. (1) **Storage** — PYRONEAR-2025's full
image+video set (~60–120 GB) strains the 87 GB local headroom; FIgLib-full (~20–25 GB) does not.
(2) **Compute** — local training on the M3 Pro (MPS) runs ~92 min/epoch; a real CUDA GPU should
be ~3–6× faster, ending the multi-day full-scale runs. Data sizing lives in
[backlog.md → Pre-scoping #3](backlog.md#pre-scoping-3--hpwren--time-to-detection).

## Cost (approximate, us-central1, 2026 — verify at provision time)

| item | rate | this project |
|---|---|---|
| GCS Standard storage | ~$0.020/GB/month | FIgLib ~$0.50/mo; +PYRONEAR ~$2.50/mo total |
| Egress (data leaving GCP) | ~$0.12/GB | $0 if compute is also on GCP — keep data + training co-located |
| T4 GPU (n1, **spot**) | ~$0.15–0.25/hr all-in | a full 40-epoch run ~$3–6 (preemptible; checkpoint) |
| L4 GPU (g2-standard-4, on-demand only) | ~$0.85/hr all-in | ~$9–17 per full run |
| A100-40GB | ~$3.7/hr (less on spot) | ~$12–22 per full run (fastest) |

**A new GCP account gets a $300 free-trial credit**, which covers this entire experiment
(storage for months + several training runs + the TTD work) at $0 out of pocket. Storage is
negligible regardless; the real value is compute speed. Source: [GCP GPU pricing](https://cloud.google.com/products/compute/gpus-pricing).

## Storage design

One bucket, region matched to the compute VM (avoid cross-region egress):

```
gs://smoke-detect-<suffix>/
  raw/figlib/         # FIgLib-full pull (WIFIRE Commons)         ~20–25 GB
  raw/pyronear2025/   # PYRONEAR-2025 images + videos             ~60–120 GB
  processed/          # YOLO-format splits (mirrors data/processed)
  runs/               # training outputs (weights, results.csv)   synced back per run
```

- **Standard** storage class for active data; move `raw/` to **Nearline** (~$0.010/GB/mo) once
  a dataset is processed and rarely re-read.
- Set a **budget alert** (e.g. $20) and, on any GPU VM, an **idle auto-shutdown** — the classic
  way to waste cloud money is a forgotten running GPU, not storage.

## Compute — spot GPU VM with checkpoint/resume

Ultralytics writes a checkpoint each epoch and supports `resume=True`, so a **spot/preemptible**
VM (60–91% cheaper, ~30 s preemption notice) is safe if checkpoints sync to GCS:

1. Create project; enable Compute Engine + set a budget alert.
2. **Request GPU quota** (`GPUs (all regions)` ≥ 1) — new accounts start at 0; approval can take
   hours to a day. *Do this first — it's the usual blocker.*
3. Spot VM with the GPU (T4 for cheapest, L4 for speed) + Deep Learning VM image (CUDA/PyTorch
   preinstalled).
4. Startup: `git clone` the repo, `gsutil -m rsync` the data from `raw/`+`processed/`, then
   `train.py ... --resume` writing checkpoints to a local dir that a cron `gsutil rsync`es to
   `gs://.../runs/` each epoch (so a preemption loses at most one epoch).
5. On completion: sync `runs/` back to GCS (and locally if wanted), then **delete the VM**
   (storage persists at the cheap rate).

For a one-command path, Vertex AI custom training jobs also work, but a plain spot Compute Engine
VM is the least-abstraction, cheapest option for this scale.

## Recommendation / sequencing

- **Do not migrate the current combined-levers run** — it's ~14 h into a local run; finishing
  locally is sunk-cost-cheapest.
- **Phase A (TTD)** needs no cloud — it runs on cached confidences locally
  ([`figlib_ttd.py`](../src/models/figlib_ttd.py)).
- **Phase B (data):** FIgLib-full fits locally (~25 GB); pull it straight to `data/`. Only reach
  for GCS when adding **PYRONEAR-2025** (the video set is what overflows local disk).
- **Phase C (learned temporal / any full-scale retrain):** this is where a spot GPU VM pays for
  itself — hours instead of days, single-digit dollars, or free on the trial credit.

## Works Cited

- Google Cloud. (2026). *GPU pricing* [Pricing page]. https://cloud.google.com/products/compute/gpus-pricing
