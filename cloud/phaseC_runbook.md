# Phase C on GCP — runbook (spot T4 GPU, fine-tune from the pyro-sdis 1280 weights)

Fine-tunes the pyro-sdis @1280 detector on the hand-corrected day-only FIgLib set (17 fires, 452
boxes, 85 background negatives) on a **spot T4** in `us-central1` (co-located with the bucket → no
egress). Everything is scripted so the VM is fire-and-forget: it trains, syncs results to GCS,
signals done, and stops itself. You run each command below.

**Expected:** ~40–90 min wall time, **~$0.30–1.00** (spot T4 ~$0.20–0.35/hr all-in). Far under the
$5.50 the full 33k-image run cost — this is 501 images.

**Prereqs (already true for this account):** authed as `joelfsilverman@gmail.com`, project
`smoke-detect-jfs`, bucket `gs://smoke-detect-jfs` (US-CENTRAL1), GPU quota approved from the last run.
Set a **budget alert** in the console (Billing → Budgets, e.g. $20) if you haven't — the cheap
insurance against a forgotten resource.

---

## 1. Build the data bundle (local)

```bash
bash cloud/build_phaseC_bundle.sh
```
Produces `data/phaseC_bundle.tar.gz` (~350 MB: 501 images + 416 label files; the 85 negatives are the
unlabeled images).

## 2. Upload the bundle to GCS

```bash
gsutil cp data/phaseC_bundle.tar.gz gs://smoke-detect-jfs/phaseC/
```
The base weights are already in the bucket (`runs/gcp_grouped_1280/weights/best.pt`), so nothing else
to upload.

## 3. Create the spot T4 VM (this starts billing)

```bash
gcloud compute instances create figlib-phasec \
  --project=smoke-detect-jfs \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --boot-disk-size=100GB \
  --scopes=storage-rw \
  --metadata=install-nvidia-driver=True \
  --metadata-from-file=startup-script=cloud/phaseC_vm_startup.sh
```
`--instance-termination-action=DELETE` means a *preemption* cleans the VM up automatically (just
re-run this command if that happens — the job is short, so preemption is unlikely). On normal
completion the startup script instead **stops** the VM (halts GPU billing); you delete it in step 6.

## 4. Watch it run

Live serial console (Ctrl-C to stop watching — does not affect the VM):
```bash
gcloud compute instances tail-serial-port-output figlib-phasec --zone=us-central1-a
```
Or poll for the completion marker the VM writes when done:
```bash
gsutil ls gs://smoke-detect-jfs/runs/figlib_phaseC/_TRAINING_COMPLETE   # exists => finished
gcloud compute instances describe figlib-phasec --zone=us-central1-a \
  --format='value(status)'                                             # RUNNING -> TERMINATED (stopped)
```

## 5. Retrieve the trained model + logs (local)

```bash
gsutil -m cp -r gs://smoke-detect-jfs/runs/figlib_phaseC ./runs/
ls runs/figlib_phaseC/weights/best.pt          # the Phase C detector
```

## 6. Tear down (stops all billing)

```bash
gcloud compute instances delete figlib-phasec --zone=us-central1-a --quiet
gcloud compute instances list                  # confirm: no instances left
```

---

## Notes & troubleshooting

- **Cost hygiene:** GPU billing stops when the VM stops (step 4 → TERMINATED) or is deleted (step 6).
  The persistent disk costs ~pennies/day until the VM is deleted — so still do step 6.
- **OOM at 1280?** Unlikely for this nano model on a 16 GB T4, but if the log shows CUDA OOM, lower
  `batch=8` → `batch=4` in `cloud/phaseC_vm_startup.sh` and recreate the VM.
- **Preempted mid-run?** With `=DELETE` the VM is gone; just re-run step 3. (No checkpoint/resume is
  wired because the job is ~1 h; add per-epoch `gsutil rsync` + `resume=True` only if you move to a
  long run.)
- **Epochs:** 40 matches the base recipe; `patience=15` early-stops if it plateaus. For a 501-image
  fine-tune, 20–30 is often enough — edit `epochs=` in the startup script if you prefer.
- **After you have `best.pt`:** hand me the path and I'll wire the TTD/motion evaluation on the 6
  held-out recent-CA fires — the real test of whether Phase C closes the distribution-shift miss.
