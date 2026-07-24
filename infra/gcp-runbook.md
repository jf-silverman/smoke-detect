# GCP runbook — training on a cloud GPU

Executable companion to [../reports/gcp-plan.md](../reports/gcp-plan.md) (strategy + costs). Scripts
here; run them from the repo root. Auth and resource creation happen on **your** account.

## Order of operations

1. **Auth + project** (interactive — you run this):
   ```
   gcloud auth login
   gcloud config set project <PROJECT_ID>
   ```
2. **Check GPU quota** — the long pole. A fresh project starts at 0; an increase can take hours.
   ```
   PROJECT=<PROJECT_ID> bash infra/gcp_setup.sh          # step 1 prints your GPU quota
   ```
   If the L4/T4 limit is 0, request an increase at the console Quotas page (link in the script
   output) before doing anything else.
3. **Bucket + data** — uncomment steps 2–3 in `gcp_setup.sh` and run. `data/` is gitignored, so it
   is pushed to GCS explicitly. FIgLib-full (~25 GB) can also land straight in `raw/figlib`.
4. **Launch the VM** — uncomment step 4. Defaults: L4 / `g2-standard-4` / `us-central1-a`, 200 GB
   disk, Deep Learning image. Add `--provisioning-model=SPOT` for the cheap preemptible path — safe,
   because `train_vm_startup.sh` mirrors `runs/` to GCS every 5 minutes.
5. **Watch / fetch / delete** — step 5 has the ssh-tail, the results-pull, and the VM delete.
   Delete the VM when done; the bucket persists at ~cents/GB/month.

## Files

- `gcp_setup.sh` — config vars + the numbered gcloud commands (bucket, VM, teardown).
- `train_vm_startup.sh` — runs on the VM at boot: clones the repo, hydrates `data/` from GCS, trains
  (edit the `train.py` flags for the experiment), and syncs `runs/` back with a `DONE.txt` marker.

## Cost reminder

Storage a few $/month; a full 40-epoch run ~$3–6 (T4 spot) to ~$9–17 (L4 on-demand); a new-account
$300 trial covers it. Set a budget alert. See [gcp-plan.md](../reports/gcp-plan.md).
