#!/usr/bin/env bash
# GCP setup for smoke-detect: bucket + training VM. Edit the vars, then run the numbered steps.
# Auth is interactive -- run `gcloud auth login` yourself first, and set the project.
#
#   gcloud auth login
#   PROJECT=my-project REGION=us-central1 bash infra/gcp_setup.sh
#
# Steps 2-4 are commented so nothing fires by accident -- uncomment as you go.
set -euo pipefail

# ---- edit these (or pass as env vars) ----
PROJECT="${PROJECT:?set PROJECT=your-gcp-project-id}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
BUCKET="${BUCKET:-gs://smoke-detect-${PROJECT}}"
GPU="${GPU:-nvidia-l4}"               # nvidia-l4 (fast, on-demand) | nvidia-tesla-t4 (cheap, spot-able)
MACHINE="${MACHINE:-g2-standard-4}"   # g2-standard-4 for L4 | n1-standard-8 for T4
VM="${VM:-smoke-train}"

gcloud config set project "$PROJECT"

# 1) CHECK GPU QUOTA (new projects start at 0; a request can take hours -- do this first)
echo "== GPU quota in $REGION =="
gcloud compute regions describe "$REGION" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -iE "gpu|nvidia" || true
echo "If the limit is 0, request an increase: https://console.cloud.google.com/iam-admin/quotas"

# 2) CREATE THE BUCKET (same region as compute => no egress)
# gcloud storage buckets create "$BUCKET" --location="$REGION" --uniform-bucket-level-access

# 3) UPLOAD DATA (run from repo root; data/ is gitignored, so push it explicitly)
# gcloud storage rsync -r data/processed "$BUCKET/processed"
# gcloud storage rsync -r data/figlib    "$BUCKET/raw/figlib"

# 4) LAUNCH THE TRAINING VM (Deep Learning image ships CUDA + PyTorch)
#    --scopes=cloud-platform lets the VM's service account read/write the bucket (required).
#    For a cheap preemptible run add:  --provisioning-model=SPOT
#    (safe -- train_vm_startup.sh mirrors runs/ to GCS every 5 min, so preemption loses <= 1 epoch)
# gcloud compute instances create "$VM" \
#   --zone="$ZONE" --machine-type="$MACHINE" \
#   --accelerator="type=$GPU,count=1" --maintenance-policy=TERMINATE \
#   --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
#   --boot-disk-size=200GB \
#   --scopes=https://www.googleapis.com/auth/cloud-platform \
#   --metadata="install-nvidia-driver=True,bucket=$BUCKET" \
#   --metadata-from-file=startup-script=infra/train_vm_startup.sh

# 5) WATCH / FETCH / TEARDOWN
# gcloud compute ssh "$VM" --zone="$ZONE" --command='tail -f /var/log/syslog | grep startup-script'
# gcloud storage rsync -r "$BUCKET/runs" runs      # pull results back locally
# gcloud compute instances delete "$VM" --zone="$ZONE"   # storage persists at the cheap rate
