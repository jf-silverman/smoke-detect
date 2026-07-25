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

# 3) UPLOAD DATA (run from repo root; data/ is gitignored, so push it explicitly).
#    Stage data/processed as ONE tarball -- copying 33k files individually is ~40 min of
#    per-object API calls and loses the race with spot preemption; one object is seconds.
#    --no-mac-metadata --no-xattrs strips the Apple xattr header that GNU tar on the VM would
#    otherwise warn about once per file (that warning spam throttles untar via the serial console).
#    Exclude the Mac-path labels.cache (the VM regenerates it) and the unneeded proof features.
# tar -C data/processed --no-mac-metadata --no-xattrs \
#   --exclude labels.cache --exclude features_proof.npz -cf /tmp/data_processed.tar .
# gcloud storage cp /tmp/data_processed.tar "$BUCKET/data_processed.tar"
# gcloud storage rsync -r data/figlib "$BUCKET/raw/figlib"   # TTD only; skip for the grouped run

# 4) LAUNCH THE TRAINING VM (Deep Learning image ships CUDA + PyTorch)
#    --scopes=cloud-platform lets the VM's service account read/write the bucket (required).
#    --max-run-duration is a billing backstop -- the VM self-deletes after this even if you forget
#    (budget alerts do NOT stop resources). Bump it above your expected wall-clock with margin.
# gcloud compute instances create "$VM" \
#   --zone="$ZONE" --machine-type="$MACHINE" \
#   --accelerator="type=$GPU,count=1" --maintenance-policy=TERMINATE \
#   --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
#   --boot-disk-size=200GB \
#   --scopes=https://www.googleapis.com/auth/cloud-platform \
#   --max-run-duration=28800s --instance-termination-action=DELETE \
#   --metadata="install-nvidia-driver=True,bucket=$BUCKET" \
#   --metadata-from-file=startup-script=infra/train_vm_startup.sh
#
#    CHEAP PREEMPTIBLE VARIANT (~half price): add these to the command above --
#      --provisioning-model=SPOT --instance-termination-action=STOP
#    Use STOP, not DELETE: on preemption the VM STOPS and its disk (repo + hydrated data +
#    checkpoints) survives. train_vm_startup.sh is idempotent and train.py --resume continues from
#    the last checkpoint, so `gcloud compute instances start "$VM"` resumes in seconds. (Spot
#    forbids --automaticRestart, so the restart is manual or via an external scheduler.)
#    NOTE: a STOPPED spot VM bills $0 for GPU/compute but its 200GB disk keeps billing -- still
#    DELETE it when the run is truly done. Do NOT combine SPOT with --max-run-duration=...DELETE
#    above; pick one termination action.

# 5) WATCH / FETCH / TEARDOWN
# gcloud compute ssh "$VM" --zone="$ZONE" --command='tail -f /var/log/syslog | grep startup-script'
# gcloud compute instances start "$VM" --zone="$ZONE"    # resume a STOP'd spot VM (picks up mid-run)
# gcloud storage rsync -r "$BUCKET/runs" runs            # pull results back locally
# gcloud compute instances delete "$VM" --zone="$ZONE"   # ALWAYS delete when done (disk bills while stopped)
