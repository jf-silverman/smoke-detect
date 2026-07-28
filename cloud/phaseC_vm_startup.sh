#!/usr/bin/env bash
# GCP VM startup script for Phase C training (runs as root on a Deep Learning GPU VM).
#
# Flow: pull the data bundle + base weights from GCS -> fine-tune at 1280 on the corrected day-only
# FIgLib set -> sync the run back to GCS -> write a DONE marker -> stop the VM (halts GPU billing).
# The VM is created with --instance-termination-action, and you delete it after pulling results
# (see cloud/phaseC_runbook.md). All output is on the serial console and in /var/log/syslog.
set -euxo pipefail

BUCKET="gs://smoke-detect-jfs"
WORK="/opt/phaseC"
RUN="figlib_phaseC"

mkdir -p "$WORK" && cd "$WORK"

# 1) data bundle + base weights from GCS (co-located in US-CENTRAL1 -> no egress)
gsutil cp "$BUCKET/phaseC/phaseC_bundle.tar.gz" .
tar xzf phaseC_bundle.tar.gz                       # -> images/train, labels/train
gsutil cp "$BUCKET/runs/gcp_grouped_1280/weights/best.pt" base.pt

# 2) dataset yaml with the VM-absolute path (Ultralytics resolves a relative path against ~/datasets)
cat > data.yaml <<YAML
path: $WORK
train: images/train
val: images/train
names:
  0: smoke
YAML

# 3) deps (DLVM ships torch+CUDA; add ultralytics)
pip install -q ultralytics

# 4) fine-tune. device=0 = the T4. batch 8 is safe on a 16 GB T4 at 1280; raise to 16 if you prefer.
yolo detect train \
  model="$WORK/base.pt" \
  data="$WORK/data.yaml" \
  imgsz=1280 epochs=40 batch=8 device=0 workers=8 patience=15 \
  project="$WORK/runs" name="$RUN" exist_ok=True

# 5) sync the whole run (weights + results.csv + plots) back to GCS, then flag completion
gsutil -m rsync -r "$WORK/runs/$RUN" "$BUCKET/runs/$RUN"
echo "phaseC complete $(date -u +%FT%TZ)" | gsutil cp - "$BUCKET/runs/$RUN/_TRAINING_COMPLETE"

# 6) stop the VM -> GPU billing ends. (Disk persists at ~pennies until you delete the VM.)
shutdown -h now
