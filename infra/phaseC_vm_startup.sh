#!/usr/bin/env bash
# GCP VM startup script for Phase C training (runs as root on a Deep Learning GPU VM).
#
# Flow: pull the data bundle + base weights from GCS -> fine-tune at 1280 on the corrected day-only
# FIgLib set -> sync the run back to GCS -> write a DONE marker -> stop the VM (halts GPU billing).
# You delete the VM after pulling results (see infra/phaseC-runbook.md). Output is on the serial
# console and, for training, in a logfile synced to GCS (see the redirect note below).
#
# Reuses the hard-won fixes from infra/train_vm_startup.sh (the July 24 base-model run):
#   - install libGL/libglib (Ultralytics' cv2 import crashes without it on a headless image)
#   - redirect training output to a logfile (Ultralytics progress bars overflow the GCE metadata
#     script-runner, which then stops draining stdout and training HANGS mid-run)
#   - silence tar's per-file xattr warnings (they throttle extraction badly over the serial console)
set -euxo pipefail

BUCKET="gs://smoke-detect-jfs"
WORK="/opt/phaseC"
RUN="figlib_phaseC"
LOG="$WORK/train.log"

# Safety net: on ANY failure, push whatever log exists to GCS and STOP the VM, so a crash never
# leaves the GPU billing and you can still read why it died. (Normal success path stops at the end.)
finish_fail() { rc=$?; echo "STARTUP FAILED (exit $rc) at line ${BASH_LINENO[0]}";
  gsutil cp "$LOG" "$BUCKET/runs/$RUN/train.log" 2>/dev/null || true; sleep 5; shutdown -h now; }
trap finish_fail ERR

mkdir -p "$WORK" && cd "$WORK"

# 1) system dep for Ultralytics' cv2 (libGL.so.1) -- no-op if already present
apt-get install -y -qq libgl1 libglib2.0-0 2>/dev/null \
  || { apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0; }

# 2) data bundle + base weights from GCS (co-located -> no egress). The bundle is macOS-built, so
#    silence GNU tar's per-file Apple-xattr warnings that otherwise throttle extraction.
gsutil cp "$BUCKET/phaseC/phaseC_bundle.tar.gz" .
tar --warning=no-unknown-keyword -xzf phaseC_bundle.tar.gz 2>/dev/null   # -> images/train, labels/train
gsutil cp "$BUCKET/runs/gcp_grouped_1280/weights/best.pt" base.pt

# 3) dataset yaml with the VM-absolute path (Ultralytics resolves a relative path against ~/datasets)
cat > data.yaml <<YAML
path: $WORK
train: images/train
val: images/train
names:
  0: smoke
YAML

# 4) deps (DLVM ships torch+CUDA; add ultralytics)
python3 -c "import ultralytics" 2>/dev/null || pip install -q ultralytics

# 5) fine-tune. device=0 = the T4. batch 8 is safe on a 16 GB T4 at 1280; raise to 16 if you prefer.
#    CRITICAL: redirect to a logfile -- Ultralytics' \r-heavy progress bars overflow the GCE
#    metadata script-runner, which then stops draining stdout and training BLOCKS on write().
mkdir -p "$WORK/runs/$RUN"
yolo detect train \
  model="$WORK/base.pt" \
  data="$WORK/data.yaml" \
  imgsz=1280 epochs=40 batch=8 device=0 workers=8 patience=15 \
  project="$WORK/runs" name="$RUN" exist_ok=True > "$LOG" 2>&1

# 6) sync the run (weights + results.csv + plots + log) back to GCS, then flag completion
cp "$LOG" "$WORK/runs/$RUN/train.log" 2>/dev/null || true
gsutil -m rsync -r "$WORK/runs/$RUN" "$BUCKET/runs/$RUN"
echo "phaseC complete $(date -u +%FT%TZ)" | gsutil cp - "$BUCKET/runs/$RUN/_TRAINING_COMPLETE"

# 7) stop the VM -> GPU billing ends. (Disk persists at ~pennies until you delete the VM.)
shutdown -h now
