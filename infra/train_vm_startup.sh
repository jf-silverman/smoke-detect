#!/usr/bin/env bash
# VM startup script: pull repo + data from GCS, train, and mirror runs/ back to GCS every 5 min
# so a spot preemption loses at most one epoch. BUCKET comes from instance metadata.
# Edit the train.py flags for the experiment you want (defaults: full-scale 1280 baseline).
#
# IDEMPOTENT: every step skips its work if already done, so on a spot VM launched with
# --instance-termination-action=STOP, a preemption+restart (`gcloud compute instances start`)
# reuses the persisted disk -- deps + data are already there and training RESUMES from the last
# checkpoint (train.py --resume) rather than rebuilding and restarting from epoch 1.
set -euo pipefail

BUCKET=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket")

cd /root
if [ -d smoke-detect ]; then (cd smoke-detect && git pull); else
  git clone https://github.com/jf-silverman/smoke-detect.git; fi
cd smoke-detect
# deps: skip if ultralytics already importable (fast on a resumed disk).
python3 -c "import ultralytics" 2>/dev/null || pip install -q --upgrade ultralytics
# Ultralytics imports cv2, which needs libGL.so.1 + libglib -- not present on this image, so
# the import crashes on a headless server. apt-get install is a no-op if already present.
apt-get install -y -qq libgl1 libglib2.0-0 2>/dev/null \
  || { apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0; }

# data/ is gitignored -> hydrate from GCS, ONCE. The .hydrated sentinel is written only after a
# successful untar + path-rewrite, so a resumed disk skips this whole block. The dataset is
# staged as a single tarball (data_processed.tar): pulling one object + untar is seconds, vs
# ~40 min copying 33k files one-by-one (each a separate API call).
if [ ! -f data/processed/.hydrated ]; then
  mkdir -p data/processed
  gcloud storage cp "$BUCKET/data_processed.tar" /root/data_processed.tar
  # The tarball is built on macOS, so every file carries an Apple xattr header that GNU tar
  # warns about once per file. Those warnings go to the serial console (a slow device) and
  # throttle extraction badly -- silence them so untar runs at disk speed (seconds, not minutes).
  tar -C data/processed --warning=no-unknown-keyword -xf /root/data_processed.tar 2>/dev/null
  rm -f /root/data_processed.tar
  # the split manifests + yamls were generated on the author's Mac and carry absolute local
  # paths -- rewrite them to this VM's checkout so YOLO can find the images.
  find data/processed \( -name '*.yaml' -o -name '*.txt' \) -print0 \
    | xargs -0 -r sed -i "s#/Users/jfs-m3/smoke_detect#$(pwd)#g"
  touch data/processed/.hydrated
fi
# figlib (TTD only) is not needed for this training run -> skip it.

# preemption-safe checkpointing: mirror runs/ to GCS every 5 min in the background
( while true; do gcloud storage rsync -r runs "$BUCKET/runs" >/dev/null 2>&1 || true; sleep 300; done ) &

# --- the experiment (edit me) ---
# L4 (24GB) fits a larger batch at 1280 than the local MPS box (which used batch 8).
# --patience 8 early-stops after 8 epochs with no val improvement; the local run plateaued
# ~epoch 24, so this typically ends in the low-mid 20s. --epochs 30 caps the worst case.
# --resume continues from runs/gcp_grouped_1280/weights/last.pt if a prior boot left one (spot
# STOP+restart); it's a no-op on the first boot. Let early stopping end it near the plateau.
#
# CRITICAL: redirect training output to a logfile. Ultralytics' progress bars (ANSI + \r, few
# newlines) overflow the GCE metadata script-runner's line scanner ("token too long"); the
# runner then stops draining stdout, the OS pipe fills, and training BLOCKS on write() and
# hangs mid-run. Writing to a file keeps the startup script's own stdout quiet. The log lands
# under runs/ so the 5-min rsync mirrors it to GCS -- tail it with:
#   gcloud storage cat gs://$BUCKET/runs/train.log | tail
mkdir -p runs
python3 src/models/train.py --split grouped --imgsz 1280 --batch 16 --epochs 30 --patience 8 --resume --name gcp_grouped_1280 > runs/train.log 2>&1

# final sync + a done-marker so you can poll from your laptop
gcloud storage rsync -r runs "$BUCKET/runs"
echo "done $(date -u)" | gcloud storage cp - "$BUCKET/runs/DONE.txt"
