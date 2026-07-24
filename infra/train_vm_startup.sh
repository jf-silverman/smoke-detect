#!/usr/bin/env bash
# VM startup script: pull repo + data from GCS, train, and mirror runs/ back to GCS every 5 min
# so a spot preemption loses at most one epoch. BUCKET comes from instance metadata.
# Edit the train.py flags for the experiment you want (defaults: full-scale 1280 baseline).
set -euo pipefail

BUCKET=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket")

cd /root
if [ -d smoke-detect ]; then (cd smoke-detect && git pull); else
  git clone https://github.com/jf-silverman/smoke-detect.git; fi
cd smoke-detect
pip install -q --upgrade ultralytics
# Ultralytics imports cv2, which needs libGL.so.1 + libglib -- not present on this image, so
# the import crashes on a headless server. Install the system libs before training.
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0

# data/ is gitignored -> hydrate from GCS. The dataset is staged as a single tarball
# (data_processed.tar): pulling one object + untar is seconds, vs ~40 min copying 33k files
# one-by-one (each a separate API call), which kept losing the race with spot preemption.
mkdir -p data/processed
gcloud storage cp "$BUCKET/data_processed.tar" /root/data_processed.tar
# The tarball is built on macOS, so every file carries an Apple xattr header that GNU tar
# warns about once per file. Those warnings go to the serial console (a slow device) and
# throttle extraction badly -- silence them so untar runs at disk speed (seconds, not minutes).
tar -C data/processed --warning=no-unknown-keyword -xf /root/data_processed.tar 2>/dev/null
rm -f /root/data_processed.tar
# figlib (TTD only) is not needed for this training run -> skip it.

# the split manifests + yamls were generated on the author's Mac and carry absolute local
# paths -- rewrite them to this VM's checkout so YOLO can find the images.
find data/processed \( -name '*.yaml' -o -name '*.txt' \) -print0 \
  | xargs -0 -r sed -i "s#/Users/jfs-m3/smoke_detect#$(pwd)#g"

# preemption-safe checkpointing: mirror runs/ to GCS every 5 min in the background
( while true; do gcloud storage rsync -r runs "$BUCKET/runs" >/dev/null 2>&1 || true; sleep 300; done ) &

# --- the experiment (edit me) ---
# L4 (24GB) fits a larger batch at 1280 than the local MPS box (which used batch 8).
# --patience 8 early-stops after 8 epochs with no val improvement; the local run plateaued
# ~epoch 24, so this typically ends in the low-mid 20s. --epochs 30 caps the worst case.
# On-demand (STANDARD) so it runs uninterrupted -- costs money the whole time, so let early
# stopping end it rather than burning epochs past the plateau.
python3 src/models/train.py --split grouped --imgsz 1280 --batch 16 --epochs 30 --patience 8 --name gcp_grouped_1280

# final sync + a done-marker so you can poll from your laptop
gcloud storage rsync -r runs "$BUCKET/runs"
echo "done $(date -u)" | gcloud storage cp - "$BUCKET/runs/DONE.txt"
