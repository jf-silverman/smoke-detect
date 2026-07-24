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

# data/ is gitignored -> hydrate from GCS
mkdir -p data
gcloud storage rsync -r "$BUCKET/processed" data/processed
gcloud storage rsync -r "$BUCKET/raw/figlib" data/figlib || true

# preemption-safe checkpointing: mirror runs/ to GCS every 5 min in the background
( while true; do gcloud storage rsync -r runs "$BUCKET/runs" >/dev/null 2>&1 || true; sleep 300; done ) &

# --- the experiment (edit me) ---
# L4 (24GB) fits a larger batch at 1280 than the local MPS box (which used batch 8)
python3 src/models/train.py --split grouped --imgsz 1280 --batch 16 --epochs 40 --name gcp_grouped_1280

# final sync + a done-marker so you can poll from your laptop
gcloud storage rsync -r runs "$BUCKET/runs"
echo "done $(date -u)" | gcloud storage cp - "$BUCKET/runs/DONE.txt"
