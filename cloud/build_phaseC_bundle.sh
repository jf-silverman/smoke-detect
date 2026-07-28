#!/usr/bin/env bash
# Build the self-contained Phase C training bundle for GCP.
#
# Packs the corrected day-only dataset (real image files + labels, symlinks dereferenced) into a
# single tar.gz for upload to GCS. The base weights are NOT bundled -- they already live in the
# bucket at runs/gcp_grouped_1280/weights/best.pt and the VM pulls them from there.
#
# Usage:  bash cloud/build_phaseC_bundle.sh
# Output: data/phaseC_bundle.tar.gz   (gitignored)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/data/figlib_corrected"
OUT="$ROOT/data/phaseC_bundle.tar.gz"

if [ ! -d "$SRC/images/train" ]; then
  echo "corrected set missing; run:  python src/data/build_figlib_corrected.py" >&2
  exit 1
fi

imgs=$(find -L "$SRC/images/train" -name '*.jpg' | wc -l | tr -d ' ')
lbls=$(find "$SRC/labels/train" -name '*.txt' | wc -l | tr -d ' ')
echo "bundling $imgs images + $lbls labels (negatives are the unlabeled images) ..."

# -h dereferences the symlinks so the tarball carries the real jpg bytes, not dangling links.
tar czhf "$OUT" -C "$SRC" images labels

echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
echo "next:  gsutil cp $OUT gs://smoke-detect-jfs/phaseC/"
