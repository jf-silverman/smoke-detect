"""Cache a per-frame embedding for every pyro-sdis image.

The temporal model does not re-run the CNN each epoch. Instead we pass the whole
corpus through the *frozen* baseline detector once, take Ultralytics' 256-d
embedding for each frame, and cache it. The temporal head then trains on sequences
of these vectors -- cheap enough to iterate on MPS in minutes.

Using the BASELINE detector (grouped_proof) as the extractor, not the hard-neg one,
is deliberate: it makes "temporal vs. single-frame" a clean comparison over the same
features, so any gain is attributable to temporal context and nothing else.

    python src/models/extract_features.py --weights runs/grouped_proof/weights/best.pt

Output: data/processed/features/<image_stem>.npy is avoided -- instead one packed
archive data/processed/features_<tag>.npz mapping stem -> 256-float vector, plus the
per-frame confidence the baseline emits (a useful scalar feature in its own right).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def all_images() -> list[Path]:
    """Real frames only. Hard-negative mining left oversampling symlinks
    (`<stem>__hn{k}.jpg`) in this directory; those carry no real timestamp and
    would corrupt sequence building, so we skip them."""
    img_dir = PROC / "images"
    return sorted(p for p in img_dir.glob("*.jpg") if not p.is_symlink())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(ROOT / "runs/grouped_proof/weights/best.pt"))
    ap.add_argument("--tag", default="proof", help="names the output archive")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    model = YOLO(args.weights)
    images = all_images()
    print(f"extracting {len(images)} embeddings on {device} from {args.weights}")

    stems: list[str] = []
    feats = np.zeros((len(images), 256), dtype=np.float32)
    confs = np.zeros(len(images), dtype=np.float32)

    B = args.batch
    for i in range(0, len(images), B):
        batch = [str(p) for p in images[i : i + B]]
        # embeddings: 256-d backbone descriptor per frame
        emb = model.embed(batch, device=device, imgsz=args.imgsz, verbose=False)
        # detections: keep the single-frame max confidence as a cheap scalar feature
        det = model.predict(batch, device=device, imgsz=args.imgsz, conf=0.001, verbose=False)
        for k in range(len(batch)):
            feats[i + k] = emb[k].detach().cpu().numpy().astype(np.float32)
            c = det[k].boxes.conf
            confs[i + k] = float(c.max()) if len(c) else 0.0
            stems.append(images[i + k].stem)
        if (i // B) % 20 == 0:
            print(f"  {i + len(batch)}/{len(images)}", flush=True)

    out = PROC / f"features_{args.tag}.npz"
    np.savez_compressed(out, stems=np.array(stems), feats=feats, confs=confs)
    print(f"saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
