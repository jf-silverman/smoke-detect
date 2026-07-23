"""Mine hard negatives from the baseline, then build an augmented training split.

Hard negative = a frame with NO smoke that the model nonetheless fires on. These are
the clouds, fog, marine layer, and glare responsible for the baseline's 42%
false-alarm rate on clean frames. Emphasizing them in a retrain is the direct,
literature-backed attack on that number (cf. SKLFS separable negative sampling / OHEM).

Two rules keep this leak-safe:

  1. Mine ONLY the training sites. Mining val/test negatives and adding them to
     training would leak the evaluation. We read the grouped_train manifest and
     restrict to its negatives.
  2. Keep the positive/easy-negative budget comparable to the baseline. The baseline
     trained on a random ~20% of grouped_train, so we build a fixed, seeded 20% base
     sample and add the mined hard negatives on top of it. That isolates the effect of
     the hard negatives rather than confounding it with "more data".

Output:
  results/hard_negatives.csv                     ranked list of mined frames
  data/processed/splits/grouped_hardneg_*.txt    augmented train, unchanged val/test
  data/processed/grouped_hardneg.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def manifest(split: str, part: str) -> list[Path]:
    return [Path(p) for p in (PROC / "splits" / f"{split}_{part}.txt").read_text().split() if p]


def is_negative(img: Path) -> bool:
    lbl = PROC / "labels" / f"{img.stem}.txt"
    return not (lbl.exists() and lbl.read_text().strip())


def max_conf(model: YOLO, images: list[Path], device: str, imgsz: int) -> np.ndarray:
    scores = np.zeros(len(images))
    B = 64
    for i in range(0, len(images), B):
        res = model.predict([str(p) for p in images[i : i + B]],
                            device=device, imgsz=imgsz, conf=0.001, verbose=False)
        for k, r in enumerate(res):
            c = r.boxes.conf
            scores[i + k] = float(c.max()) if len(c) else 0.0
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(ROOT / "runs/grouped_proof/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.05, help="alarm threshold that defines 'hard'")
    ap.add_argument("--oversample", type=int, default=3, help="copies of each hard negative")
    ap.add_argument("--base-frac", type=float, default=0.2, help="fixed base sample of grouped_train")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output split name "
                    "(e.g. '_1280full' -> grouped_hardneg_1280full); keeps proof artifacts intact")
    args = ap.parse_args()
    split_name = f"grouped_hardneg{args.tag}"

    device = pick_device()
    RESULTS.mkdir(exist_ok=True)

    train = manifest("grouped", "train")
    negatives = [p for p in train if is_negative(p)]
    print(f"grouped_train: {len(train)} images, {len(negatives)} negatives to mine", flush=True)

    model = YOLO(args.weights)
    scores = max_conf(model, negatives, device, args.imgsz)

    order = np.argsort(-scores)
    hard = [(negatives[i], scores[i]) for i in order if scores[i] >= args.conf]
    print(f"hard negatives (max_conf >= {args.conf}): {len(hard)} "
          f"({len(hard) / len(negatives) * 100:.1f}% of clean training frames)")

    with (RESULTS / "hard_negatives.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "site", "max_conf"])
        for p, s in hard:
            import re
            site = re.sub(r"-\d+$", "", re.sub(r"^[^_]+_", "", p.stem).rsplit("_", 1)[0])
            w.writerow([p.name, site, round(float(s), 4)])

    # fixed seeded base sample of the full train, matching the baseline's ~20% budget
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(train), size=int(len(train) * args.base_frac), replace=False)
    base = [train[i] for i in sorted(idx)]

    # oversample hard negatives via distinct symlinks so YOLO counts them as separate
    # dataset entries (label lookup replaces /images/->/labels/, .jpg->.txt, so the
    # duplicate needs a matching empty label file alongside it)
    dup_lines: list[str] = []
    img_dir, lbl_dir = PROC / "images", PROC / "labels"
    for p, _ in hard:
        for k in range(1, args.oversample + 1):
            dup = img_dir / f"{p.stem}__hn{k}.jpg"
            dlbl = lbl_dir / f"{p.stem}__hn{k}.txt"
            if not dup.exists():
                dup.symlink_to(p.resolve())
            if not dlbl.exists():
                dlbl.write_text("")  # empty == negative
            dup_lines.append(str(dup.resolve()))

    aug_train = [str(p.resolve()) for p in base] + dup_lines
    split_dir = PROC / "splits"
    (split_dir / f"{split_name}_train.txt").write_text("\n".join(aug_train) + "\n")
    # val/test identical to grouped -> same held-out sites, clean comparison
    for part in ("val", "test"):
        (split_dir / f"{split_name}_{part}.txt").write_text(
            (split_dir / f"grouped_{part}.txt").read_text())

    (PROC / f"{split_name}.yaml").write_text(
        f"# grouped split + {len(hard)} mined hard negatives x{args.oversample}\n"
        f"path: {PROC.resolve()}\n"
        f"train: splits/{split_name}_train.txt\n"
        f"val: splits/{split_name}_val.txt\n"
        f"test: splits/{split_name}_test.txt\n"
        f"nc: 1\nnames:\n  0: smoke\n")

    print(f"\naugmented train: {len(base)} base ({args.base_frac:.0%} sample) + "
          f"{len(hard)}x{args.oversample}={len(dup_lines)} hard-neg copies = {len(aug_train)} images")
    print(f"wrote {split_name}.yaml + manifests; ranked list -> {RESULTS / 'hard_negatives.csv'}")


if __name__ == "__main__":
    main()
