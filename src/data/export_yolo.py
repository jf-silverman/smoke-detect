"""Materialize pyro-sdis into an Ultralytics-YOLO layout.

Images and labels are written ONCE:

    data/processed/images/<image_name>.jpg
    data/processed/labels/<image_name>.txt      (empty file == true negative)

A split is then just a manifest -- a text file listing image paths -- plus a
data.yaml pointing at the manifests:

    data/processed/splits/<split_name>_{train,val,test}.txt
    data/processed/<split_name>.yaml

This matters because we evaluate with 8-fold leave-one-site-out. Copying the image
corpus per fold would mean ~26 GB on disk for zero benefit; manifests cost kilobytes,
so all 8 folds plus the random and grouped splits coexist over one copy of the data.

Ultralytics reads a manifest path in the `train:`/`val:` field natively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import grouped_split, loso_folds, prepare, random_split  # noqa: E402

REPO = "pyronear/pyro-sdis"
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def to_yolo(ann: str) -> str:
    """Normalize a pyro-sdis annotation string to Ultralytics class indices.

    pyro-sdis ships its single smoke class with class id `1`, but Ultralytics
    expects zero-based indices: with `nc: 1` the only legal id is `0`, and a `1`
    is silently dropped or errors as out-of-range. Every box in the corpus is
    affected, so this remap is load-bearing -- without it the model trains on no
    positives at all.
    """
    lines = []
    for line in ann.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        lines.append(" ".join(["0", *parts[1:]]))
    return "\n".join(lines) + ("\n" if lines else "")


def materialize(force: bool = False) -> pd.DataFrame:
    """Write every image + label to disk once. Returns the metadata frame."""
    img_dir, lbl_dir = PROC / "images", PROC / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = sorted(
        f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")
    )

    meta = []
    written = skipped = 0
    for i, f in enumerate(files, 1):
        path = hf_hub_download(REPO, f, repo_type="dataset")
        table = pq.read_table(path)
        cols = table.column_names
        print(f"  [{i}/{len(files)}] {f}: {table.num_rows} rows", flush=True)

        for batch in table.to_batches(max_chunksize=512):
            d = batch.to_pydict()
            for j in range(len(d["image_name"])):
                name = Path(d["image_name"][j]).stem
                ann = (d["annotations"][j] or "").strip()
                meta.append(
                    {
                        "image_name": d["image_name"][j],
                        "camera": d["camera"][j],
                        "partner": d["partner"][j],
                        "date": d["date"][j],
                        "annotations": ann,
                    }
                )

                jpg = img_dir / f"{name}.jpg"
                txt = lbl_dir / f"{name}.txt"
                if jpg.exists() and txt.exists() and not force:
                    skipped += 1
                    continue

                # HF gives {'bytes':..., 'path':...} for Image features when not decoded
                img = d["image"][j]
                raw = img["bytes"] if isinstance(img, dict) else img
                jpg.write_bytes(raw)
                # empty label file is meaningful: YOLO reads it as "no objects here"
                txt.write_text(to_yolo(ann))
                written += 1

        del table

    print(f"  images written: {written}, already present: {skipped}")
    return pd.DataFrame(meta)


def write_manifest(df: pd.DataFrame, name: str) -> Path:
    """Write train/val/test manifests + a data.yaml for one split."""
    split_dir = PROC / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for part in ("train", "val", "test"):
        rows = df[df["split"] == part]
        lines = [
            str((PROC / "images" / f"{Path(n).stem}.jpg").resolve()) for n in rows["image_name"]
        ]
        (split_dir / f"{name}_{part}.txt").write_text("\n".join(lines) + "\n")

    yaml_path = PROC / f"{name}.yaml"
    yaml_path.write_text(
        f"# pyro-sdis :: split '{name}'\n"
        f"path: {PROC.resolve()}\n"
        f"train: splits/{name}_train.txt\n"
        f"val: splits/{name}_val.txt\n"
        f"test: splits/{name}_test.txt\n"
        f"nc: 1\n"
        f"names:\n  0: smoke\n"
    )
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rewrite images even if present")
    ap.add_argument("--skip-images", action="store_true", help="manifests only")
    args = ap.parse_args()

    if args.skip_images:
        cache = PROC / "meta.parquet"
        if not cache.exists():
            sys.exit("no cached meta.parquet -- run once without --skip-images")
        df = pd.read_parquet(cache)
    else:
        print("materializing images + labels ...")
        df = materialize(force=args.force)
        PROC.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROC / "meta.parquet")

    print("\nwriting split manifests ...")
    r, rrep = random_split(df)
    write_manifest(r, "random")
    print(f"  random  (LEAKY): {len(r)} images")

    g, grep_ = grouped_split(df)
    write_manifest(g, "grouped")
    print(f"  grouped (HONEST): held-out sites {sorted(grep_.test_sites)}")

    for site, fold, rep in loso_folds(prepare(df) if "site" not in df else df):
        write_manifest(fold, f"loso_{site}")
        n_test = int(rep.counts.loc["test", "n_images"])
        print(f"  loso_{site}: test={n_test}")

    print(f"\ndone -> {PROC}")


if __name__ == "__main__":
    main()
