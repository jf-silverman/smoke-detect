"""Assemble the Phase C corrected-label training set from the CVAT export.

Takes the "Ultralytics YOLO Detection 1.0" export of the hand-corrected FIgLib pre-annotations
(data/figlib_corrected_boxes.zip) and builds a clean, ready-to-train YOLO dataset at
data/figlib_corrected/:

    images/train/<fire>__<stem>.jpg   (symlinked to the pre-annotation frames; no data copied)
    labels/train/<fire>__<stem>.txt   (the corrected boxes; absent = background negative)
    data.yaml

What it fixes relative to the raw export:
  * NIGHT EXCLUSION -- drops nocturnal sequences (day-only scope; mirrors figlib_preannotate.py and
    figlib_ttd.py). The night fire carried 0 boxes anyway; here we also drop its frames so they do
    not enter training even as backgrounds.
  * PATH REMAP -- the export lists frames under data/images/Train/...; we point at the real frames
    in data/figlib_preannot/images/train/ instead.
  * NEGATIVES PRESERVED -- frames with no box are kept as images with no label file, which
    Ultralytics consumes as background/negative images. Every non-night frame in the export's
    Train.txt is materialized, labeled or not.

Run:  python src/data/build_figlib_corrected.py
"""

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORT_ZIP = ROOT / "data" / "figlib_corrected_boxes.zip"
FRAMES_SRC = ROOT / "data" / "figlib_preannot" / "images" / "train"   # the actual jpg frames
OUT = ROOT / "data" / "figlib_corrected"

# day-only scope: match figlib_preannotate.NIGHT_PATTERN / figlib_ttd.NIGHT_PATTERN
NIGHT_RE = re.compile(r"night|nocturnal", re.I)


def fire_of(stem: str) -> str:
    """`<fire>__<frame>` -> `<fire>` (the sequence directory name)."""
    return stem.split("__", 1)[0]


def main() -> None:
    if not EXPORT_ZIP.exists():
        raise SystemExit(f"missing CVAT export at {EXPORT_ZIP}")
    if not FRAMES_SRC.exists():
        raise SystemExit(f"missing pre-annotation frames at {FRAMES_SRC}")

    img_out = OUT / "images" / "train"
    lbl_out = OUT / "labels" / "train"
    for d in (img_out, lbl_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    zf = zipfile.ZipFile(EXPORT_ZIP)

    # 1) collect the frame list (Train.txt) and the corrected label members
    train_list = next((n for n in zf.namelist() if n.lower().endswith("train.txt")), None)
    if train_list is None:
        raise SystemExit("no Train.txt in the export")
    listed = [Path(line.strip()).stem for line in
              zf.read(train_list).decode().splitlines() if line.strip()]
    label_members = {Path(n).stem: n for n in zf.namelist()
                     if n.lower().endswith(".txt") and "labels/" in n.lower()
                     and not n.lower().endswith("train.txt")}

    # 2) materialize each non-night frame: symlink the image, write the label if it has boxes
    n_pos = n_neg = n_night = n_missing_img = 0
    boxes_total = 0
    fires_kept: set[str] = set()
    for stem in listed:
        if NIGHT_RE.search(stem):
            n_night += 1
            continue
        src_img = FRAMES_SRC / f"{stem}.jpg"
        if not src_img.exists():
            n_missing_img += 1
            print(f"  WARN: listed frame has no source image: {stem}.jpg")
            continue
        # relative symlink so the tree survives the repo being moved
        link = img_out / f"{stem}.jpg"
        link.symlink_to(Path("../../../figlib_preannot/images/train") / f"{stem}.jpg")
        fires_kept.add(fire_of(stem))

        member = label_members.get(stem)
        if member is not None:
            body = zf.read(member).decode()
            lines = [ln for ln in body.splitlines() if ln.strip()]
            if lines:
                (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                n_pos += 1
                boxes_total += len(lines)
                continue
        n_neg += 1  # no label file -> Ultralytics background negative

    # 3) data.yaml. Ultralytics resolves a *relative* `path` against its datasets_dir (~/datasets),
    #    not this file -- so write the absolute dataset root (the generated yaml is gitignored, and the
    #    script recomputes it on any machine).
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\n"
        "train: images/train\n"
        "val: images/train\n"          # no held-out val here; Phase C sets its own eval via the TTD harness
        "names:\n  0: smoke\n"
    )

    print("\n=== figlib_corrected built ===")
    print(f"  fires (day-only):        {len(fires_kept)}")
    print(f"  frames w/ boxes:         {n_pos}  ({boxes_total} boxes)")
    print(f"  background negatives:    {n_neg}")
    print(f"  night frames dropped:    {n_night}")
    if n_missing_img:
        print(f"  listed-but-missing imgs: {n_missing_img}")
    print(f"  output:                  {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
