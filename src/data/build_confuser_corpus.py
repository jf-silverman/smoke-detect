"""Turn the mined false alarms into a typed confuser corpus.

The literature review found that no public corpus of labeled smoke *confusers* --
fog banks, glare, cloud, ridge haze, dust -- exists, and named building one the
single highest-leverage contribution an individual can make to this field. We are
already sitting on the raw material: `results/hard_negatives.csv` lists the 2,305
clean frames the baseline detector false-alarms on, and `features_proof.npz` holds a
256-d embedding for every one.

This script clusters those confusers by appearance (KMeans on the embeddings) so the
false-alarm problem stops being one undifferentiated blob and becomes a handful of
named failure modes with per-type counts. It emits:

  results/confuser_clusters.csv       every confuser frame + its cluster id
  reports/figures/confuser_montage.png rows = clusters, cols = representative frames

A human then reads the montage and assigns each cluster a label (fog / glare / ...);
those labels live in reports/confuser-corpus.md. The images themselves stay gitignored,
so the shareable artifact is the manifest + labels + montage -- reconstructible by
anyone with pyro-sdis.

    python src/data/build_confuser_corpus.py --k 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG = ROOT / "reports" / "figures"

# Human labels for the k=6 clustering (KMeans is deterministic at random_state=0, so
# cluster ids are stable). Assigned by eye from reports/figures/confuser_montage.png;
# see reports/confuser-corpus.md. If you change --k or the features, re-inspect and
# re-map -- the ids will move.
CLUSTER_LABELS = {
    0: ("low-sun glare & lens flare", "sun/glare"),
    1: ("bright scattered / cumulus clouds", "cloud"),
    2: ("fog / marine-layer whiteout", "fog"),
    3: ("clear sky / thin high cloud over forest", "clear/haze"),
    4: ("overcast stratus & valley haze (backlit)", "cloud"),
    5: ("broken overcast over ridgelines", "cloud"),
}


def load_confuser_embeddings() -> tuple[pd.DataFrame, np.ndarray]:
    hn = pd.read_csv(RESULTS / "hard_negatives.csv").dropna(subset=["image"])
    hn["stem"] = hn["image"].map(lambda n: Path(n).stem)

    arch = np.load(PROC / "features_proof.npz", allow_pickle=True)
    idx = {s: i for i, s in enumerate(arch["stems"].astype(str))}
    hn = hn[hn["stem"].map(lambda s: s in idx)].reset_index(drop=True)
    rows = hn["stem"].map(idx).to_numpy()
    return hn, arch["feats"][rows]


def choose_k(X: np.ndarray, ks: range) -> None:
    print("  k  silhouette")
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        print(f"  {k:2d}  {silhouette_score(X, km.labels_):.3f}")


def representatives(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, k: int, n: int):
    """Indices of the n frames closest to each cluster centroid."""
    reps = {}
    for c in range(k):
        members = np.where(labels == c)[0]
        d = np.linalg.norm(X[members] - centers[c], axis=1)
        reps[c] = members[np.argsort(d)[:n]]
    return reps


def montage(hn: pd.DataFrame, reps: dict, sizes: dict, n_cols: int, out: Path) -> None:
    thumb = 200
    pad, label_w = 6, 150
    k = len(reps)
    W = label_w + n_cols * (thumb + pad) + pad
    H = k * (thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)

    for r, c in enumerate(sorted(reps)):
        y = pad + r * (thumb + pad)
        draw.text((8, y + thumb // 2 - 10), f"cluster {c}\nn={sizes[c]}", fill="black")
        for j, ridx in enumerate(reps[c][:n_cols]):
            stem = hn.iloc[ridx]["stem"]
            p = PROC / "images" / f"{stem}.jpg"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB").resize((thumb, thumb))
            canvas.paste(im, (label_w + pad + j * (thumb + pad), y))
    canvas.save(out)
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--cols", type=int, default=6, help="representative frames per cluster")
    ap.add_argument("--scan", action="store_true", help="print silhouette over k and exit")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    hn, feats = load_confuser_embeddings()
    print(f"confuser frames: {len(hn)}  embedding dim: {feats.shape[1]}")

    Xz = StandardScaler().fit_transform(feats)
    X = PCA(n_components=args.pca, random_state=0).fit_transform(Xz)

    if args.scan:
        choose_k(X, range(4, 13))
        return

    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(X)
    hn["cluster"] = km.labels_
    sizes = hn["cluster"].value_counts().to_dict()

    reps = representatives(X, km.labels_, km.cluster_centers_, args.k, args.cols)
    montage(hn, reps, sizes, args.cols, FIG / "confuser_montage.png")

    # attach human labels when the clustering matches the mapped k
    if args.k == 6:
        hn["confuser_label"] = hn["cluster"].map(lambda c: CLUSTER_LABELS[c][0])
        hn["confuser_family"] = hn["cluster"].map(lambda c: CLUSTER_LABELS[c][1])
        cols = ["image", "site", "max_conf", "cluster", "confuser_label", "confuser_family"]
        hn[cols].to_csv(RESULTS / "confuser_corpus.csv", index=False)
        print(f"saved -> {RESULTS / 'confuser_corpus.csv'}")
        fam = hn.groupby("confuser_family").size().sort_values(ascending=False)
        print("\nper-family share of false alarms:")
        for k_, v in fam.items():
            print(f"  {k_:12s} {v:5d}  {100*v/len(hn):4.1f}%")
    else:
        hn[["image", "site", "max_conf", "cluster"]].to_csv(
            RESULTS / "confuser_clusters.csv", index=False)
        print(f"saved -> {RESULTS / 'confuser_clusters.csv'} (no labels: --k != 6)")

    print("\nper-cluster summary (label these from the montage):")
    summ = hn.groupby("cluster").agg(
        n=("image", "size"),
        mean_conf=("max_conf", "mean"),
        top_site=("site", lambda s: s.value_counts().index[0]),
        site_share=("site", lambda s: round(s.value_counts(normalize=True).iloc[0], 2)),
    ).sort_values("n", ascending=False)
    print(summ.to_string())


if __name__ == "__main__":
    main()
