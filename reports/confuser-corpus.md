# A typed confuser corpus for wildfire smoke detection

**What this is.** The single-frame detector false-alarms on 42% of clean frames. Until now
that number was one undifferentiated blob. This turns it into a small set of **named failure
modes with counts** — a confuser corpus — because knowing the detector fails is worth little
next to knowing *what it fails on*. The [state-of-the-field report](state-of-smoke-detection.md)
found no public corpus of labeled smoke confusers exists, and named building one the single
highest-leverage, lowest-cost contribution an individual can make to this problem.

**How it was built.** The 2,305 clean frames the baseline detector false-alarms on (mined
leak-safely from *training* sites only — see [hard-negative findings](hard-negative-findings.md))
were clustered on their 256-d detector embeddings (KMeans, k=6 chosen near the silhouette peak),
representatives pulled per cluster, and each cluster labeled by eye from the montage below.

## The finding in one number: **74% of the false alarms are clouds**

| confuser family | frames | share of all false alarms |
|---|---:|---:|
| **cloud** (cumulus + stratus + broken overcast) | 1,710 | **74.2%** |
| clear sky / thin high cloud over forest | 277 | 12.0% |
| low-sun glare & lens flare | 270 | 11.7% |
| fog / marine-layer whiteout | 48 | 2.1% |

The detector does not fail on a long tail of exotic edge cases. It fails, three times out of
four, on **clouds** — bright cumulus, backlit stratus, broken overcast on a ridgeline. This is
the documented single-frame failure mode of firing on nearly every cloud, measured and quantified on real camera data.
It is also *why the temporal model didn't help* ([temporal findings](temporal-findings.md)):
clouds are persistent, not flicker, so no temporal smoothing removes them — but they are a
learnable appearance, which is exactly why [hard-negative mining](hard-negative-findings.md)
(teaching the detector what a cloud looks like) is what actually moved the false-alarm rate.

## The six confuser types

![Six clusters of the frames the detector false-alarms on: sun glare, cumulus clouds, fog,
clear sky over forest, backlit stratus/haze, and broken overcast over ridgelines.](figures/confuser_montage.png)

| cluster | label | frames | mean conf | family |
|---:|---|---:|---:|---|
| 1 | bright scattered / cumulus clouds | 737 | 0.29 | cloud |
| 4 | overcast stratus & valley haze (backlit) | 633 | 0.30 | cloud |
| 5 | broken overcast over ridgelines | 340 | 0.33 | cloud |
| 3 | clear sky / thin high cloud over forest | 277 | 0.35 | clear/haze |
| 0 | low-sun glare & lens flare | 270 | 0.23 | sun/glare |
| 2 | fog / marine-layer whiteout | 48 | 0.22 | fog |

A few things worth reading off this:

- **Cumulus and stratus are separate failure modes** (clusters 1 and 4), and both are large. A
  fix aimed only at puffy cumulus would leave the backlit-stratus half untouched.
- **The highest-confidence confuser is clear sky over forest** (cluster 3, mean conf 0.35) —
  the detector is *most* certain on some of its emptiest frames. Thin high cloud and the
  horizon haze-line appear to be read as faint plumes.
- **Fog is rare here** (2%). That is a property of these particular French mountain towers, not
  a general truth — a marine-layer coast (e.g. HPWREN's California cameras) would weight this
  far higher. A confuser corpus is site-dependent, and that is part of the point.

## Honest caveats

- **Clusters partly track terrain, not only sky.** The global detector embedding encodes the
  whole scene, so the tower mast and each site's skyline leak into the clustering (the two
  cloud-with-mast clusters are brison-heavy). The *atmospheric* label on each cluster is the
  legible, dominant axis, but these are not pure weather classes — a cleaner corpus would crop
  to the alarm region before embedding, or label sky condition directly.
- **k=6 is a choice, not a truth.** Silhouette peaked at k=5 and was flat to k=6; the six types
  are a useful human summary, not canonical categories.
- **Mined from training sites only**, so the corpus is leak-safe with respect to the held-out
  evaluation, but it therefore describes 4 of the 8 towers.
- Built on the proof-scale (underfit) detector; the *kinds* of confuser are trustworthy, the
  exact proportions will shift with a stronger detector.

## The artifact

- `results/confuser_corpus.csv` — every confuser frame with its cluster, human label, and
  family. The shareable corpus: image pixels stay gitignored, so this manifest (plus the
  montage) lets anyone with pyro-sdis reconstruct the labeled set.
- `reports/figures/confuser_montage.png` — the visual key above.

## Reproduce

    python src/data/build_confuser_corpus.py --scan      # silhouette over k
    python src/data/build_confuser_corpus.py --k 6        # clusters + montage + manifest
