"""Grouped train/val/test splits for pyro-sdis.

The whole point of this module is to make the evaluation leak-safe.

pyro-sdis images come from fixed detection towers. Each record carries a `camera`
string like `brison-200` or `cabanelle-125`, where the trailing number is the
camera's bearing on the tower mast -- `brison-110` and `brison-200` are two views
from the *same physical tower*, sharing terrain, hardware, lens characteristics and
sky. Grouping on the raw `camera` string therefore still leaks: the model can
memorize a site's background from one bearing and be scored on another.

So we group by SITE (the camera string with the trailing bearing stripped), and hold
entire sites out of the test set. A model evaluated this way has never seen the
terrain it is being tested on, which is the only condition under which the number
means anything for deployment on a new tower.

We deliberately produce two splits so the gap between them can be reported:

    random  -- the naive, leaky, flattering split (image-level shuffle)
    grouped -- site-held-out, the leak-safe one

The delta between those two numbers is a headline result, not an embarrassment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

BEARING_SUFFIX = re.compile(r"-\d+$")


def site_of(camera: str) -> str:
    """Strip the trailing bearing to get the physical tower.

    'brison-200' -> 'brison'; 'croix-augas-161' -> 'croix-augas'
    """
    return BEARING_SUFFIX.sub("", camera)


def has_smoke(annotations: str) -> bool:
    """pyro-sdis stores YOLO boxes as a string; empty means a true negative."""
    return bool(str(annotations).strip())


@dataclass(frozen=True)
class SplitReport:
    """Enough detail to prove the split is clean, not just assert it."""

    name: str
    counts: pd.DataFrame  # rows=split, cols=n_images/n_smoke/smoke_rate/n_sites
    train_sites: set[str]
    val_sites: set[str]
    test_sites: set[str]

    @property
    def is_clean(self) -> bool:
        """No site may appear in more than one partition."""
        return not (
            self.train_sites & self.test_sites
            or self.train_sites & self.val_sites
            or self.val_sites & self.test_sites
        )

    def __str__(self) -> str:
        lines = [f"[{self.name}] split", self.counts.to_string()]
        if self.train_sites or self.test_sites:
            leak = (self.train_sites & self.test_sites) | (self.val_sites & self.test_sites)
            lines.append(f"site overlap train/val vs test: {sorted(leak) if leak else 'NONE'}")
        return "\n".join(lines)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split")
    out = pd.DataFrame(
        {
            "n_images": g.size(),
            "n_smoke": g["has_smoke"].sum(),
            "smoke_rate": g["has_smoke"].mean().round(3),
            "n_sites": g["site"].nunique(),
        }
    )
    return out.reindex(["train", "val", "test"]).dropna(how="all")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns every split depends on."""
    df = df.copy()
    df["site"] = df["camera"].map(site_of)
    df["has_smoke"] = df["annotations"].map(has_smoke)
    return df


def random_split(
    df: pd.DataFrame, *, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> tuple[pd.DataFrame, SplitReport]:
    """The naive image-level shuffle. Leaks by construction -- that is the point.

    Included so the project can *quantify* how much a random split inflates the
    score, rather than merely asserting that it does.
    """
    df = prepare(df)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    split = pd.Series("train", index=shuffled.index)
    split.iloc[:n_test] = "test"
    split.iloc[n_test : n_test + n_val] = "val"
    shuffled["split"] = split.values

    return shuffled, SplitReport("random (LEAKY)", _summarize(shuffled), set(), set(), set())


def grouped_split(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[pd.DataFrame, SplitReport]:
    """Hold out whole SITES. The leak-safe split.

    Sites are assigned greedily largest-first into whichever partition has the most
    unfilled capacity. Greedy beats a random draw here because site sizes are heavily
    skewed -- a few towers dominate -- and a random assignment can hand the test set a
    single site or wildly miss the target fractions.
    """
    df = prepare(df)

    sizes = df.groupby("site").size().sort_values(ascending=False)
    total = int(sizes.sum())
    targets = {
        "test": total * test_frac,
        "val": total * val_frac,
        "train": total * (1.0 - val_frac - test_frac),
    }
    assigned: dict[str, float] = {k: 0.0 for k in targets}
    site_to_split: dict[str, str] = {}

    for site, n in sizes.items():
        # Absolute remaining capacity, not relative deficit: with skewed site sizes
        # a relative criterion sends the single biggest site into the smallest
        # bucket (every bucket starts 100% empty), blowing the target fractions.
        remaining = {k: targets[k] - assigned[k] for k in targets}
        pick = max(remaining, key=remaining.get)
        site_to_split[site] = pick
        assigned[pick] += float(n)

    df["split"] = df["site"].map(site_to_split)

    sites = {k: set(df.loc[df.split == k, "site"]) for k in ("train", "val", "test")}
    report = SplitReport(
        "grouped by site (LEAK-SAFE)",
        _summarize(df),
        sites["train"],
        sites["val"],
        sites["test"],
    )
    if not report.is_clean:
        raise AssertionError(f"site leaked across partitions: {report}")
    return df, report


def loso_folds(df: pd.DataFrame, *, val_frac: float = 0.15, seed: int = 0):
    """Leave-one-site-out cross-validation. The headline evaluation.

    pyro-sdis has only 8 physical sites, and they are badly skewed (brison alone is
    ~38% of the corpus). A single held-out-site test set is therefore dominated by
    whichever one tower landed in it -- the resulting score says more about that
    tower than about the model's ability to generalize.

    With 8 groups the sound move is to hold out each site in turn and report the mean
    and the spread. The spread is not noise to be averaged away: it is the finding.
    A model that scores 0.85 on one tower and 0.55 on another has not learned smoke,
    it has learned terrain, and only per-site results reveal that.

    Yields (fold_name, dataframe-with-split-column) for each of the N sites.
    """
    df = prepare(df)
    sites = sorted(df["site"].unique())

    for held_out in sites:
        fold = df.copy()
        is_test = fold["site"] == held_out

        # carve a val set out of the remaining sites, by site where possible so val
        # is also background-disjoint from train
        train_pool = sorted(set(sites) - {held_out})
        pool_sizes = fold[~is_test].groupby("site").size().reindex(train_pool)
        target = pool_sizes.sum() * val_frac
        val_sites: list[str] = []
        acc = 0.0
        # smallest-first so val lands near the target instead of overshooting
        for site, n in pool_sizes.sort_values().items():
            if acc >= target:
                break
            val_sites.append(site)
            acc += float(n)

        fold["split"] = "train"
        fold.loc[fold["site"].isin(val_sites), "split"] = "val"
        fold.loc[is_test, "split"] = "test"

        s = {k: set(fold.loc[fold.split == k, "site"]) for k in ("train", "val", "test")}
        report = SplitReport(
            f"LOSO holdout={held_out}", _summarize(fold), s["train"], s["val"], s["test"]
        )
        if not report.is_clean:
            raise AssertionError(f"site leaked in fold {held_out}: {report}")
        yield held_out, fold, report
