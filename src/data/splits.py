"""Grouped train/val/test splits for pyro-sdis.

The whole point of this module is to make the evaluation honest.

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
    grouped -- site-held-out, the honest one

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
    """Hold out whole SITES. The honest split.

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
        "grouped by site (HONEST)",
        _summarize(df),
        sites["train"],
        sites["val"],
        sites["test"],
    )
    if not report.is_clean:
        raise AssertionError(f"site leaked across partitions: {report}")
    return df, report
