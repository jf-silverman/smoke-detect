"""Figure for the temporal finding: it does not beat single-frame, and why.

Two panels, from the numbers in reports/temporal-findings.md:

  (left)  false-alarm rate on clean frames at MATCHED recall -- the three methods
          sit on top of each other, and persistence is worse where it matters.
  (right) the mechanism: the false alarms are persistent, not flicker, so a
          temporal model has nothing to grab onto.

    python src/models/plot_temporal.py   ->  reports/figures/temporal_result.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"

INK, GRID = "#1a1a1a", "#d9d9d9"
COLORS = {"single-frame": "#0072B2", "persistence": "#D55E00", "temporal-gru": "#009E73"}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    comp = json.loads((ROOT / "results" / "temporal_comparison.json").read_text())
    recalls = comp["targets"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.suptitle("Temporal context does not beat a single frame on pyro-sdis",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.0)

    # ---- left: FA rate at matched recall -----------------------------------
    for m, d in comp["methods"].items():
        ys = [d[f"recall_{r}"] * 100 for r in recalls]
        style = dict(marker="o", lw=2.2, ms=6, color=COLORS[m], label=m)
        if m == "single-frame":
            style.update(lw=3.0, ms=7, zorder=3)
        axL.plot(recalls, ys, **style)
    axL.set_xlabel("recall held fixed (operating point)", fontsize=10.5)
    axL.set_ylabel("false alarms on clean frames  (%)", fontsize=10.5)
    axL.set_title("At matched recall, lower is better", fontsize=11, color=INK)
    axL.invert_xaxis()  # tighter operating points (rightward) are stricter
    axL.grid(True, color=GRID, lw=0.7)
    axL.set_axisbelow(True)
    axL.legend(frameon=False, fontsize=10, loc="upper left")
    axL.annotate("persistence is WORSE\nat tight operating points",
                 xy=(0.30, 10.2), xytext=(0.44, 15.5), fontsize=8.8, color="#D55E00",
                 arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2))

    # ---- right: why -- the false alarms are persistent, not flicker --------
    groups = ["true smoke\ndetections", "false alarms\n(the problem)"]
    persistent = [90.7, 75.7]
    flicker = [9.3, 24.3]
    x = range(len(groups))
    axR.bar(x, persistent, width=0.55, color="#7a7a7a")
    axR.bar(x, flicker, width=0.55, bottom=persistent, color="#F0C808")
    for i, (p, f) in enumerate(zip(persistent, flicker)):
        axR.text(i, p / 2, f"persistent\n{p:.0f}%", ha="center", va="center", color="white",
                 fontsize=10.5, fontweight="bold")
        axR.text(i, p + f / 2, f"flicker  {f:.0f}%", ha="center", va="center", color="#5c4b00",
                 fontsize=9)
    axR.set_xticks(list(x))
    axR.set_xticklabels(groups, fontsize=10)
    axR.set_ylabel("share of frames  (%)", fontsize=10.5)
    axR.set_ylim(0, 100)
    axR.set_title("Why: the false alarms don't flicker\n"
                  "(grey = persistent across frames, yellow = flicker a temporal model could catch)",
                  fontsize=10, color=INK)
    axR.grid(True, axis="y", color=GRID, lw=0.7)
    axR.set_axisbelow(True)

    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.text(0.5, -0.02,
             "pyro-sdis held-out towers (marguerite, serre-de-barre). Persistent confusers "
             "(fixed cloud banks, glare, ridge haze) look the same frame-to-frame, so requiring "
             "temporal persistence keeps them while costing recall.",
             ha="center", fontsize=8.3, color="#555555", wrap=True)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = FIG / "temporal_result.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
