"""Figure for the FIgLib TTD Phase-A result: resolution lowers time-to-detection.

Reads the two cached-signal TTD sweeps (native-res tiled vs whole-frame @640) and plots the
operator trade-off: as the pre-ignition false-alarm budget loosens, how detection rate rises and
median time-to-detection falls -- for both resolutions. No model, no GPU.

    python src/models/plot_ttd.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG = ROOT / "reports" / "figures"

TILED = "#c6660f"     # matches the reports' medium-orange accent
WHOLE = "#6b7280"     # muted slate for the weaker whole-frame signal


def load(path: Path):
    d = json.loads(path.read_text())["far_sweep"]
    x = [r["far_target"] * 100 for r in d]
    det = [r["detection_rate"] * 100 for r in d]
    ttd = [r["median_ttd_min"] for r in d]
    return x, det, ttd


def main() -> None:
    xt, dt, tt = load(RESULTS / "figlib_ttd.json")            # tiled native-res
    xw, dw, tw = load(RESULTS / "figlib_ttd_wholeframe.json")  # whole-frame @640

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle("FIgLib time-to-detection: native-resolution tiling detects more fires, sooner",
                 fontsize=12, fontweight="bold")

    ax1.plot(xt, dt, "-o", color=TILED, lw=2, label="native-res tiled (AUC 0.658)")
    ax1.plot(xw, dw, "--s", color=WHOLE, lw=2, label="whole-frame @640 (AUC 0.454)")
    ax1.set_ylabel("detection rate (% of fires)")
    ax1.set_ylim(0, 100)

    ax2.plot(xt, tt, "-o", color=TILED, lw=2, label="native-res tiled")
    ax2.plot(xw, tw, "--s", color=WHOLE, lw=2, label="whole-frame @640")
    ax2.axhline(3.6, color="#2f7d32", ls=":", lw=1.5, label="SmokeyNet ref ~3.6 min")
    ax2.set_ylabel("median time-to-detection (min, detected fires)")
    ax2.set_ylim(bottom=0)

    for ax in (ax1, ax2):
        ax.set_xlabel("pre-ignition false-alarm budget (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.text(0.5, -0.02,
             "Leave-one-fire-out, 18 fires (n small — magnitudes directional). Same zero-shot "
             "pyro-sdis detector; only inference resolution differs. Loosening the false-alarm "
             "budget trades trigger-happiness for earlier, more complete detection.",
             ha="center", fontsize=7.5, color="#555")

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "ttd_result.png"
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
