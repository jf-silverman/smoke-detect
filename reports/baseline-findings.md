# Baseline findings — single-frame YOLO on pyro-sdis

**Status: proof run, deliberately underfit.** yolo11n, 15 epochs, 20% of the training
data (~4,800 images), ~55 min on Apple MPS. These numbers are directional, not final —
the point was to validate the pipeline and see whether the predicted failure modes appear.
They do, cleanly.

## Setup

- Train/val/test are **site-disjoint** (grouped split). Test sites = `marguerite` +
  `serre-de-barre`, seen in neither training nor validation.
- Evaluated as an operator would: per-frame *alarm / no-alarm*, image-level
  precision/recall/F1 over a confidence sweep, plus the false-alarm rate on the
  true-negative (no-smoke) frames specifically.

## Result 1 — the model works, on unseen sites

At the best-F1 operating point (conf 0.05) on held-out sites:

| precision | recall | F1 |
|---|---|---|
| 0.938 | 0.676 | 0.786 |

An F1 of 0.79 from an underfit proof model on **sites it has never seen** is a legitimately
encouraging baseline — in the same range as the ~0.70 that PyroNear reports on their hard
in-the-wild benchmark. A properly-trained model (full data, more epochs) should clear this.

## Result 2 — the precision-collapse, quantified

That 0.938 precision is **an artifact of the test set's 90%-positive base rate**, and it is
misleading. The honest signal is that at the same operating point, **42% of the clean,
no-smoke frames triggered a false alarm** — the single-frame model firing on clouds, fog, and
glare, exactly as the literature predicts.

Real detection towers see smoke on a tiny fraction of frames. Recomputing precision from the
measured recall (TPR) and false-alarm-rate (FPR) at realistic base rates:

| conf | recall | FA-rate on clean frames | precision @ 5% | @ 1% | @ 0.5% |
|---|---|---|---|---|---|
| 0.05 (best-F1) | 0.676 | 42.0% | 7.8% | **1.6%** | 0.8% |
| 0.10 | 0.575 | 28.9% | 9.5% | 2.0% | 1.0% |
| 0.30 | 0.319 | 8.3% | 16.8% | 3.7% | 1.9% |
| 0.50 | 0.099 | 0.8% | 38.5% | 10.7% | 5.6% |

**At a 1% deployment base rate, best-F1 precision falls from 94% to ~2%.** This is the
benchmark-vs-field gap the state-of-the-art report warned about, reproduced on our own model
with our own data. It is not a bug — it is the central, honest finding, and it is exactly the
thing a naive "94% precision!" portfolio project would hide.

## What this tells us to do next

1. **The base-rate-corrected table is the headline metric**, not aggregate precision. Every
   future eval reports it (`base_rate_corrected` in the eval JSON).
2. **The precision floor is set by the false-alarm rate on clean frames**, so the highest-value
   next move is hard-negative mining: curate the frames the model false-alarms on (they will be
   clouds/fog/glare) and retrain. This directly attacks the 42% number.
3. **Temporal context is the differentiator.** A single frame cannot tell a static haze from a
   growing plume; the report's evidence (SmokeyNet's +26 precision points from temporal fusion)
   says this is where the false-alarm rate actually falls. This baseline exists to be beaten by it.

## Resolution is a hidden lever (added after the FIgLib work)

The FIgLib work showed that downscaling to 640 px was destroying small onset plumes. That
prompted the obvious question: does the same thing hurt pyro-sdis? It does. pyro-sdis is
1280×720, and its smoke boxes are small — median shorter side 28 px natively, **~14 px at 640,
with 60% of boxes under 16 px at 640.** Re-running the same detector at native 1280:

| inference | precision | recall | F1 | false alarms |
|---|---:|---:|---:|---:|
| 640 (downscaled) | 0.938 | 0.676 | 0.786 | 42% |
| 1280 (native) | 0.911 | **0.859** | **0.884** | 78% |

Native resolution lifts recall by **+18 points** — 640 literally cannot see smoke it detects at
1280 (its recall caps at 0.68). But it is a trade, not a free win: at *matched* recall inside the
overlap, 640 is actually more precise (42% vs 56% false alarms at recall 0.68), because the
detector was *trained* at 640 and over-fires when run at 1280. So resolution buys **recall
headroom** (catching small/distant smoke) at a precision cost, and capturing it cleanly would
mean *training* at higher resolution or tiling — not just upscaling inference. A real lever for a
future full-scale model, quantified here rather than assumed.

## Reproduce

    python src/models/train.py --split grouped --epochs 40           # full baseline
    python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped
    python src/models/evaluate.py --weights runs/grouped/weights/best.pt --split grouped \
        --imgsz 1280 --out results/eval_grouped_proof_test_1280.json  # resolution probe

Raw sweep + base-rate table: `runs/eval_grouped_test.json`.
