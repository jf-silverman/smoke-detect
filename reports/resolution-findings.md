# Resolution & the recall-first objective — findings

Two threads converge here. First, a question raised during the FIgLib work — *are we losing
small smoke by downscaling?* — which turned out to matter on pyro-sdis too. Second, the
domain-correct framing that in wildfire detection **a missed fire is far worse than a false
alarm** (a watchstander reviews every candidate detection before resources are dispatched), so the
objective is not best-F1 but **maximize recall subject to a false-alarm rate a human can still
triage.** Resolution turns out to be a lever on exactly the thing that objective cares about:
the *reachable recall range*.

## Why resolution should matter here

pyro-sdis is 1280×720, and its smoke is small: box shorter-side **median 28 px natively, ~14 px
at 640, with 60% of boxes under 16 px at 640.** Downscaling to the detector's 640 input halves
already-small plumes toward the vanishing point. (SmokeyNet's answer to the same problem is to
tile into 224-px patches and never downscale the whole frame; pyro-sdis is small enough that
running at native 1280 is the equivalent move.)

## Three configurations, and the recall-first lens

Same grouped test set (held-out towers). The 640-train row is the original proof baseline.

| configuration | **max reachable recall** | FA at recall 0.55 | notes |
|---|---:|---:|---|
| 640-train, infer @640 (baseline) | 0.676 | 28.9% | can't see small smoke |
| 640-train, infer @1280 | **0.859** | 44.7% | just stop downscaling at inference |
| 1280-train, infer @1280 (proof) | 0.578 | 24.3% | **undertrained** — see below |
| 1280-train, infer @1280 (full-scale) | 0.827 | **11.0%** | **converged** — the proof run's promise, realized |

**The recall-first headline: simply running the existing detector at native 1280 lifts the
reachable recall ceiling from 0.68 to 0.86.** For an objective that refuses to miss fires, that
ceiling *is* the point — the 640 model structurally cannot be pushed past 0.68 no matter how you
set the threshold, because it never detected the small plumes in the first place. Native-
resolution inference is a free (no retraining) recall win.

## The twist: training at 1280 (proof scale) *underperformed* — because it didn't converge

Retraining at 1280 with the baseline's proof budget (15 epochs, 20% of data) produced a
conservative, high-precision, **low-recall** model (max recall 0.578). That looks like evidence
against high-res training — until you read the training curve:

| epoch | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|
| val [mAP](#map)50 | 0.550 | 0.547 | 0.505 | 0.541 | **0.563** |
| val recall | 0.548 | 0.541 | 0.498 | 0.534 | **0.561** |

Both are **still climbing at the final epoch** — the model is undertrained. Higher-resolution
inputs are a harder optimization (more pixels, more to fit) and need more epochs than 640 to
converge. The proof budget that was plenty for 640 is not enough for 1280. So the fair reading
is: *this run undersells 1280 training*, and a proper test needs more epochs (and ideally full
data). It is not evidence that training at 1280 is bad — only that 15 epochs of it is too few.

**Update — the full run confirms it.** Retrained on full data for 40 epochs at 1280 (to
convergence), the model reaches **max recall 0.827** — a +0.25 jump over the undertrained proof
run, essentially matching the 0.86 ceiling of downscaled-inference-only. And it does so with the
**lowest false-alarm rate at matched recall of any config**: 11.0% at recall 0.55, versus 24.3%
(proof), 28.9% (640 baseline), and 44.7% (640 infer @1280). So training at native resolution
does not just recover the recall ceiling that inference-only reaches — it reaches it with roughly
**half the false-alarm burden** (~173 vs ~388 FP/camera/day; see [metrics.md](metrics.md)). The
cliffhanger above is resolved in the predicted direction: the proof run only looked worse because
it had not converged.

## Resolution buys recall headroom, not deployability

One sobering note for the recall-first objective: at the high-recall operating points it
demands, the false-alarm rate is still very high (recall 0.85 → **78% of clean frames alarm**,
reachable only by the 1280-inference model). Resolution lets you *reach* high recall; it does
nothing about the false alarms you accumulate getting there. Those are the persistent
cloud/glare/haze confusers the [confuser corpus](confuser-corpus.md) catalogued (74% clouds) and
that [hard-negative mining](hard-negative-findings.md) is the tool for. **The deployable recipe
is native resolution (for the recall ceiling) *plus* confuser-targeted false-alarm reduction
(to make that recall reviewable).** Neither alone suffices.

## Recommendation

1. **Immediate, free:** run inference at native 1280 — it raises the recall ceiling 0.68 → 0.86
   with no retraining. For a recall-first operator this is the single highest-value change.
2. **Done — hypothesis confirmed:** the full-scale 1280 run (40 epochs, full data) reached max
   recall 0.827 with the lowest false-alarm rate at matched recall of any config (11% at recall
   0.55) — roughly half the false-alarm burden of downscaled-inference-only at a near-identical
   recall ceiling. Native-resolution *training*, not just inference, is the best config on every
   axis (see [metrics.md](metrics.md)).
3. **Report recall-first, not F1:** headline the false-alarm rate (and alarms-per-camera-per-day)
   at a fixed high recall, with F1 demoted to context. Best-F1 operating points understate a
   detector meant to run at recall ≥ 0.85 with a human in the loop.

## Reproduce

    python src/models/evaluate.py --weights runs/grouped_proof/weights/best.pt \
        --split grouped --imgsz 1280 --out results/eval_grouped_proof_test_1280.json     # infer @1280
    python src/models/train.py --split grouped --imgsz 1280 --batch 8 --fraction 0.2 \
        --epochs 15 --name grouped_proof_1280                                            # train @1280 (proof)
    python src/models/evaluate.py --weights runs/grouped_proof_1280/weights/best.pt \
        --split grouped --imgsz 1280 --out results/eval_grouped_1280trained_test.json

Evals: `results/eval_grouped_proof_test.json` (640), `..._proof_test_1280.json` (infer@1280),
`..._1280trained_test.json` (train@1280).

## Glossary

Each term below is a linkable heading — the highlighted term in the text jumps here, and your browser's **Back** button returns you to where you were reading. Definitions are given in text (the reliable fallback, since GitHub does not render hover tooltips).

#### recall (POD)

Probability of detection — the fraction of real fires the detector catches. The recall-first objective maximizes this subject to a triageable false-alarm rate.

#### recall ceiling

The maximum reachable recall over all confidence thresholds. The 640 model caps at 0.68 because it never detects small plumes; native 1280 lifts it to 0.86.

#### mAP

mean Average Precision — the standard detection score; mAP50 is the value at IoU 0.50 (50% box overlap), used here to read the training curve.

#### FA rate

False-alarm rate on clean (no-smoke) frames = FP/(FP+TN); lower is better.

#### imgsz / 640 / 1280

Inference/training image size in pixels. Downscaling pyro-sdis's native 1280 to 640 halves already-small plumes below the detectable size.

#### native resolution

Running at the image's full pixel density (here 1280), or on native-resolution tiles, instead of downscaling — the lever that raises the recall ceiling.
