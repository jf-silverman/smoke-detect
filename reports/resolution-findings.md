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

Same honest grouped test set (held-out towers). The 640-train row is the original proof baseline.

| configuration | **max reachable recall** | FA at recall 0.55 | notes |
|---|---:|---:|---|
| 640-train, infer @640 (baseline) | 0.676 | 28.9% | can't see small smoke |
| 640-train, infer @1280 | **0.859** | 44.7% | just stop downscaling at inference |
| 1280-train, infer @1280 (proof) | 0.578 | 24.3% | **undertrained** — see below |

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
| val mAP50 | 0.550 | 0.547 | 0.505 | 0.541 | **0.563** |
| val recall | 0.548 | 0.541 | 0.498 | 0.534 | **0.561** |

Both are **still climbing at the final epoch** — the model is undertrained. Higher-resolution
inputs are a harder optimization (more pixels, more to fit) and need more epochs than 640 to
converge. The proof budget that was plenty for 640 is not enough for 1280. So the fair reading
is: *this run undersells 1280 training*, and a proper test needs more epochs (and ideally full
data). It is not evidence that training at 1280 is bad — only that 15 epochs of it is too few.

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
2. **Worth doing properly:** a full-scale 1280 training run (drop `--fraction`, raise epochs to
   convergence). The proof run was undertrained; the curve says it had not stopped improving.
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
