# How this project was actually built: a research narrative

Most write-ups present findings as if they arrived in order, cleanly. This one didn't, and the
messy version is more useful — because the sharpest turns came from the *interaction* between a
human with domain instincts and an AI that could build and test fast. This document keeps the
real sequence: what we tested, what broke, and which move unlocked the next.

The through-line is a division of labor that kept paying off:

- **The AI** supplied breadth and speed — surveying datasets, writing the pipeline, catching
  silent bugs, running experiments, and (importantly) reporting negative results honestly
  instead of burying them.
- **The human** supplied direction and domain judgment — setting the honest-evaluation frame,
  choosing which thread to pull, and, at the decisive moment, asking the physics question that
  rescued an experiment the AI had written off.

Neither half would have produced this arc alone.

## The sequence of what we tested

| # | Step | Result | Whose move |
|--:|---|---|---|
| 1 | Survey datasets, methods, metrics; write a state-of-the-field report | Framed the whole project around benchmark-vs-field honesty | Human set the frame; AI executed the survey |
| 2 | Leak-safe splits on pyro-sdis | Caught that 40 "cameras" = 8 physical towers; held out whole sites | AI (bug caught by inspection) |
| 3 | Single-frame YOLO baseline | Reproduced the precision collapse: 42% false alarms on clean frames | AI |
| 4 | Base-rate correction | Precision at 1% deployment base rate = **1.6%** — the field number | AI |
| 5 | Hard-negative mining | False alarms **42% → 20%**; precision@1% doubled | Human said go; AI built |
| 6 | Temporal model (the literature's expected fix) | **Negative result** — no gain on pyro-sdis; 76% of confusers are persistent, not flicker | Human said go; AI built *and* reported the null honestly |
| 7 | Typed confuser corpus | **74% of false alarms are clouds** — an original artifact | Human chose this direction |
| 8 | FIgLib positive control (onset data) | First run confounded: detector AUC **0.454**, worse than random | Human asked to run it |
| 9 | Native-resolution tiled inference | AUC **0.454 → 0.658**; positive control then lands, temporal cuts false alarms 12–19 pts | **Human's resolution question**; AI tested it |

## The turns worth remembering

**The honest frame came first, from the human.** The opening instruction was not "build a smoke
detector" but "measure what a detector would actually do in the field." Every later decision —
site-holdout splits, operator metrics, base-rate correction — descends from that frame. An AI
left to optimize a number would have reported mAP and moved on.

**The AI's job was partly to distrust itself.** Two subagents attributed a "79% field
false-positive rate" to SmokeyNet; verification against the primary source showed it belonged to
a different system (Govil et al.), and SmokeyNet was never field-deployed at all. The class-id
remap (pyro-sdis ships smoke as class `1`; Ultralytics expects `0`) would have silently trained
the model on zero positives if it hadn't been caught. Speed is only useful with a verification
habit bolted to it.

**The most valuable results were the negative ones.** The temporal model was *supposed* to be
the differentiator — the literature is unanimous that frame-to-frame context is the fix. On
pyro-sdis it did nothing, and rather than quietly drop it, we measured *why*: the false alarms
are persistent structures (76% of them), not the flicker a temporal model suppresses. That null
result reframed the whole project — it said the leverage was in the negatives, which led
directly to the confuser corpus.

**The human chose the fork that turned into an original contribution.** Offered several next
steps, the human picked the confuser corpus. Clustering the false alarms produced the cleanest
line in the project — *74% of them are clouds* — and an artifact (a typed confuser manifest) the
literature review had specifically flagged as missing from the field.

**And then the decisive moment.** The FIgLib positive control — meant to confirm that temporal
helps on onset data — came back broken: the detector scored AUC 0.454, worse than a coin flip.
The AI diagnosed it as domain shift (a French-tower detector loose in California) and had begun
scoping an expensive in-domain retrain as the only way forward. Then the human asked a plain
question:

> *"Did you say there were high-res images? Would that be worth anything? I'm especially thinking
> we may lose things if we're downscaling resolution, looking for small smoke objects."*

That is domain physics, not machine learning. A smoke plume at the moment of ignition is a few
dozen pixels in a 3072-wide frame; our pipeline was resizing every frame to 640 px before
inference, pooling those pixels into nothing. We tested it immediately — same detector, same
weights, but run on native-resolution **tiles** instead of a downscaled whole frame. AUC jumped
**0.454 → 0.658**, and the positive control we'd nearly abandoned came to life: on the tiled
signal, requiring temporal persistence cut false alarms by 12–19 points — the exact mirror of
pyro-sdis, where the same rule *raised* them.

The AI had the mechanism right (temporal helps on onset data) but had misattributed the failure
to something expensive to fix. The human's instinct about resolution — cheap to test, easy to
overlook — was the difference between "confounded, needs a big retrain" and "confirmed."

## Why this is the interesting story

The finished results table is respectable. But the *reason* it exists is a loop that a solo
human or a solo model would both have run more slowly and less well:

- The AI could build a leak-safe pipeline, a base-rate correction, a hard-negative miner, a
  temporal model, a clustering corpus, and a tiled-inference probe in the time it takes to
  discuss them — and could be trusted to say "this didn't work" out loud.
- The human kept the project pointed at honesty over vanity metrics, chose the threads that
  mattered, and supplied the one piece of physical intuition — *small objects die under
  downscaling* — that no amount of pipeline speed would have surfaced on its own.

Domain expertise and AI are not substitutes. The expertise decides *what is worth testing and
why*; the AI collapses the cost of testing it to near zero. When the loop is tight, you get to
run the experiment the instant the insight arrives — and that is when the good results happen.

## Pointers

Each step has its own report with the numbers and caveats:
[state-of-smoke-detection](state-of-smoke-detection.md) ·
[baseline-findings](baseline-findings.md) ·
[hard-negative-findings](hard-negative-findings.md) ·
[temporal-findings](temporal-findings.md) ·
[confuser-corpus](confuser-corpus.md) ·
[figlib-findings](figlib-findings.md)
