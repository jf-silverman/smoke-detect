# How this project was actually built: a research narrative

Most write-ups present findings as if they arrived in order, cleanly. This one didn't, and the
messy version is more useful — because the sharpest turns came from the *interaction* between an
author with domain instincts and an LLM that could build and test fast. This document keeps the
real sequence: what we tested, what broke, and which move unlocked the next.

The through-line is a division of labor that kept paying off:

- **The LLM** supplied breadth and speed — surveying datasets, writing the pipeline, catching
  silent bugs, running experiments, and (importantly) reporting negative and incremental results in
  full instead of dressing them up.
- **The author** supplied direction and domain judgment — setting the field-realistic-evaluation
  frame, correcting the *metric* (not just the model), choosing which thread to pull, and, at the
  decisive moment, asking the physics question that rescued an experiment the LLM had written off.

Neither half would have produced this arc alone.

For the findings themselves — the numbers, per stage, with a table of contents — see the companion
[findings report](smoke-detection-report.md). This document is the *story*.

**Contents:** [The sequence of what we tested](#the-sequence-of-what-we-tested) ·
[The turns worth remembering](#the-turns-worth-remembering) ·
[Why this is the interesting story](#why-this-is-the-interesting-story) ·
[Pointers](#pointers)

## The sequence of what we tested

| # | Step | Result | Whose move |
|--:|---|---|---|
| 1 | Survey datasets, methods, metrics; write a state-of-the-field report | Framed the whole project around the benchmark-vs-field gap | Author set the frame; LLM executed the survey |
| 2 | Leak-safe splits on pyro-sdis | Caught that 40 camera IDs = 8 physical towers; held out whole sites | LLM (bug caught by inspection) |
| 3 | Single-frame YOLO baseline | Reproduced the precision collapse: 42% false alarms on clean frames | LLM |
| 4 | Base-rate correction | Precision at 1% deployment base rate = **1.6%** — the field number | LLM |
| 5 | Hard-negative mining | False alarms **42% → 20%**; precision@1% doubled | Author said go; LLM built |
| 6 | Temporal model (the literature's expected fix) | **Negative result** — no gain on pyro-sdis; 76% of confusers are persistent, not flicker | Author said go; LLM built *and* reported the null result plainly |
| 7 | Typed confuser corpus | **74% of false alarms are clouds** — an original artifact | Author chose this direction |
| 8 | FIgLib positive control (onset data) | First run confounded: detector AUC **0.454**, worse than random | Author asked to run it |
| 9 | Native-resolution tiled inference | AUC **0.454 → 0.658**; positive control lands, temporal cuts false alarms 12–19 pts | **Author's resolution question**; LLM tested it |
| 10 | Recall-first metric reframe | Rebuilt evaluation on POD / false-alarm burden / relative economic value — F1 demoted | **Author's asymmetric-cost insight**; LLM grounded it in the field literature |
| 11 | Full-scale native-resolution (1280) *training* | Confirmed a standing prediction: POD **0.83** at ~half the burden of inference-only; the proof run had merely been undertrained | Author said pursue high-res training; LLM ran it |
| 12 | Time-to-detection (Phase A, FIgLib) | First TTD numbers in the project; **resolution lowers TTD, not just AUC** | Author chose the HPWREN/TTD thrust |
| 13 | Combine the two levers (native-res + hard-neg) | Burden ~173 → **133 FP/day** (−23%), −2.4 pts recall — a real but *incremental* win, reported as such | Author sequenced it; LLM built and stopped it at the plateau |
| 14 | Scout a Phase C box-data source (PYRONEAR-2025) before spending GPU | **Negative result** — it can't close the recent-CA gap and re-adds leaky HPWREN; pivoted to pseudo-labeling our own fires | Author asked for a recent labeled source; LLM profiled and eliminated it cheaply |
| 15 | Horizon-anchored motion probe on FIgLib (author's scanning observation) | Anchored inter-frame motion separates onset from pre-ignition at per-fire AUC **0.71 vs 0.57** for the detector, and is *complementary* to it — plus two clean negative results along the way | **Author's motion insight** (and the physics that refined it); LLM built the probe and reported the nulls |

## The turns worth remembering

**The field-realistic frame came first, from the author.** The opening instruction was not to build
a smoke detector but to measure what a detector would actually do in the field. Every later
decision — site-holdout splits, operator metrics, base-rate correction — descends from that frame.
An LLM left to optimize a number would have reported mAP and moved on.

**The LLM's job was partly to distrust itself.** Two subagents attributed a 79% field
false-positive rate to SmokeyNet; verification against the primary source showed it belonged to
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

**The author chose the fork that turned into an original contribution.** Offered several next
steps, the author picked the confuser corpus. Clustering the false alarms produced the cleanest
line in the project — *74% of them are clouds* — and an artifact (a typed confuser manifest) the
literature review had specifically flagged as missing from the field.

**And then the decisive moment.** The FIgLib positive control — meant to confirm that temporal
helps on onset data — came back broken: the detector scored AUC 0.454, worse than a coin flip.
The LLM diagnosed it as distribution shift (a French-tower detector loose in California) and had
begun scoping an expensive in-distribution retrain as the only way forward. Then the author asked a
plain question:

> *"Did you say there were high-res images? Would that be worth anything? I'm especially thinking
> we may lose things if we're downscaling resolution, looking for small smoke objects."*

That is domain physics, not machine learning. A smoke plume at the moment of ignition is a few
dozen pixels in a 3072-wide frame; our pipeline was resizing every frame to 640 px before
inference, pooling those pixels into nothing. We tested it immediately — same detector, same
weights, but run on native-resolution **tiles** instead of a downscaled whole frame. AUC jumped
**0.454 → 0.658**, and the positive control we'd nearly abandoned came to life: on the tiled
signal, requiring temporal persistence cut false alarms by 12–19 points — the exact mirror of
pyro-sdis, where the same rule *raised* them.

The LLM had the mechanism right (temporal helps on onset data) but had misattributed the failure
to something expensive to fix. The author's instinct about resolution — cheap to test, easy to
overlook — was the difference between a confounded result needing a big retrain and a confirmed one.

The insight then paid a second dividend. The author asked whether resolution had been costing us
on pyro-sdis all along. It had: re-running the baseline at native 1280 instead of 640 lifted
recall from 0.68 to 0.86 (+18 points) — the detector had simply been blind to smoke too small to
survive downscaling. One question, asked once, surfaced a lever on both datasets.

**The author corrected the metric, not just the model.** Midway through, the author made a domain
point the LLM had not pressed: in wildfire, sensitivity and specificity are not equally important —
a missed fire is catastrophic, a false alarm costs a watchstander a glance, and a human reviews
every candidate before resources move. That reframed the whole scorecard. The LLM went to the
field and meteorology literature and rebuilt evaluation around **probability of detection**, the
**false-alarm burden per camera per day**, and **relative economic value across cost-loss ratios** —
with F1 demoted to context, and the reports re-led accordingly. A model can be tuned; a *metric*
has to be chosen, and choosing the right one was the author's call.

**A prediction, then its confirmation.** The proof-scale 1280 training run had *underperformed* —
and the LLM had flagged, from the training curve, that it was merely undertrained rather than
evidence against high-res training. The full-scale run confirmed it: POD 0.83 at roughly half the
false-alarm burden of inference-only, the best config in the project. A called shot, then the data.

**The incremental result, reported as the incremental result.** The two proven levers —
native-resolution training and hard-negative mining — had never been combined, and the reports kept
naming that as "the deployable recipe." So we built it. It helped: the false-alarm burden fell
~23% (173 → 133 FP/camera/day) and deployment precision rose — but it cost ~2.4 points of recall,
and 133 false alarms per camera per day is still nowhere near the < 1 an operator can live with.
The temptation with a long-promised recipe is to oversell the payoff. We wrote it up as what it
was: a real but marginal win that does not, on its own, close the deployability gap. Mining also
surfaced a clean supporting fact — the *converged* full-scale model still false-alarms on 57.6% of
clean frames, so resolution buys recall, not calm.

**Extending the finding onto the field's headline metric.** Time-to-detection is what operators and
the academic reference actually lead with, and the project had never computed it. A cache-only,
leak-safe harness produced the first numbers — and the resolution lever showed up here too: native
tiling detects more fires *and* detects them minutes sooner, not merely at higher AUC. The through
-line held one more time.

**Cheap elimination beats an expensive pull.** Phase C — an in-distribution detector — needs
bounding boxes FIgLib doesn't ship, and the recent-California miss made a *recent, labeled* source
the obvious want. PYRONEAR-2025 looked like it: images and videos, 640 fires, US among them. Rather
than download several GB on faith, we read its manifest first — the repo ships a per-fire test list
with the source encoded in every folder name. That five-minute read killed it: the US portion is all
old HPWREN (2016–2021), the only recent fires are Chilean and French, and it predates the very
2025 California fires we miss — so it *cannot* close the gap. It was also leakier than what we had,
its training pool literally containing the AI-For-Mankind set we'd already measured at 21% overlap
with our test fires. A negative result, arrived at before spending a dollar of GPU or an hour of
download — and it pointed to the cleaner path the author had already sensed: pseudo-label our *own*
fires, where we hold the frames and onset labels and only the boxes are missing, with zero external
leak. Knowing what *not* to pull is part of the work.

**The author saw motion where the pixels hid it — and the physics sharpened it twice.** Correcting
pre-annotations in CVAT, the author noticed a plume is often legible only by its *motion* across
frames, invisible in any single still. That became a probe: does inter-frame change separate onset
from pre-ignition frames? The first framing — weight motion *below* the horizon — the author then
corrected himself: plumes *rise*, so the diagnostic property isn't staying low, it's staying
**anchored to the ground** (base at the horizon, column connected to it), while clouds float
disconnected above the skyline. That reframing is what made the feature work: on the training fires,
horizon-anchored motion separates onset at per-fire AUC 0.71 versus 0.57 for the detector's own
confidence, with a floating-change *control* sitting at chance — so it is anchoring, not merely
motion, doing the work. And it is *complementary*: fires the appearance detector is weak on (conf
0.23) are rescued by motion (0.97), which is the whole case for a motion channel in Phase C.

Two things about this turn are worth keeping. First, **the measurement was as important as the
feature**: pooling frames across fires understated the signal (AUC ~0.62) because absolute motion
scale varies by scene; scored the operationally correct way — *within* each fire — it was 0.71. The
same feature, measured wrong, would have been dismissed. Second, **it produced two clean negative
results, reported as negatives.** A texture cue (diffuse haze + hard column edge) was cut once the
author pointed out it breaks under the very conditions that matter — a high-Haines day gives a
hard-edged column with no haze, a windy day an all-diffuse one with no edge. And a proposed
smoothed-skyline horizon *upgrade* was built, measured, found to make things worse (0.715 → 0.616),
and reverted — a fix that wasn't. Neither was papered over. See
[motion-findings.md](../research/motion-findings.md).

## Why this is the interesting story

The finished results table is respectable. But the *reason* it exists is a loop that a solo
author or a solo model would both have run more slowly and less well:

- The LLM could build a leak-safe pipeline, a base-rate correction, a hard-negative miner, a
  temporal model, a clustering corpus, a tiled-inference probe, a recall-first metric suite, and a
  time-to-detection harness in the time it takes to discuss them — and could be trusted to say
  plainly when something did not work, or only half-worked.
- The author kept the project pointed at real-world value over vanity metrics, corrected the
  scorecard itself, chose the threads that mattered, and supplied the physical intuition — *small
  objects die under downscaling* — that no amount of pipeline speed would have surfaced on its own.

Domain expertise and LLMs are not substitutes. The expertise decides *what is worth testing, which
metric is right, and why*; the LLM collapses the cost of testing it to near zero. When the loop is
tight, you get to run the experiment the instant the insight arrives — and that is when the good
results happen.

## Pointers

Each step has its own report with the numbers and caveats:
[state-of-smoke-detection](../research/state-of-smoke-detection.md) ·
[metrics](../research/metrics.md) ·
[baseline-findings](../research/baseline-findings.md) ·
[hard-negative-findings](../research/hard-negative-findings.md) ·
[temporal-findings](../research/temporal-findings.md) ·
[confuser-corpus](../research/confuser-corpus.md) ·
[figlib-findings](../research/figlib-findings.md) ·
[resolution-findings](../research/resolution-findings.md) ·
[ttd-findings](../research/ttd-findings.md) ·
[motion-findings](../research/motion-findings.md) ·
[backlog](../research/backlog.md)
