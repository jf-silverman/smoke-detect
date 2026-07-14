# How Smoke Detection Performance Is Measured

## The headline finding

**Verified against the SmokeyNet paper (ar5iv/2112.08598) — the widely-repeated attribution
of this number to SmokeyNet is wrong, and the truth is sharper.**

An earlier deployed detector (**Govil et al. 2020**, the Fuego/Open-Climate-Tech lineage) was
field-tested on 65 HPWREN cameras over nine days in October 2019. The SmokeyNet paper reports:
"After suppressing repeat detections in a one-hour timespan, only 21% of notifications showed
smoke from real fires (i.e., a 79% false positive rate)."

**SmokeyNet itself was never field-deployed.** Its 83.49% accuracy / 82.59% F1 / 3.12-min
time-to-detection are controlled test-set results only.

So the state of the art is: the one system we have a field number for produced ~4 false alerts
for every real one, and the benchmark leader has no field number at all. That gap between
benchmark and deployment should shape this entire project.

## Metrics by task framing

### Classification
Use precision, recall, F1, AUC-PR. **Not accuracy** — camera networks see no smoke in the
overwhelming majority of frames, so "always predict no-smoke" scores near-100% accuracy
while being worthless. AUC-PR beats AUC-ROC under this imbalance, since ROC can look good
even with many false positives when negatives dominate.

### Detection (bounding boxes)
mAP@0.5, mAP@0.5:0.95, IoU.

**IoU is a genuinely poor fit for smoke.** IoU assumes a crisp, agreed-upon boundary.
Smoke is amorphous and translucent and fades into haze — there is no physical edge to
annotate. Annotators demonstrably disagree with each other in transparent and boundary
regions; some dataset protocols use three annotators with majority-vote pixel labeling
specifically because of this. The consequence is an **annotation-noise ceiling**: a perfect
model cannot exceed inter-annotator agreement, so a mediocre IoU may reflect label
ambiguity rather than model failure.

This is why smoke papers report loose thresholds (mAP@0.5) rather than tight ones, and why
some use AP@0.1 deliberately (SKLFS-WildFire ContrastSwin, AP@0.1 = 0.503).

### Segmentation
mIoU, Dice. Same ceiling, arguably worse — it demands pixel-exact boundary agreement.

### What operational systems actually report
**Time-to-detection (minutes from ignition to alert)** and **false positives per camera per
day.** Not mAP. These are the numbers that decide whether a crew can suppress a fire while
it's small, and whether human reviewers keep trusting the dashboard at all.

## What counts as "good" — verified numbers

| System | Reported result | Source quality |
|---|---|---|
| SmokeyNet (FIgLib) | detection usually within 15 min of ignition, <1 false positive/camera/day after suppression logic | peer-reviewed |
| SmokeyNet multimodal | 83.49% acc, F1 82.59%, precision 89.84%, recall 76.45%; time-to-detection 4.70 min, → 3.66 min with weather data | reported-by-source |
| **Govil et al. 2020** (field, 65 cameras, 9 days) | **79% of notifications were false positives** | the only real field number |
| SmokeyNet field | **never deployed** — test-set results only | the gap |
| YOLO v5–v10 on D-Fire | mAP@0.5 ~0.89–0.91; mAP@0.5:0.95 ~0.62–0.68 | reported-by-source |
| Pyronear PYRONEAR-2025 (hard, in-the-wild) | **F1 ~70%** | peer-reviewed |
| Pyronear community YOLOv8 (curated holdout) | precision 0.922, recall 0.898, F1 0.910 | **community blog, not peer-reviewed** |

The Pyronear pair is the most instructive row in the table: ~70% vs ~91% F1 from the same
lineage of models and data, differing only in how hard the test set is. Cherry-picking
between them is exactly the failure mode to avoid.

**Realistic ceiling:** F1 in the low-to-mid 80s is genuinely good. High 80s/90s means your
test set is probably too easy.

## Precision/recall tradeoff

The asymmetry is structural: a missed fire can become catastrophic; a false alarm costs
reviewer time and is recoverable. Real systems therefore lean **recall-favoring** and absorb
the precision cost with a cheap human triage step — Pyronear routes alerts to a human-staffed
supervision platform rather than auto-dispatching.

There is **no industry-standard operating point.** The expected deliverable is the full PR
curve plus an explicit, justified choice of where to sit on it.

## Evaluation pitfalls — the ones that will sink a portfolio project

- **Incident leakage.** Frames from the same fire in both train and test lets the model
  memorize that plume, that camera's background, that day's light. Naive splits "yield
  optimistic performance estimates through incident-level leakage." Fix: grouped
  leave-one-incident-out (LOIO).
- **Same-camera leakage.** Splitting by image lets the model learn a specific ridge line or
  lens artifact. SKLFS-WildFire splits by administrative region so test cameras are in
  entirely different geography from train.
- **Curated "obvious smoke" test sets.** See the Pyronear 70/91 gap above.
- **A single number with no PR curve.**

**Correct split recipe:** group frames by fire incident; assign whole incidents to
train/val/test, never splitting within one; additionally hold out entire camera sites from
test; report "seen-camera/unseen-fire" vs "unseen-camera/unseen-fire" separately — the gap
between them is itself the interesting result.

## Recommended protocol for this project

1. Split by fire incident, and hold out camera sites entirely for at least one fold.
2. Report the full PR curve; justify the operating point explicitly.
3. Classification: precision, recall, F1, AUC-PR. Never lead with accuracy.
4. Detection: mAP@0.5 headline, mAP@0.5:0.95 caveated with the annotation-ceiling argument.
5. **Time-to-detection proxy:** on FIgLib's timestamped sequences, compute minutes into the
   fire sequence at which confidence first crosses threshold, averaged per test fire.
   Comparable to SmokeyNet's 4.70 min.
6. **False positives per camera per day** on held-out negative sequences.
7. Report both an easy and a hard split.

## Sources

- SmokeyNet: https://arxiv.org/abs/2112.08598 · https://www.mdpi.com/2072-4292/14/4/1007
  (verified: the 79% FP field result is Govil et al. 2020, cited *within* this paper, NOT
  SmokeyNet's own; SmokeyNet was never field-deployed)
- Multimodal SmokeyNet: https://arxiv.org/pdf/2212.14143
- YOLO on D-Fire: https://pmc.ncbi.nlm.nih.gov/articles/PMC11398105/
- SKLFS-WildFire: https://arxiv.org/html/2311.10116v3
- PYRONEAR-2025: https://arxiv.org/abs/2402.05349
- Pyronear community writeup: https://www.earthtoolsmaker.org/posts/protecting-the-forest-early-forest-fire-detector/
- Incident leakage: https://arxiv.org/pdf/2605.18911
- FUEGO: https://fuego.ssl.berkeley.edu/

## Unverified — do not cite as fact

- No systematic org-wide time-to-detection stat for ALERTWildfire exists; the "2 minutes"
  figure circulating is a single anecdotal case from a figure caption.
- **No quantified alarm-fatigue threshold** for wildfire smoke was found. The phenomenon is
  discussed qualitatively; the specific false-positive rate at which operators start
  ignoring alerts is an open number. Do not invent one.
- The D-Fire table and SmokeyNet decimals were extracted by automated fetch, not
  hand-verified against source PDFs.
