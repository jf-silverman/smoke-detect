# When Smoke Detection Is Most and Least Useful — Literature Review

Prioritizes peer-reviewed reviews, a GAO technology assessment, and NASA/NOAA documentation.
Vendor content is flagged. Unverifiable claims are named as such rather than filled in.

## Where it works

**The operational niche is the pre-911 window** — compressing time-to-dispatch in the minutes
when a fire is a quarter-acre in a place nobody is looking. Not "beating a human at seeing smoke."

- **ALERTCalifornia / CAL FIRE**, the only large network with public claims: first season, AI
  detected 1,200+ fires and "beat 911 call reporting over 30% of the time."
  https://today.ucsd.edu/story/alert-california
  A 2026 claim of 3,600 fires with >50% ahead of 911 comes from a **press release** — treat as
  marketing, not evaluation.
- **The one consistently-cited first-detection case:** Wolf Mountain, Grass Valley CA,
  11 Sep 2023 — AI alert 5:19 a.m.; first 911 call 6:01 a.m.; crews already on scene; fire held
  to <¼ acre. **Source is the operator itself**; no independent after-action verification found.
- **No peer-reviewed, agency-audited catalogue of camera-first detections exists. That absence
  is itself a finding.**

Conditions where the ML problem is genuinely easy: daylight, high-contrast sky background, a
static camera with a learned background so the plume is the only novel structure, and a column
that grows monotonically across frames.

Where cameras beat humans structurally: remote terrain with **no reporting population**, and
relief from watchstander fatigue (human lookout accuracy degrades with "operator fatigue, time
of day, time of year, and geographic location"). https://pmc.ncbi.nlm.nih.gov/articles/PMC12987208/

## Where it fails

### Night — the regime is essentially abandoned
Plumes are not self-luminous. At night, optical smoke detection collapses into *bright-object
detection* (visible flame/glow), meaning the fire has already passed the incipient stage.

**Every major dataset excludes night frames.** PyroNear filters nighttime images (cameras switch
to grayscale IR); SmokeFrames-50k removed nighttime fires; FIgLib's authors dropped 19 night
fires. **So every published accuracy figure is a daytime figure.**

Thermal IR works at night but **does not see smoke** — it sees heat, requiring line-of-sight to
the source (sub-canopy and behind-ridge fires are lost), and is itself attenuated by smoke.
https://research.fs.usda.gov/firelab/about/thermal-imaging

ALERTCalifornia's claim of being "proven effective at night" is **unverified** and contradicts
the dataset practice above.

### Look-alikes — the dominant false-positive class
Low clouds, fog, marine layer, haze, dust, sun/water/metal glare, moving hillside shadows, steam.
PyroNear names "clouds, fog, or sunlight reflection" as the core discrimination problem;
SmokeyNet's error analysis names low-altitude cloud and haze as the common FP class.

**Anthropogenic smoke — prescribed burns, agricultural burning, industrial stacks — is
essentially indistinguishable to a plume detector.** No published quantification exists of what
fraction of operational alerts are controlled burns. Real gap in the literature.

### Geometry and hardware
Fixed cameras are "limited by their static coverage"; terrain and dense vegetation obstruct
line-of-sight. **A plume behind a ridge is invisible until it tops the ridge — by which point the
early-detection premise is gone.** Fog, rain, snow, wind, lens contamination, and glare all
degrade the sensor. No rigorous published statistics on maintenance-driven downtime.

### Satellite
- MODIS (1 km) routinely detects fires ~1000 m²; ~100 m² under near-ideal conditions. VIIRS at
  375 m detects smaller fires. https://www.earthdata.nasa.gov/data/tools/firms/faq
- **The tradeoff is unavoidable:** geostationary (GOES ABI, ~2 km, 5-min CONUS) gives temporal
  density but poor small-fire sensitivity; LEO gives resolution but a few overpasses per day,
  "often missing the critical afternoon window."
- **Cloud cover, thick smoke, and canopy can fully obscure a fire.** Sub-canopy smoldering is a
  documented blind spot.
- FIRMS latency runs "up to several hours" — insufficient for tactical initial attack.
- GAO: satellite "image resolution is limited, clouds can interfere, and data lags may occur."
  https://www.gao.gov/products/gao-25-108161

**Net: a fresh ignition is sub-pixel and often invisible from orbit.** Satellites are for
confirmed-fire monitoring and perimeter awareness, not the ignition window cameras target.

### False alarms and the human cost
**The single most important number in this review** (attribution verified against the source —
it is commonly misattributed to SmokeyNet): an earlier detector, **Govil et al. 2020**, was
field-tested on 65 HPWREN cameras over nine days (Oct 2019). "After suppressing repeat detections
in a one-hour timespan, only 21% of notifications showed smoke from real fires (i.e., a 79% false
positive rate)." Misses were never quantified.

**SmokeyNet — the benchmark leader — was never field-deployed at all.** Its 83.49% accuracy is a
test-set result. https://ar5iv.labs.arxiv.org/html/2112.08598

Alarm fatigue is a documented human-factors failure pathway. In wildfire ops, "human operators
must manually validate many false alarms" and FP volume "significantly burdens the workload of
personnel."

**Every deployed system found is human-in-the-loop: AI proposes, a trained watchstander confirms,
only then does dispatch happen. The model is not the system.**

## The honest framing

**Not solved — but the unsolved part is generalization and integration, not architecture.**

- **Benchmarks are inflated by partitioning.** PyroNear2025's cross-dataset table is the cleanest
  evidence: **Nemo scores 86.8% F1 on its own test set but 63.2% cross-dataset; SmokeFrames-2.4k
  82.8% → 69.9%.** The authors attribute the high in-dataset scores to "overfitting issues because
  of the partitioning." A realistic, diverse benchmark sits at **~70% F1**.
  https://arxiv.org/html/2402.05349v3
- **Datasets are unrepresentative by construction.** FIgLib is ~50/50 positive/negative — a ratio
  nothing like the real world, where the prior on "this frame contains a new fire" is minuscule.
- **Reviews converge:** "operational use of AI in wildfire contexts remains limited," blocked by
  dataset imbalance, **inadequacy of the standard metrics**, and model trustworthiness; systems are
  largely "not evaluated within operational emergency-response pipelines."
  https://www.mdpi.com/2673-2688/6/10/253 · https://www.mdpi.com/2076-3417/15/18/10255
- **GAO is bluntly agnostic:** "The best mix of emerging technologies is not known because their
  effectiveness for detecting wildfires is still being assessed."
- **An uncomfortable datapoint for the whole field:** across 4,934 Alberta fires (2015–2020), each
  hour of reporting-delay reduction cut suppression cost by only **~0.25%** (~$56 against a $22,667
  mean); detection delay accounted for ~3% of total suppression spend. The authors conclude the
  cost-savings case is "modest" and must rest on broader socioeconomic benefits.
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0313200
  This does not say detection is worthless — it says **the standard "early detection saves money"
  pitch is weakly supported**, and the real payoff (avoided catastrophic tail events, life safety,
  evacuation lead time) is not what most papers measure.

**Bottom line:** on a clear afternoon with a good sky background, smoke detection is close to
solved. Unsolved: (a) false-positive rate at operational base rates, (b) night, (c) generalization
to new cameras and geographies, (d) the dispatch pipeline the alert lands in. **The bottleneck is
the deployment envelope, not the backbone.**

## What's genuinely open — ranked for a portfolio project

1. **Operating at realistic base rates.** Nearly all metrics come from balanced datasets. **Almost
   nobody reports precision at a fixed alert budget** (e.g. ≤1 false alert per camera per week) —
   the only metric an operator cares about. Under-occupied and legitimate.
2. **Cross-camera / cross-geography generalization.** The 86.8% → 63.2% drop is the field's real
   open problem. Leave-one-camera-out protocols are rare.
3. **Negative-class engineering.** The literature obsesses over the positive class. A curated hard-
   negative corpus — fog banks, marine layer, dust, prescribed burns, industrial stacks, glare,
   shadow fronts, contrails — with per-confuser breakdowns **essentially does not exist publicly.**
   Highest-leverage, lowest-cost thing an individual can build.
4. **Wildfire vs. anthropogenic/prescribed smoke.** No published work solves or even benchmarks
   this, despite it being a first-order operational nuisance. Genuinely open.
5. **Night.** Datasets literally delete it.
6. **Temporal reasoning at deployment scale.** The strongest signal — a plume grows and persists
   while a cloud drifts — is what single-frame detectors throw away.
7. **Fusion + geolocation.** Camera bearing triangulation → satellite confirmation → dispatch. GAO
   calls out cross-system data incompatibility as unresolved.
8. **Calibration and human-facing confidence.** Since every deployed system has a human confirmer,
   well-calibrated confidence plus evidence presentation may matter more than a point of F1.

## Could not verify

- No independent/audited catalogue of "camera caught it first" incidents — all claims trace to the
  operator or press releases.
- **No published operational false-alarm rate for any currently-deployed statewide network.** The
  79% figure is a 2019 research field test, not a production system. Current systems are presumably
  better; **no one has published the number.**
- No quantification of what share of alerts are prescribed/agricultural burns.
- No public data on camera downtime from lens dirt, weather, or power.
- No NIST report specific to wildland smoke *detection* located (NIST's wildfire work found is
  WUI structure-ignition focused). Not claiming none exists — only that none was verified.

## Works Cited

- Dewangan, A., Pande, Y., Braun, H.-W., Vernon, F., Perez, I., Altintas, I., Cottrell, G. W., &
  Nguyen, M. H. (2022). FIgLib & SmokeyNet: Dataset and deep learning model for real-time
  wildland fire smoke detection. *Remote Sensing, 14*(4), 1007. https://doi.org/10.3390/rs14041007
- Govil, K., Welch, M. L., Ball, J. T., & Pennypacker, C. R. (2020). Preliminary results from a
  wildfire detection system using deep learning on remote camera images. *Remote Sensing, 12*(1),
  166. https://doi.org/10.3390/rs12010166
- Lostanlen, M., Isla, N., Guillen, J., Zanca, R., Veith, F., Buc, C., & Barriere, V. (2024).
  Constructing a real-world benchmark for early wildfire detection with the new PYRONEAR-2025
  dataset. *arXiv:2402.05349.* https://arxiv.org/abs/2402.05349
- NASA. (n.d.). *FIRMS frequently asked questions.* Earthdata.
  https://www.earthdata.nasa.gov/data/tools/firms/faq
- U.S. Department of Agriculture, Forest Service. (n.d.). *Thermal imaging* [Fire Lab].
  https://research.fs.usda.gov/firelab/about/thermal-imaging
- U.S. Government Accountability Office. (2025). *Wildland fire: Technology and data to support
  detection and response* (GAO-25-108161). https://www.gao.gov/products/gao-25-108161
- University of California San Diego. (n.d.). *ALERTCalifornia.* UC San Diego Today.
  https://today.ucsd.edu/story/alert-california
- *Alberta wildfire reporting-delay cost study.* (n.d.). *PLOS ONE.*
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0313200 (cited for the
  suppression-cost-vs-delay figure; author list not individually verified)
- *Wildfire detection review.* (2025). *Applied Sciences, 15*(18), 10255.
  https://www.mdpi.com/2076-3417/15/18/10255 (author list not individually verified)
- *Early wildfire detection survey.* (2025). *AI, 6*(10), 253. https://www.mdpi.com/2673-2688/6/10/253
  (author list not individually verified)

<sub>Titles marked "author list not individually verified" were located by URL during research
but their full author lists were not confirmed against the source; treat those citations as
provisional.</sub>
