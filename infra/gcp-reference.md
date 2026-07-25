# GCP reference — GPU training, done cheaply and correctly

Read this before touching GCP infra. Companion docs: [../reports/gcp-plan.md](../reports/gcp-plan.md)
(strategy + cost model for this project) and [gcp-runbook.md](gcp-runbook.md) (the executable
steps + scripts we actually run: `gcp_setup.sh`, `train_vm_startup.sh`). This file is the
"why" and the gotchas; the runbook is the "how."

Scope: single-node, single-GPU Ultralytics YOLO training, occasional short runs (hours),
reading/writing a ~3.3 GB tarball + ~2.2 GB of FIgLib `.tgz` from `gs://smoke-detect-jfs/`.
Cost-conscious — no free-trial credit, everything bills.

---

## 1. Compute Engine GPU VMs done right

**Shape:** `g2-standard-4` (4 vCPU, 16 GB RAM) + 1× NVIDIA L4 is the right default for a single
YOLO11n job — L4 is the cheapest current-gen GPU family with enough VRAM, and `g2-standard-4` is
the smallest g2 shape that isn't starved on CPU during data loading (Google Cloud, "About GPU
instances," accessed 2026-07-24).

**Zone/capacity:** L4 is not in every zone. Check
[GPU regions and zones](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones)
before hardcoding a zone, and be ready to fall back to a sibling zone (e.g. `us-central1-a` →
`-b`/`-c`) if `CreateInstance` fails with a stockout error — that's a capacity issue, not a quota
issue, and retrying the same zone rarely helps within the hour (Google Cloud, "About GPU
instances," accessed 2026-07-24).

**GPU quota starts at 0 — the actual first blocker.** A new project's GPU quota (both the
regional `NVIDIA_L4_GPUS` metric and global `GPUS_ALL_REGIONS`) is 0 until you request an
increase; approval can take hours. Check and request from **IAM & Admin → Quotas**, filter by
metric name or GPU model, select the quota, "Edit Quotas," fill in the new limit and a
justification. Request the *regional* quota for the zone you'll use, not just the global one
(Google Cloud, "Compute Engine quota and limits overview," accessed 2026-07-24). `gcp_setup.sh`
step 1 already prints current quota — run that before anything else, every time you touch a new
project.

**Image family: list it, never hardcode a name you remember.** Deep Learning VM image family
names encode `framework-version-cuda-os-driver`, e.g. `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`
(the one `train_vm_startup.sh` uses today). Old convenience aliases like `pytorch-latest-gpu`
have been retired and will fail outright — that's the exact failure we hit. Always resolve the
current family at provision time instead of trusting a doc or a script comment:

```bash
gcloud compute images list --project deeplearning-platform-release \
  --format="value(NAME)" --no-standard-images | grep pytorch
```

(Google Cloud, "Choose an image | Deep Learning VM Images," accessed 2026-07-24). Each image
family has a documented support end date (the current PyTorch 2.9 family is supported into
2028) — worth a glance so a run doesn't land on something about to be deprecated.

**Boot disk sizing:** the boot disk holds the OS + CUDA/PyTorch image (tens of GB) plus whatever
you hydrate onto it (our ~5.5 GB of data, repo, and `runs/` output). 200 GB (current default in
`gcp_setup.sh`) is comfortable headroom; disks can be resized up later but Compute Engine
disk resizing is a manual step and does not need to match the source image size — a common
mistake is assuming the disk must equal the image size, when the boot disk size and the
image content are independent (Google Cloud, "Overview of creating an instance with attached
GPUs," accessed 2026-07-24). Bigger boot disk also means faster sustained persistent-disk
throughput during extraction, which mattered for our tarball pull.

## 2. Spot vs on-demand

Spot is 60–91% cheaper than on-demand for the same shape, but it comes with **no automatic
restart** — `automaticRestart` cannot be set on a Spot VM at all — and a best-effort ~30-second
preemption notice (a Preview flag can extend this to 120 s) (Google Cloud, "Create and use Spot
VMs," accessed 2026-07-24). Google explicitly frames Spot as appropriate "only for fault-tolerant
workloads that can withstand VM preemption" (same source) — that's a real constraint, not
boilerplate, given what we saw.

**`instanceTerminationAction` is the setting that burned us.** On preemption:
- `STOP` (default) — the VM transitions to `TERMINATED`, disk state is preserved, and the VM can
  be restarted (manually, or you can build a Cloud Scheduler / Pub/Sub restart loop). This is the
  resumable option.
- `DELETE` — the VM (and, unless the disk is set to survive independently, its boot disk) is
  deleted outright. **This is what we had set**, which is exactly why three back-to-back
  preemptions in ~20 minutes each left nothing to resume from — every restart was a cold boot
  from the startup script, redoing the git clone + data hydration before training could even
  begin.

Set `--instance-termination-action=STOP` for any Spot L4 run going forward, so a preemption loses
only in-flight epoch progress, not the whole VM state.

**Concrete preemption-safety guidance for our workload:** the ~20-minute preemption window we
observed is *shorter than one YOLO11n epoch* on this dataset in some configurations, meaning a
purely per-epoch checkpoint/sync cadence can still lose 100% of a run's progress if preemption
lands before epoch 1 finishes. Two changes make Spot actually safe here:
1. Keep `instanceTerminationAction=STOP` so the disk (with cloned repo + already-hydrated data)
   survives — the expensive setup work isn't repeated on every preemption.
2. Rely on Ultralytics' built-in `save_period`/`resume=True` checkpointing plus the existing
   5-minute `rsync` to GCS, but treat "hydrate data + clone repo" as a **one-time, idempotent**
   step in the startup script (skip re-download if data already present on disk) so a STOP→restart
   cycle resumes training within seconds, not tens of minutes.

If a run is short (single-digit hours) and you are actively watching it, **on-demand is often the
pragmatic choice** — L4 on-demand is only ~$0.71–0.85/hr all-in for `g2-standard-4`
(see §4), so the absolute dollars saved by Spot on a 2–4 hour run are small, while the
preemption/restart toil is not. Reach for Spot when a run is long enough (many hours) that the
savings dominate, and only after `instanceTerminationAction=STOP` + resume is verified to work.

## 3. Efficient data movement to/from GCS

**One tarball beats 33k loose objects.** Each GCS object fetch is a separate HTTPS request with
its own latency and API-call overhead; 33k small files serialized to ~40 minutes of per-object
calls in our case. A single ~3.3 GB tar object is one `cp`, saturates bandwidth instead of
request-rate limits, and is the documented pattern for "working with big data" on Cloud Storage
(Google Cloud, "Use Cloud Storage with big data," accessed 2026-07-24). Keep this pattern for
any future dataset additions — tar (or another single-archive format) before upload, untar on
the VM.

**Use `gcloud storage`, not `gsutil`, for anything new.** `gcloud storage cp`/`rsync` uses a newer
parallelization strategy and is measured at up to ~79% faster on download and ~33–57% faster on
upload than `gsutil` for the same transfer, with parallelism on by default — no flags needed
(Google Cloud, "New gcloud storage CLI for your data transfers," accessed 2026-07-24). `gsutil`
still works but is legacy; don't add new `gsutil` calls to scripts.

**Parallel composite uploads** kick in automatically for files above ~150 MiB unless disabled,
splitting a large upload into components uploaded in parallel and composed server-side — good
for our tarball, but note the resulting object only has a CRC32C checksum (no MD5), which is fine
for our use but worth knowing if something downstream checks MD5 (Google Cloud, "Parallel
composite uploads," accessed 2026-07-24).

**Clean the tarball before it leaves macOS.** GNU tar on the Linux VM doesn't understand Apple's
`AppleDouble`/xattr sidecar entries (`._*` files, `com.apple.quarantine` etc.) that macOS `tar`
embeds by default, and it spammed warnings to the serial console during extraction — which is
also what throttled it. This isn't a GCP-specific setting; fix it at creation time on macOS:
```bash
COPYFILE_DISABLE=1 tar -cf data_processed.tar data_processed/
```
`COPYFILE_DISABLE=1` tells macOS's `tar` to skip the xattr sidecar files entirely, so the archive
extracts cleanly with plain GNU `tar -xf` on the VM with no per-file warnings.

## 4. Cost controls & the "forgot to delete the VM" problem

**This is the risk that matters most for us.** A running on-demand VM (or a Spot VM stuck STOPped
and never restarted+deleted) bills continuously with no natural end — unlike a batch job, nothing
tells Compute Engine the work is "done."

What actually accrues cost, roughly, for our shapes (all approximate, **verify against the live
pricing page before relying on a number**, since GPU pricing changes and varies by commitment
type — Google Cloud, "GPU pricing," accessed 2026-07-24, and Google Cloud, "VM instance pricing,"
accessed 2026-07-24):

| item | approx. rate (us-central1) | notes |
|---|---|---|
| `g2-standard-4` on-demand (vCPU+RAM only) | ~$0.71/hr | independent of the GPU line item |
| NVIDIA L4 GPU, on-demand | brings all-in `g2-standard-4`+L4 to ~$0.85–1.00/hr | matches our existing estimate in gcp-plan.md |
| NVIDIA L4 GPU, Spot | ~60–91% off on-demand | see §2 for when it's actually worth it here |
| Persistent disk (boot, 200 GB, standard) | a few cents/hr, negligible for hour-scale runs | bills whether the VM is running or stopped |
| GCS Standard storage | ~$0.020/GB/month | our ~5.5 GB dataset ≈ $0.11/month |
| Egress (GCS → internet, e.g. pulling `runs/` to your Mac) | ~$0.12/GB | $0 if compute stays co-located with the bucket's region |

Idle GPU-hours from a forgotten VM dwarf every other line item here — a weekend-long oversight at
~$0.85/hr is ~$40–60, versus cents for the data sitting in GCS.

**Controls, layered:**
1. **Budget alert** (Billing → Budgets & alerts → Create budget): set a monthly threshold (e.g.
   $20–30) with email notifications at 50/90/100%. Be clear-eyed about the limitation: **a budget
   alert does not cap spending or stop any resource** — Google's own docs state this explicitly
   (Google Cloud, "Create, edit, or delete budgets and budget alerts," accessed 2026-07-24). It's
   a tripwire, not a circuit breaker, unless you also wire the budget's Pub/Sub notification to a
   Cloud Function that calls `instances.stop`/`delete` — worth doing once if forgetting VMs keeps
   happening, but out of scope for "occasional short runs" unless it recurs.
2. **A hard stop in the runbook, not just memory:** the existing `gcp_setup.sh` step 5 already
   bundles fetch-results + delete-VM — keep using it as the last step of every run, not an
   afterthought.
3. **`gcloud compute instances list --filter="status=RUNNING"`** as a 5-second habit at the start
   of any new session, before you spin up another VM — catches a prior run's leak immediately.
4. Consider `--max-run-duration` / instance scheduling for a hard ceiling: Compute Engine supports
   limiting a VM's total run time so it self-terminates even if nobody remembers to check
   (Google Cloud, "Limit the run time of a VM," accessed 2026-07-24) — a good belt-and-suspenders
   addition to `gcp_setup.sh` for training VMs specifically, since a training run has a known
   rough upper bound (a few hours).

## 5. The migration question: keep hand-rolled Compute Engine, or go managed?

**Recommendation: keep the hand-rolled Compute Engine VM for now; revisit Vertex AI custom
training only if runs become frequent/routine enough that teardown discipline becomes the
bottleneck rather than an occasional habit.**

Compared, for our situation (cost-conscious, occasional, single-node, single-GPU, reads/writes
GCS):

| dimension | Compute Engine (current) | Vertex AI custom training job | Cloud Batch |
|---|---|---|---|
| Automatic teardown | **No** — the "forgot to delete" risk is entirely on us (§4) | **Yes** — the job is defined to run a container/script to completion; Vertex AI provisions the worker, runs it, and tears the VM down when the job finishes or fails, with no persistent VM to forget (Google Cloud, "Create a serverless training job," accessed 2026-07-24) | **Yes** — same managed-job model; Batch explicitly frames itself around defined task lifecycles rather than a persistent VM (Google Cloud, "Create and run a job that uses GPUs," accessed 2026-07-24) |
| Cost premium over raw Compute Engine | none (it *is* raw Compute Engine) | Vertex AI custom training bills the same underlying machine/GPU/disk rates as Compute Engine — no separate per-hour platform fee for custom jobs (Google Cloud, "Configure compute resources for Vertex AI serverless training," accessed 2026-07-24) | Same — Batch is a scheduler over Compute Engine VMs, not a separately priced product |
| Spot + checkpoint-resume | Full manual control (what we have); we got the `instanceTerminationAction` setting wrong once | Documented first-class support ("Use Spot VMs with training" is a dedicated doc section) with the same underlying preemption behavior to design checkpointing around (Google Cloud, "Use Spot VMs with training," accessed 2026-07-24) | Documented Spot support via `provisioningModel: SPOT` in the job's allocation policy (Google Cloud, "Create and run a job that uses GPUs," accessed 2026-07-24) |
| Complexity to adopt from current setup | none — already running | Moderate: package `train.py` + deps into a container (or use a prebuilt PyTorch container + a packaged Python source dist), define a `worker_pool_specs` YAML/JSON with `machineSpec.machineType`, `acceleratorType`/`acceleratorCount`, and a `baseOutputDirectory` GCS URI; submit with `gcloud ai custom-jobs create` | Similar order of packaging effort, JSON job definition instead of YAML worker-pool-spec, GPU driver install flag |
| GPU quota | Same underlying Compute Engine GPU quota applies — no separate quota pool | Same Compute Engine GPU quota, consumed under the hood | Same |
| Submit + get outputs to GCS | `gcloud compute instances create ...` + hand-rolled startup script + hand-rolled `rsync` back (current) | `gcloud ai custom-jobs create --config=job.yaml` where the job's `baseOutputDirectory.outputUriPrefix` is a `gs://` URI; Vertex AI sets `AIP_MODEL_DIR` etc. inside the container so `train.py` writes straight to GCS conventions (Google Cloud, "Create a serverless training job," accessed 2026-07-24) | JSON job spec with a runnable script/container; GCS output handling is left to the job's own script (mount or explicit `gcloud storage cp`), less opinionated than Vertex AI here |

**Why not migrate now:** the entire pain list in this doc (§2–4) is process discipline, not a
structural flaw in Compute Engine that only a managed service fixes. Vertex AI's real win —
automatic teardown — directly answers pain point (6) from the top of this doc, but it comes with
a one-time cost: containerizing `train.py` (Dockerfile, requirements, GCS mount/`baseOutputDirectory`
wiring) for a codebase that currently just runs from a cloned repo on a VM. For a job run "a few
times, hours at a time," that packaging cost is not obviously worth paying versus just following
the `gcp_setup.sh` step-5 delete discipline (§4) and adding a `--max-run-duration` ceiling as a
backstop. **If run frequency increases** — say, weekly retrains as an ongoing habit rather than
occasional experiments — revisit this: the packaging cost amortizes, and automatic teardown
removes an entire failure mode permanently.

Cloud Batch sits between the two: same automatic-teardown benefit as Vertex AI, less ML-specific
scaffolding (no `AIP_MODEL_DIR` conventions, you write your own GCS I/O), and a JSON job spec
that's arguably a smaller conceptual jump from a gcloud VM script than Vertex AI's ML-training
abstractions. If the containerization step for Vertex AI ever feels like overkill, Cloud Batch is
the fallback managed option — but it doesn't offer anything Vertex AI doesn't already cover for
this specific workload, so it's not the first migration target.

**Sketch of what adopting Vertex AI would look like later** (not being done now — for reference
if run frequency grows):

```yaml
# job.yaml
workerPoolSpecs:
  machineSpec:
    machineType: g2-standard-4
    acceleratorType: NVIDIA_L4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: us-central1-docker.pkg.dev/PROJECT_ID/smoke-detect/train:latest
    args:
      - --data=gs://smoke-detect-jfs/data_processed.tar
      - --epochs=30
```
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=yolo11n-smoke-train \
  --config=job.yaml \
  --base-output-directory=gs://smoke-detect-jfs/runs/
```
(Google Cloud, "gcloud ai custom-jobs create," accessed 2026-07-24; Google Cloud, "Create a
serverless training job," accessed 2026-07-24.)

## 6. Modern conveniences worth adopting now

- **`gcloud config configurations`** — if more than one GCP project ever gets used (e.g. a
  scratch project for quota experiments vs. the real one), `gcloud config configurations create`
  lets you name and switch whole config profiles (project, zone, account) instead of repeatedly
  passing `--project` or running `gcloud config set` by hand (Google Cloud, "Managing gcloud CLI
  configurations," accessed 2026-07-24). Low effort, avoids "ran the delete command against the
  wrong project" mistakes.
- **`gcloud storage` over `gsutil`** — already covered in §3; just don't add new `gsutil` calls.
- **Idempotent startup scripts** — make `train_vm_startup.sh`'s data-hydration step skip work if
  the data is already present (check for a marker file before re-downloading/re-extracting). This
  is what actually makes `instanceTerminationAction=STOP` + restart cheap in wall-clock time
  (§2), and it's a startup-script change, not a GCP feature — but it's the single highest-leverage
  change available given how we set up spot resumability.
- **Cloud Shell** is fine for one-off `gcloud` commands (quota checks, listing instances) without
  installing anything locally, but for anything scripted, keep using the local `gcloud` CLI +
  `gcp_setup.sh` so the steps are versioned in the repo rather than living only in a Cloud Shell
  session.

## Works Cited

- Google Cloud. (2026). *About GPU instances* [Compute Engine documentation]. https://docs.cloud.google.com/compute/docs/gpus/about-gpus
- Google Cloud. (2026). *Overview of creating an instance with attached GPUs* [Compute Engine documentation]. https://docs.cloud.google.com/compute/docs/gpus/create-vm-with-gpus
- Google Cloud. (2026). *GPU regions and zones* [Compute Engine documentation]. https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones
- Google Cloud. (2026). *Compute Engine quota and limits overview* [Compute Engine documentation]. https://docs.cloud.google.com/compute/quotas-limits
- Google Cloud. (2026). *Choose an image | Deep Learning VM Images* [documentation]. https://docs.cloud.google.com/deep-learning-vm/docs/images
- Google Cloud. (2026). *Create and use Spot VMs* [Compute Engine documentation]. https://docs.cloud.google.com/compute/docs/instances/create-use-spot
- Google Cloud. (2026). *Limit the run time of a VM* [Compute Engine documentation]. https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime
- Google Cloud. (2026). *Use Cloud Storage with big data* [Cloud Storage documentation]. https://docs.cloud.google.com/storage/docs/working-with-big-data
- Google Cloud. (2026). *New gcloud storage CLI for your data transfers* [blog post]. https://cloud.google.com/blog/products/storage-data-transfer/new-gcloud-storage-cli-for-your-data-transfers
- Google Cloud. (2026). *Parallel composite uploads* [Cloud Storage documentation]. https://docs.cloud.google.com/storage/docs/parallel-composite-uploads
- Google Cloud. (2026). *GPU pricing* [pricing page]. https://cloud.google.com/products/compute/gpus-pricing
- Google Cloud. (2026). *VM instance pricing* [pricing page]. https://cloud.google.com/products/compute/pricing
- Google Cloud. (2026). *Create, edit, or delete budgets and budget alerts* [Billing documentation]. https://docs.cloud.google.com/billing/docs/how-to/budgets
- Google Cloud. (2026). *Create a serverless training job* [Vertex AI documentation]. https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job
- Google Cloud. (2026). *Configure compute resources for Vertex AI serverless training* [Vertex AI documentation]. https://docs.cloud.google.com/vertex-ai/docs/training/configure-compute
- Google Cloud. (2026). *Use Spot VMs with training* [Vertex AI documentation]. https://docs.cloud.google.com/vertex-ai/docs/training/use-spot-vms
- Google Cloud. (2026). *gcloud ai custom-jobs create* [Google Cloud SDK reference]. https://docs.cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create
- Google Cloud. (2026). *Create and run a job that uses GPUs* [Batch documentation]. https://docs.cloud.google.com/batch/docs/create-run-job-gpus
- Google Cloud. (2026). *Managing gcloud CLI configurations* [Google Cloud SDK documentation]. https://docs.cloud.google.com/sdk/docs/configurations
