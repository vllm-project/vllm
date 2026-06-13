# Code Notes: Kinfer Eval Integration

vLLM CI can deploy models on a Cohere K8s cluster via the **kinfer SDK** and
run bee evals against them. Bee logs results directly to W&B from inside the
cluster. A post-processing script (`log_kinfer_eval_to_wandb.py`) then extracts
scalar metrics from Bee's artifact tables and appends them to persistent
per-(model, device) W&B trend runs for nightly time-series charts.

## 1) Entry point: `nightly-benchmark.yaml`

Five jobs trigger the kinfer eval pipeline — one per hardware variant — all
calling `.github/workflows/kinfer-pipeline.yaml` as a reusable workflow:

```text
trigger-kinfer-base-eval-b200      → models: mls-base-fp8-b200,      device: b200
trigger-kinfer-base-eval-mi300x    → models: mls-base-fp8-mi300x,    device: mi300x
trigger-kinfer-base-eval-h100-llh  → models: mls-base-fp8-h100-llh,  device: h100
trigger-kinfer-base-eval-h100-gen  → models: mls-base-fp8-h100-gen,  device: h100
```

H100 is split into two jobs (llh + gen) to avoid OOM when running all eval
types in a single deployment. All jobs run in parallel and share a
`run_timestamp` generated once by the `generate-run-timestamp` job so that
W&B trend runs from the same nightly land at the same x-axis position.

The `run_mode` workflow dispatch input gates execution:

- `tests` — skips kinfer jobs entirely
- `evals` — runs only kinfer jobs (skips self-hosted dispatcher); also skips
  the Docker build when `docker_tag` is provided
- `both` (default for schedule) — runs everything

When `docker_tag` is provided, `trigger-build-and-push` is skipped entirely —
all kinfer eval jobs use the pre-existing image tag directly.

## 2) The kinfer pipeline: `kinfer-pipeline.yaml`

A `workflow_call` reusable workflow. Per (model, device) it:

1. **Set image ref**: selects `vllm-rocm` for `mi300x`, `vllm-nvidia` for all
   others. Fails loudly on unknown device values.

2. **Kinfer setup** (`.github/actions/kinfer-setup` composite action):
   - GCP Workload Identity Federation → fetches kubeconfig from GCP Secret
     Manager (`github-ci-kubeconfig-<cluster>` in project `cohere-cd`),
     normalises it to YAML via `kubectl config view --raw`
   - Logs in to Google Artifact Registry (`us-central1-docker.pkg.dev`)
   - Logs in to Oracle Container Registry (`iad.ocir.io`) if `OCI_AUTH_TOKEN`
     is set (required for H100/CoreWeave where kinfer runtime images live in OCIR)
   - Installs kinfer SDK: defaults to `kinfer==0.3.2` from the cohere-py GAR
     index. Override via `kinfer_install_spec` input (e.g. a git branch URL)
     — empty strings fall back to the default in the shell script.

3. **Run kinfer eval** (`run_kinfer_eval.py eval`):
   - Loads the per-model deploy spec YAML; injects the nightly Docker image
   - Deploys via `client.deployments.deploy_full_pipeline(wait_for_health=True)`
   - Writes `KINFER_TRAINJOB` + `KINFER_CLUSTER_CONTEXT` to `GITHUB_ENV`
     so the safety-net cleanup step has them even if the runner is killed
   - For each eval suite: injects W&B run name → `client.evals.launch()` →
     `client.evals.wait_for_logs()`; bee logs metrics directly to W&B
   - After each suite: calls `log_kinfer_eval_to_wandb.py` to extract scalar
     metrics from Bee's artifact tables into persistent trend runs
   - `finally: client.deployments.delete(trainjob)`

4. **Cleanup TrainJob (safety net)** — `if: always()`, guarded:
   ```bash
   python -c "import kinfer" 2>/dev/null || exit 0
   [ -n "$KINFER_TRAINJOB" ] || exit 0
   python run_kinfer_eval.py cleanup
   ```
   Skips silently when kinfer was never installed or no TrainJob was created.

### Secrets required

| Secret | Purpose |
| --- | --- |
| `CO_API_KEY_STAGING` | Bee uses this to call the Cohere API during evals |
| `WANDB_API_KEY` | Bee logs metrics to W&B; trend script reads/writes W&B |
| `OCI_AUTH_TOKEN` | Docker login to `iad.ocir.io` for H100/CoreWeave runtime images |

`DEPOT_PROJECT_ID=rkhjmshqw9` is set as an env var so kinfer's bundled
image builder targets the vllm-cohere Depot project rather than kinfer's own.

## 3) The driver: `run_kinfer_eval.py`

Located at `tests/cohere/scripts/run_kinfer_eval.py`. Two subcommands:

- **`eval`** (default): load deploy spec → deploy → run eval suites → teardown
- **`cleanup`**: read `KINFER_TRAINJOB` + `KINFER_CLUSTER_CONTEXT` from env
  and delete the TrainJob; used by the GHA safety-net step

Key design points:

- The per-model deploy spec YAML (in `configs/kinfer_deploy/`) is fully
  authoritative for hardware, weights, cluster, queue, vllm_args, etc.
  The only runtime injection is `cluster.image` (the nightly-built SHA).
- `_EarlyExportReporter` wraps `ConsoleReporter` and writes `KINFER_TRAINJOB`
  to `GITHUB_ENV` as soon as kinfer creates the TrainJob (before pod startup),
  so the safety-net step has it even if the runner is killed during queue wait.
- W&B run name is `{model}-{suite_stem}-{YYYYMMDD-HHMM}` (e.g.
  `mls-base-fp8-b200-base-20260605-1317`), injected at runtime into
  `eval_config["log_wandb_run_name"]`. Including the suite stem ensures each
  suite in a multi-suite deployment gets a distinct run.

## 4) W&B trend logging: `log_kinfer_eval_to_wandb.py`

Located at `tests/cohere/scripts/log_kinfer_eval_to_wandb.py`. Called once
per eval suite after `wait_for_logs` returns.

Flow:

1. Locates the Bee W&B run by display name (with up to 5 retries × 30s to
   handle indexing lag after the run completes).
2. Downloads each `*_metrics` artifact table and extracts the primary score
   column (see `_FALLBACK_COLS` for the lookup order).
3. NIAH tasks (`niah_multikey_2`, `niah_single_2`) emit one metric per context
   length (4k/8k/32k/49k) instead of a single average.
4. Appends ONE scalar step to the stable per-(model, device) persistent W&B
   run (`resume="allow"`, stable run ID prefix `vllm-ci-4-`).
5. Uses `commit_date_unix` (UTC midnight of the commit date) as the x-axis
   step metric so all hardware variants from the same nightly align.

W&B project: `cohere/cohere-vllm-ci-nightly-evals`

Persistent run naming: `{base_model} ({device})`, e.g. `mls-base-fp8 (b200)`.
For H100 variants that share a run ID (llh + gen), both log to the same run
— they write to non-overlapping metric namespaces so there is no key collision.

## 5) Configs

| File | Purpose |
| --- | --- |
| `configs/kinfer_deploy/{model}.yaml` | per-model deploy spec (hardware, weights, cluster, queue, priority, vllm_args) |
| `configs/kinfer_eval/eval_suite_map.json` | model key → ordered list of eval suite YAML filenames |
| `configs/kinfer_eval/base.yaml` | full MLS base eval suite (b200) |
| `configs/kinfer_eval/base_llh.yaml` | likelihood-only suite (H100-llh split) |
| `configs/kinfer_eval/base_gen.yaml` | gen + LC + code suite (H100-gen split) |
| `configs/kinfer_eval/base_mi300x.yaml` | full suite for MI300X; caps NIAH context at 32k |

All eval YAMLs set `raw_prompting: true` globally. Gen/code suites additionally
set `estimators.mls-base-fp8.raw_prompting: true` at the estimator level to
override the VLLMEstimator default of `false` introduced in bee 0.29.31
(the `mls_bf16_generation.toml` and `mls_bf16_code.toml` includes do not set
this, unlike `mls_bf16_lc.toml` which does).

Adding a new model = create `configs/kinfer_deploy/{model}.yaml` and add an
entry to `eval_suite_map.json`. Both resolvers fail loudly on missing entries.

## 6) Change hotspots

- `kinfer-setup/action.yaml` — `kinfer_install_spec` shell fallback and `default:`
  must both be updated when bumping the kinfer wheel version. No GitHub token
  is required — kinfer is installed from the cohere-py GAR index.
- `kinfer-pipeline.yaml` — `DEPOT_PROJECT_ID` must match vllm-cohere's Depot
  project; update if it changes.
- `nightly-benchmark.yaml` — `docker_tag` default should be `''` for production;
  only set to a pinned SHA during debugging.
- kinfer SDK method names (`deploy_full_pipeline`, `evals.launch`,
  `evals.wait_for_logs`, `deployments.delete`, `KinferSpec`, `KubeClient`,
  `switch_kube_context`) — verify on SDK upgrades.
- `bee_version` in eval suite YAMLs — bump when a new bee release is needed.
- `_TASK_METRIC_KEY` / `_MULTI_SCORE_TASKS` in `log_kinfer_eval_to_wandb.py` —
  update when eval tasks are added, removed, or renamed.
- `HARDWARE_GPU_PER_NODE` in `run_kinfer_eval.py` — keep in sync with
  kinfer's `_HARDWARE_GPU_PER_NODE` if hardware is added.
