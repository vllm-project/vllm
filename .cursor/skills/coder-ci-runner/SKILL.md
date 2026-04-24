---
name: coder-ci-runner
description: Run Cohere vLLM CI-equivalent tests on the current VM or on a remote Coder workspace using Docker or the target shell (mirrors build-and-test / build-and-eval / build-and-bench + dispatcher + test-pipeline). Use when the user asks to run CI tests locally, run them on another Coder workspace, or match GitHub Actions test behavior.
---

# Coder CI Runner

## Quick Start
Use this skill to replicate the same path GitHub uses after images exist: `dispatcher.yaml` + `test-pipeline.yaml` (test execution inside Docker). That path is reached from any of these entry workflows:

| GitHub workflow | Purpose | Maps to local `TEST_GROUP` / notes |
|-----------------|---------|-------------------------------------|
| `build-and-test.yaml` | Build image (unless tag provided), fan out GPUs, **feature tests only** | Set `TEST_GROUP` to a feature group (`fast_check`, `vision`, `model_arch`, …). Eval benchmarks are not run from this workflow. |
| `build-and-eval.yaml` | Same build/fanout, **eval only** (`lm_eval`, `bee_eval`; `features` fixed to none) | Set `TEST_GROUP` to `lm_eval` or `bee_eval`. Pick one `TEST_GROUP` per run. |
| `build-and-bench.yaml` | Same build/fanout, **perf only** (`features` fixed to none in CI) | Use `TEST_GROUP=performance` and set `BENCHMARK_OUTPUT_LEN` (e.g. `100` or `1000`) to mirror `perf_100` / `perf_1000`. |

Locally you skip the `build-and-push` job unless you build/pull an image yourself; the rest is the same `run_tests.sh` contract as CI.

Execution target can be either the current workspace (default) or a remote
Coder workspace. In remote mode, run the same test flow on the target workspace
via `coder ssh`.

Three execution modes are available:
- **One-shot** (default): `docker run --rm` - start, setup, test, container removed on exit.
- **Iterative**: `docker run -d` + `docker exec` - start once, setup once, then re-run tests after source changes without container/setup overhead.
- **Current env**: run directly in the shell on the chosen target (no Docker image).

## Workflow

### 1) Choose execution target
Prompt for one of:
- `current-workspace` (default)
- `coder-remote`

If the user chooses `coder-remote`, gather:
- `remote_owner`
- `remote_workspace`
- optional `remote_repo_root` if the repo is not at the same absolute path as
  the current workspace
- optional `commit_sha` if the remote run should match a specific commit;
  default to the local `git rev-parse HEAD` when the user wants parity with the
  current checkout

Before doing any remote test setup, use the `/sync-coder-remotes` skill
workflow to:
- verify `coder` is installed locally,
- verify `coder whoami` succeeds,
- verify the target workspace is visible with `coder list`,
- sync the remote repo to the exact commit SHA whenever repo contents affect the
  run.

Default remote command shape:
`coder ssh <owner>/<workspace> -- bash -lc '<command>'`

Use that form for short probes only. For download/setup/docker/test flows, prefer
`coder ssh <owner>/<workspace> -- env HF_TOKEN=... bash -s` and feed one remote
script over stdin to avoid brittle quoting and env propagation issues.

#### `coder ssh` reliability notes

- **Short probes** (`coder ssh ws -- <simple command>`) are generally reliable.
- **`bash -c` with complex quoting** can hang indefinitely, especially when
  combined with stdin piping or nested shell expansions. Prefer `bash -s` with
  a heredoc-fed script for anything beyond a one-liner.
- **Stdin piping of file content** does NOT work reliably through `coder ssh`.
  Commands like `cat file | coder ssh ws -- bash -c 'cat > /path'` or
  `coder ssh ws -- tee /path < file` may appear to stream content but leave
  **empty files** on the remote. Base64-encoding into command args also fails
  for large files (exceeds shell argument limits). To transfer file changes to
  the remote, **commit + push + fetch + checkout** via git is the reliable path
  (see section 7).
- **Backgrounded `coder ssh` sessions** can leave the local agent shell in a
  stuck state where subsequent Shell calls complete in 0ms with no output. If
  this happens, kill the orphaned `coder ssh` PID explicitly.

### 2) Detect target GPU (auto)
Try to infer the GPU type on the chosen execution target and suggest a default
before prompting.

Detection order:
1. **NVIDIA**: `nvidia-smi --query-gpu=name --format=csv,noheader`
2. **AMD**: `rocminfo | grep -i "Name:"`
3. **Fallback**: `lspci | grep -i -E "nvidia|amd|mi300|h100|a100|b200|gb200"`

Map common strings to supported values:
- If output contains `H100` -> `h100`
- If output contains `A100` -> `a100`
- If output contains `B200` -> `b200`
- If output contains `GB200` -> `gb200`
- If output contains `MI300` -> `mi300x`

If the target is remote, run the same probes through
`coder ssh <owner>/<workspace> -- bash -lc '<probe>'`.

If detection fails or is ambiguous, prompt for `gpu` without a default.

When GPUs are detected, also probe utilization so the user can pick free GPUs
if some are already in use:

**NVIDIA**:
```
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

**AMD**:
```
rocm-smi --showuse --showmemuse
```

Show the output and suggest which GPU indices appear free (low memory usage and
utilization). If all GPUs are idle, default to using all of them and skip the
`CUDA_VISIBLE_DEVICES` prompt. If some GPUs appear busy, suggest setting
`cuda_visible_devices` to the idle indices.

### 3) Gather run parameters (prompt)
- `execution_target`: `current-workspace` (default) or `coder-remote`
- `run_mode`: `one-shot` (default), `iterative`, or `current-env`
- `test_group`: one of `cpu`, `fast_check`, `model_arch`, `model_arch_logits`, `model_arch_reward`, `model_arch_c5_3a30t`, `lm_eval`, `bee_eval`, `performance`, `guided_generation`, `speculative_decoding`, `vision`
- `gpu`: one of `h100`, `a100`, `b200`, `gb200`, `mi300x` (default to detected value if available)
- `models`: comma-separated list (used for eval/perf; ignored for feature tests). Use **plain model ids** locally (no `:tpN` suffix); set `TP_SIZE` separately to match the desired tensor parallel degree. On GitHub, **`build-and-eval.yaml`** and **`build-and-bench.yaml`** take the same `models` string with optional per-entry `model:tpN`; `.github/scripts/dispatcher-set-matrix.js` strips suffixes and sets runner + `TP_SIZE` before `run_tests.sh`. To mirror a CI run that used `command-r7b_fp8:tp1`, use `MODELS=command-r7b_fp8` and `TP_SIZE=1` locally.
- Public HF repo IDs (for example `CohereLabs/c4ai-command-r7b-12-2024`) are supported in local runs for `TEST_GROUP=performance`. Keep them as raw repo IDs in `MODELS`; `run_tests.sh` now passes the raw ID through to vLLM when no local checkpoint directory exists.
- `BENCHMARK_OUTPUT_LEN`: required when `TEST_GROUP=performance` (e.g. `100` or `1000`), matching `build-and-bench` inputs `perf_100` / `perf_1000`.
- `model_path`: GCS prefix (default: `gs://cohere-model-efficiency-ci/engines/`)
- `cuda_visible_devices`: comma-separated GPU indices to expose (e.g. `0,1` or `2,3`). Optional; defaults to all GPUs. Use when some GPUs on the target machine are occupied by other containers or workloads. Passed as `-e CUDA_VISIBLE_DEVICES=<value>` to Docker or `export CUDA_VISIBLE_DEVICES=<value>` in current-env mode. Works on both NVIDIA (CUDA) and AMD (ROCm/HIP respects this via its CUDA compatibility layer). For NVIDIA Docker, also replaces `--device nvidia.com/gpu=all` with per-index `--device nvidia.com/gpu=<index>` flags for stricter device-level isolation. For AMD Docker, `--device=/dev/kfd --device=/dev/dri` always exposes all GPUs, so the env var is the sole restriction mechanism.
- `docker_env_flags`: extra docker flags for local runs (format `-e VAR=value -e VAR2=value2`, optional)
- `result_upload_branch`: only affects reporting in GitHub Actions (local runs can ignore)
- `remote_owner`, `remote_workspace`, and optional `remote_repo_root` when `execution_target=coder-remote`
- `commit_sha` when `execution_target=coder-remote` and the remote checkout
  should match an exact revision

**Note**: `docker_env_flags` is for **local docker runs only**. The GitHub Actions workflows use `hardware_profiles_override` (YAML content) instead of environment variable flags.

Notes:
- Feature tests (`cpu`, `vision`, `guided_generation`, `speculative_decoding`, `model_arch`, `model_arch_logits`, `model_arch_reward`, `model_arch_c5_3a30t`, `fast_check`) ignore `models` and tensor parallel; run once with `TP_SIZE=0`.
- The `cpu` test group does not require a GPU or model downloads. It can run in `current-env` mode without Docker.

### 4) Ensure HF token is available
The agent's Shell tool runs in its own process, separate from the user's IDE
terminals. Environment variables set in user terminals are **not** inherited by
the agent shell. In remote mode, agent-shell env vars are also **not**
automatically inherited by the remote workspace shell.

Flow:
1. Check `HF_TOKEN` on the chosen execution target.
2. If set and non-empty, proceed.
3. If not set, ask the user to **paste the token value in chat**.
4. For `current-workspace`, export it in the agent shell.
5. For `coder-remote`, prefer
   `coder ssh <owner>/<workspace> -- env HF_TOKEN="$HF_TOKEN" bash -s` so the
   token is injected once into the remote script environment without printing
   or interpolating it into nested shell quotes.
6. **Never echo or log the token value** in shell output.

Note: Do **not** ask the user to export in their own terminal and "forward the
session" - that has no effect on the agent's shell environment.

### 5) Choose execution environment
Prompt the user with three options:
1) **Use an existing image** (default)
2) **Trigger build-and-push** to build a new image, then use it
3) **Run on current env** (no Docker image on the chosen target)

If the user chooses build:
- Invoke the `/trigger-github-actions` skill to run the `build-and-push` workflow.
- Ask for the resulting image tag (or fetch it from the workflow output if available).

If the user chooses existing image:
- Ask for full image reference, e.g. `us-central1-docker.pkg.dev/cohere-artifacts/cohere/vllm-nvidia:<tag>` or `.../vllm-rocm:<tag>`.

If the user chooses current env:
- Skip Docker image selection.
- Skip `docker_env_flags` unless the user later switches back to a Docker mode.

### 6) Dependency checks (prompt-on-missing)
Before running anything, check for required tools. If a check fails, prompt the user to install the missing dependency and wait.

Minimum checks:
- `docker` (only for one-shot/iterative Docker modes)
- `gsutil` (or `gcloud` with `gsutil` available)
- `jq` (only if you need to parse local config files)
- `gh` (only if triggering build-and-push)
- `coder` (only when `execution_target=coder-remote`)

For `coder-remote`:
- check `coder` on the local agent machine,
- check `docker`, `gsutil`, `jq`, and other runtime tools on the remote
  workspace through `coder ssh`.

Use this pattern:
1. Run `command -v <tool>`
2. If missing, prompt: "`<tool>` is required. Install now? (y/n)"
3. If yes, provide the install command for the current OS and run it.

### 7) Paths on the chosen target
Use CI-style defaults, but ask for overrides only if the default path does not
exist or is not writable.

Defaults:
- `repo_root`: current workspace for `current-workspace`; same absolute path on
  the remote workspace by default for `coder-remote`
- `output_dir`: `<repo_root>/.cache/output`
- `engines_dir`: `<repo_root>/.cache/engines`
- `hf_cache_dir`: `<repo_root>/.cache/hf_cache`
If the user explicitly wants `/tmp` mounts and Docker can see them, allow override to:
- `output_dir`: `/tmp/vllm-output`
- `engines_dir`: `/tmp/engines`
- `hf_cache_dir`: `/tmp/hf_cache`

If the remote repo is checked out somewhere else, prompt for `remote_repo_root`
instead of guessing.

For `coder-remote`, when the run depends on repository contents, use
`/sync-coder-remotes` to create or refresh a synced detached worktree at the
requested `commit_sha`, then use that synced path as `repo_root` for subsequent
download, Docker mount, and test commands.

After choosing that synced path, treat it as the authoritative `repo_root` for
the whole run. Do not mix the base checkout back into later download or Docker
mount commands.

#### Syncing uncommitted local changes to the remote

When iterating on local source changes that need to run on the remote, do **not**
attempt to pipe files through `coder ssh` (see reliability notes in section 1).
Instead:

1. Commit the local changes (even a WIP commit is fine).
2. `git push origin <branch>`.
3. On the remote, fetch into the **main repo** (not the worktree directly):
   `coder ssh <ws> -- git -C /root/repos/vllm-cohere fetch origin <branch>`
4. Then checkout the new SHA in the **worktree**:
   `coder ssh <ws> -- git -C <worktree_path> checkout -f <new_sha>`

This works because the worktree's `.git` file points back to the main repo's
object store, so objects fetched into the main repo are immediately visible to
the worktree. Use `checkout -f` if previous failed file-transfer attempts left
dirty state in the worktree.

Note: `git fetch --all --tags` on remote workspaces can fail with "would
clobber existing tag" errors when multiple remotes have conflicting tags. Prefer
`git fetch origin <branch>` to fetch only what is needed.

### 8) Download model artifacts
Run:
```
MODEL_PATH_PREFIX="<model_path>" \
ENGINES_DIR="<engines_dir>" \
HF_CACHE_DIR="<hf_cache_dir>" \
  bash ./tests/cohere/scripts/download_checkpoints.sh <test_group> "<models>"
```
Notes:
- Requires `gsutil` access to the model bucket.
- For `TEST_GROUP=performance`, public HF repo IDs are skipped by `download_checkpoints.sh`, so you can still run the command for mixed lists of internal + public models.
- If every performance model is a public HF repo ID, this step is optional locally because vLLM will download those models on demand via the HF cache.
- For `coder-remote`, run the same command through
  `coder ssh <owner>/<workspace> -- bash -lc 'cd <repo_root> && ...'` when it
  is a simple one-liner, or prefer a stdin-fed `bash -s` script when the remote
  step already needs env vars, setup, or multiple shell commands.

### 9) Run tests

For Docker modes, pick image based on GPU type:
- NVIDIA: `vllm-nvidia:<tag>`
- AMD: `vllm-rocm:<tag>`

**Important**: The `setup_tests.sh` script will reinstall vLLM in editable mode using precompiled wheels from `/app/cohere/dist/`. This allows testing local source changes from the mounted workspace (`/vllm-workspace`) while using the precompiled C++/CUDA extensions from the Docker image.
Also note: `setup_tests.sh` may leave the current working directory at `/vllm-workspace/tests`, so use an explicit `cd` before invoking `run_tests.sh`.

**Performance (`TEST_GROUP=performance`)**: add `-e BENCHMARK_OUTPUT_LEN=100` or `1000` to match `build-and-bench.yaml` `benchmarks` values `perf_100` / `perf_1000`.
For public HF repo IDs, keep `MODELS` as the raw repo ID and set `TP_SIZE` explicitly. Example: `MODELS=CohereLabs/c4ai-command-r7b-12-2024` with `TP_SIZE=1`.

If `execution_target=coder-remote`, execute the same commands on the remote
workspace via:

```bash
coder ssh <owner>/<workspace> -- bash -lc 'cd <repo_root> && <command>'
```

For remote Docker flows, prefer sending one stdin-fed script to `bash -s`
instead of stitching together several nested `bash -lc` strings. This is
especially helpful when the script must both use the synced `repo_root` and
inject `HF_TOKEN`.

That means:
- `one-shot`: run the `docker run --rm ...` command on the remote workspace
- `iterative`: create and reuse the container on the remote workspace
- `current-env`: run the export + `run_tests.sh` flow on the remote workspace

---

#### 8a) One-shot mode (default)

A single `docker run --rm` that starts the container, runs setup + tests, and removes the container on exit.

**NVIDIA run**
```
docker run --rm \
  --name vllm-tests \
  <gpu_device_flags> \
  -v <output_dir>:/root/output \
  -v <repo_root>:/vllm-workspace \
  -v <engines_dir>:/root/engines \
  -v <hf_cache_dir>:/root/.cache/huggingface \
  --shm-size=256g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN=<hf_token> \
  -e GCLOUD_PROJECT=valued-sight-253418 \
  -e TEST_GROUP=<test_group> \
  -e TP_SIZE=<tp_size> \
  -e MODELS=<models> \
  -e GPU_TYPE=<gpu> \
  -e MODEL_PATH_PREFIX=<model_path> \
  <cuda_visible_devices_flag> \
  <docker_env_flags> \
  --entrypoint /bin/bash \
  <image> \
  -c "
    cd /vllm-workspace
    source ./tests/cohere/scripts/setup_tests.sh
    cd /vllm-workspace/tests
    df -h /dev/shm
    bash ./cohere/scripts/run_tests.sh
  "
```

Where `<gpu_device_flags>` and `<cuda_visible_devices_flag>` depend on whether
`cuda_visible_devices` is set:

| `cuda_visible_devices` | `<gpu_device_flags>` | `<cuda_visible_devices_flag>` |
|------------------------|----------------------|-------------------------------|
| not set (all GPUs) | `--device nvidia.com/gpu=all` | *(omit)* |
| e.g. `0,2` | `--device nvidia.com/gpu=0 --device nvidia.com/gpu=2` | `-e CUDA_VISIBLE_DEVICES=0,1` |

Note: when specific GPUs are requested via `--device`, the container sees them
re-indexed starting from 0. Set `CUDA_VISIBLE_DEVICES` to the re-indexed range
(`0,1,...,N-1` where N is the count of requested GPUs) so that CUDA and vLLM
see all exposed devices. If the user wants to further restrict within the
exposed set, they can override this.

**AMD run**
```
docker run --rm \
  --name vllm-tests \
  --device=/dev/kfd --device=/dev/dri \
  -v <output_dir>:/root/output \
  -v <repo_root>:/vllm-workspace \
  -v <engines_dir>:/root/engines \
  -v <hf_cache_dir>:/root/.cache/huggingface \
  --shm-size=256g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN=<hf_token> \
  -e GCLOUD_PROJECT=valued-sight-253418 \
  -e TEST_GROUP=<test_group> \
  -e TP_SIZE=<tp_size> \
  -e MODELS=<models> \
  -e GPU_TYPE=<gpu> \
  -e MODEL_PATH_PREFIX=<model_path> \
  <cuda_visible_devices_flag> \
  <docker_env_flags> \
  --entrypoint /bin/bash \
  <image> \
  -c "
    cd /vllm-workspace
    source ./tests/cohere/scripts/setup_tests.sh
    cd /vllm-workspace/tests
    df -h /dev/shm
    bash ./cohere/scripts/run_tests.sh
  "
```

AMD note: `--device=/dev/kfd --device=/dev/dri` always exposes all GPUs at the
kernel level (no per-GPU CDI like NVIDIA), so `CUDA_VISIBLE_DEVICES` is the only
way to restrict which GPUs are used. ROCm/HIP respects this env var through its
CUDA compatibility layer. `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` also
work if needed, but `CUDA_VISIBLE_DEVICES` is sufficient for PyTorch/vLLM.

Remind the user that `--rm` removes the container on exit.

---

#### 8b) Iterative mode

Keeps the container alive between runs so that setup (dependency install, editable install) is done once and subsequent test runs start immediately. Source changes on the host are visible instantly via the volume mount + editable install.

##### Why this works
- The workspace is bind-mounted (`-v <repo_root>:/vllm-workspace`), so host edits are live inside the container.
- `setup_tests.sh` installs vLLM in editable mode (`pip install -e . --no-deps`), so Python picks up source changes without reinstall.
- Only C++/CUDA extension changes require a new Docker image (extensions come from the precompiled wheel).

##### Step 1: Start persistent container

Start the container in detached mode with `sleep infinity` as the entrypoint. Do **not** use `--rm`.

Treat `vllm-tests` below as a placeholder. On remote or shared workspaces,
prefer a unique container name such as `vllm-tests-<test_group>-<gpu>` and only
remove or reuse the specific container you created, so you do not clobber an
unrelated developer container.

**NVIDIA**
```
docker run -d \
  --name vllm-tests \
  <gpu_device_flags> \
  -v <output_dir>:/root/output \
  -v <repo_root>:/vllm-workspace \
  -v <engines_dir>:/root/engines \
  -v <hf_cache_dir>:/root/.cache/huggingface \
  --shm-size=256g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN=<hf_token> \
  -e GCLOUD_PROJECT=valued-sight-253418 \
  -e GPU_TYPE=<gpu> \
  -e MODEL_PATH_PREFIX=<model_path> \
  <cuda_visible_devices_flag> \
  <docker_env_flags> \
  --entrypoint /bin/bash \
  <image> \
  -c "sleep infinity"
```

Use the same `<gpu_device_flags>` / `<cuda_visible_devices_flag>` rules as the
one-shot table above.

**AMD**
```
docker run -d \
  --name vllm-tests \
  --device=/dev/kfd --device=/dev/dri \
  -v <output_dir>:/root/output \
  -v <repo_root>:/vllm-workspace \
  -v <engines_dir>:/root/engines \
  -v <hf_cache_dir>:/root/.cache/huggingface \
  --shm-size=256g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN=<hf_token> \
  -e GCLOUD_PROJECT=valued-sight-253418 \
  -e GPU_TYPE=<gpu> \
  -e MODEL_PATH_PREFIX=<model_path> \
  <cuda_visible_devices_flag> \
  <docker_env_flags> \
  --entrypoint /bin/bash \
  <image> \
  -c "sleep infinity"
```

##### Step 2: Run setup (once)

Run `setup_tests.sh` inside the container. This installs dependencies and does the editable install. Only needs to run once per container lifecycle.

```
docker exec vllm-tests \
  bash -c "cd /vllm-workspace && source ./tests/cohere/scripts/setup_tests.sh"
```

Notes:
- The first remote setup can take several minutes if packages or editable
  installs need to be refreshed.
- In synced detached worktrees, `setup_tests.sh` may emit a
  `setuptools-scm was unable to detect version for /vllm-workspace` warning
  during `uv pip install -e . --no-deps`. If the script continues to export
  `PYTHONPATH` and reaches `/vllm-workspace/tests`, treat that warning alone as
  non-fatal.

##### Step 3: Run tests (repeatable)

Run tests via `docker exec`. Pass `TEST_GROUP`, `TP_SIZE`, and `MODELS` as `-e` flags so they can be changed between runs without restarting the container.

```
docker exec \
  -e TEST_GROUP=<test_group> \
  -e TP_SIZE=<tp_size> \
  -e MODELS=<models> \
  <cuda_visible_devices_flag> \
  vllm-tests \
  bash -c "
    cd /vllm-workspace
    source ./tests/cohere/scripts/setup_tests.sh
    cd /vllm-workspace/tests
    bash ./cohere/scripts/run_tests.sh
  "
```

Include `<cuda_visible_devices_flag>` (e.g. `-e CUDA_VISIBLE_DEVICES=0,1`) if
`cuda_visible_devices` was set. Even though `CUDA_VISIBLE_DEVICES` was passed
at container creation, each `docker exec` gets a fresh env, so re-pass it here.

Notes on re-sourcing `setup_tests.sh`:
- Each `docker exec` creates a new shell, so env vars from prior execs are lost.
- Re-sourcing `setup_tests.sh` is safe and fast on subsequent runs: `uv pip install` skips already-installed packages, system dep checks skip existing tools, and the editable install is a no-op. Typical overhead: a few seconds.
- This also re-applies hardware-specific env vars and CLI args (via `VLLM_HARDWARE_PROFILE_ARGS`) from `apply_hardware_profiles.py`.
- On remote workspaces, prefer monitoring the original long-running wrapper
  command output until setup reaches a steady state instead of repeatedly
  rebuilding nested remote inspection commands.

##### Step 4: Iterate

After making source changes on the host, repeat Step 3. No restart, no re-download, no rebuild needed.

To change test parameters (e.g. switch from `fast_check` to `lm_eval`), just change the `-e` flags in Step 3.

##### Step 5: Cleanup

When done iterating:
```
docker stop vllm-tests && docker rm vllm-tests
```

##### Interactive debugging

Attach a shell to the running container at any time:
```
docker exec -it vllm-tests /bin/bash
```

From inside, you can inspect logs, GPU state, or run individual test files directly:
```
cd /vllm-workspace/tests
python -m pytest <test_file> -v
```

---

#### 8c) Current env mode (no Docker)

Use this when the user wants to execute directly on the machine's current environment instead of an image.

##### Preconditions
- Run from `repo_root`.
- Python dependencies and runtime tooling must already be installed in the current environment.
- If needed, run setup manually first (for example, `pip install -e .`) before test execution.

##### Single run
```
cd <repo_root>
export HF_HOME="<hf_cache_dir>"
export HF_TOKEN="<hf_token>"
export GCLOUD_PROJECT=valued-sight-253418
export TEST_GROUP=<test_group>
export TP_SIZE=<tp_size>
export MODELS=<models>
export GPU_TYPE=<gpu>
export MODEL_PATH_PREFIX=<model_path>
export CUDA_VISIBLE_DEVICES=<cuda_visible_devices>
cd tests
bash ./cohere/scripts/run_tests.sh
```

Omit the `CUDA_VISIBLE_DEVICES` export when using all GPUs.

##### Re-run after source changes
- Re-run the same command block (or just update `TEST_GROUP` / `TP_SIZE` / `MODELS` exports and run `bash ./cohere/scripts/run_tests.sh` again).
- No container restart is needed.

### 10) Multi-TP runs
To sweep tensor parallel sizes, re-run the same `test_group` and `models` with
different `TP_SIZE` values (for example `1`, then `2`, then `4`). For feature
tests, use `TP_SIZE=0` once.

In iterative Docker mode, run Step 3 multiple times with different `-e TP_SIZE=` values.
In current-env mode, re-export `TP_SIZE` and re-run `bash ./cohere/scripts/run_tests.sh` for each value.

## Notes
- Feature tests ignore models/TP size; eval/perf tests require model support for the GPU/TP.
- GitHub `build-and-bench` supports public HF repo IDs only when dispatched with an explicit shared `:tpN` suffix, but local runs should omit the suffix and set `TP_SIZE` directly.
- If a step fails due to missing dependencies, prompt to install immediately and retry the step.
- Keep all prompts minimal and actionable.
- **Iterative mode** is recommended when developing/debugging tests or iterating on source changes. **One-shot mode** is better for clean CI-like runs.
- **Current env mode** is useful when Docker is unavailable or when the user explicitly wants to run against an already-prepared local environment.
- For remote Coder execution, reuse `/sync-coder-remotes` for auth, workspace
  discovery, and exact-commit repo sync instead of duplicating that logic here.

## Example prompt flow

### One-shot
1. "Detected GPU: h100. Use this GPU?" (or prompt if not detected)
2. GPU utilization check: "GPUs 0,1 are idle; GPUs 2,3 are in use. Use all GPUs or restrict to specific indices?" (skip if all idle)
3. "One-shot or iterative mode?"
4. "Which test_group, models, and TP_SIZE?"
5. "`HF_TOKEN` is not set in my shell. Please paste your HF token so I can export it."
6. "Use existing image (default), trigger build-and-push, or run on current env?"
7. "Paste image reference", "Trigger build-and-push now?", or "Proceed with current env?"
8. "Any extra docker env flags (e.g., `-e VAR=value`)?"
9. "`gsutil` missing. Install now? (y/n)"

### Iterative
1. "Detected GPU: h100. Use this GPU?"
2. GPU utilization check (same as one-shot step 2; skip if all idle)
3. "One-shot or iterative mode?" -> "iterative"
4. "Which test_group, models, and TP_SIZE for the first run?"
5. "`HF_TOKEN` is not set in my shell. Please paste your HF token so I can export it."
6. "Use existing image (default), trigger build-and-push, or run on current env?"
7. "Paste image reference", "Trigger build-and-push now?", or "Proceed with current env?"
8. Starting container... Running setup... Running tests...
9. (after tests complete) "Container `vllm-tests` is still running. Make your changes and say 'run again' or specify new parameters (e.g. `test_group=lm_eval TP_SIZE=2`)."
10. (user iterates) Re-running tests with updated source...
11. "Done iterating? I'll stop and remove the container."

### Current env
1. "Detected GPU: h100. Use this GPU?" (or prompt if not detected)
2. GPU utilization check (same as one-shot step 2; skip if all idle)
3. "One-shot, iterative, or current-env mode?" -> "current-env"
4. "Which test_group, models, and TP_SIZE?"
5. "`HF_TOKEN` is not set in my shell. Please paste your HF token so I can export it."
6. "Use existing image (default), trigger build-and-push, or run on current env?" -> "run on current env"
7. "Running directly in current environment with exported test vars."

### Remote workspace
1. "Run on this workspace or a remote Coder workspace?" -> "remote"
2. "Which `owner/workspace` should I use? If the repo lives elsewhere there, also give me `remote_repo_root`."
3. "I'll use `/sync-coder-remotes` setup first to verify `coder`, auth, workspace visibility, and exact-commit repo sync."
4. "Detected remote GPU: h100. Use this GPU?"
5. Remote GPU utilization check (same as one-shot step 2, run via `coder ssh`; skip if all idle)
6. "Which run_mode, test_group, models, and TP_SIZE?"
7. "`HF_TOKEN` is not set on the remote target. Please paste your HF token so I can inject it into the remote run."
8. "Use existing image, trigger build-and-push, or run on current env on the remote workspace?"
9. "Running the same `coder-ci-runner` flow on `coder ssh <owner>/<workspace>`."
