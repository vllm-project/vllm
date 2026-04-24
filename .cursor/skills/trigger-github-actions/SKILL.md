---
name: trigger-github-actions
description: Trigger GitHub Actions workflows (build-and-push, build-and-test, build-and-eval, build-and-bench) with pre-flight checks. Verifies git sync, checks gh CLI, and triggers workflows. Use when the user wants to run CI/CD pipelines or build/test Docker images.
---

# Trigger GitHub Actions Workflows

Trigger `build-and-push`, `build-and-test`, `build-and-eval`, or `build-and-bench` with the required pre-flight checks and repo scoping.

## Always Do First

1. Determine the target ref.
   Default to the current branch, but if the user explicitly asks for another branch, tag, or SHA, use that exact ref for `gh workflow run --ref`.
2. Verify git sync for branch-based runs.
   If the ref is the current branch, confirm `HEAD` matches `origin/<branch>` before triggering.
   If the user asks for a tag or SHA, skip the branch sync comparison and just use the requested ref.
3. Verify GitHub CLI access.
   `gh` must be installed and authenticated.
4. Scope every GitHub command explicitly.
   Always pass `-R cohere-ai/vllm-cohere`.
5. Confirm the final workflow inputs with the user before triggering.

## Shared Conventions

- `gpu`: `all`, `h100`, `mi300x`, `a100`, `b200`, or `gb200`
- `models`: comma-separated list; entries may use `model:tpN`
- There is no separate `tp_size` workflow input; TP is encoded in `models`

### Model Rules

- `build-and-eval` expects map-backed internal model ids. If `:tpN` is omitted, TP comes from each model's `recommended_tp` in `tests/cohere/configs/tp_model_map.json`
- `build-and-bench` supports both internal model ids and public Hugging Face repo ids
- Public HF models are only valid on `build-and-bench` and must include an explicit suffix such as `CohereLabs/c4ai-command-r7b-12-2024:tp1`
- `model_path` only applies to internal GCS-backed checkpoints; public HF repo ids are loaded directly by vLLM
- When triggering `build-and-bench` with a public HF model, remind the user that they may need a `max-model-len` override in `hardware_profiles_override` if the default profile value exceeds the model's `max_position_embeddings` / `model_max_length`

### `hardware_profiles_override` Rule

- `hardware_profiles_override` replaces the entire `tests/cohere/configs/hardware_profiles.yaml` file for that run
- Do not send only a partial YAML fragment; send the full file contents with your edits applied
- Use this only for advanced perf or test debugging

Example full-file override:

```yaml
profiles:
  - name: vllm-default
    when: server.type == "vllm"
    args:
      gpu-memory-utilization: "0.95"
      enable-chunked-prefill: ""
  - name: vllm-mi300x
    when: server.type == "vllm" && matches(gpu.name, "mi300x")
    args:
      swap-space: "64"
```

## Pre-flight Commands

### Check Git Sync

Use this when the target ref is the current branch:

```bash
git fetch origin
BRANCH=$(git branch --show-current)
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" != "$REMOTE" ]; then
  echo "Local branch is not in sync with origin/$BRANCH."
  echo "Local:  $LOCAL"
  echo "Remote: $REMOTE"
  exit 1
fi
```

If out of sync, stop and ask the user to sync before triggering.

### Check `gh`

```bash
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is not installed."
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated."
  echo "Run: gh auth login"
  exit 1
fi
```

If `gh` is missing, provide the install command for the current OS and wait for the user to confirm.
If `gh` is unauthenticated, ask the user to run `gh auth login` and confirm when done.

## Workflow Inputs

### `build-and-push`

Required:
- No extra required user inputs beyond the target ref

Common optional inputs:
- `force_build`
- `upload_wheels`
- `use_precompiled`
- `sccache_key_prefix`
- `custom_tag`
- `incremental_build`
- `torch_cuda_arch_list`
- `providers` (JSON array, e.g. `["nvidia","amd","cpu"]`; defaults to all three)

Example:

```bash
TARGET_REF=$(git branch --show-current)

gh workflow run build-and-push.yaml \
  -R cohere-ai/vllm-cohere \
  --ref "$TARGET_REF" \
  --field git_ref="$TARGET_REF"
```

### `build-and-test`

Required:
- `gpu`

Common optional inputs:
- `docker_image_tag`
- `model_path`
- `features`
- `hardware_profiles_override`
- `result_upload_branch`

Example:

```bash
TARGET_REF=$(git branch --show-current)

gh workflow run build-and-test.yaml \
  -R cohere-ai/vllm-cohere \
  --ref "$TARGET_REF" \
  --field gpu=h100 \
  --field features=fast_check
```

### `build-and-eval`

Required:
- `gpu`

Common optional inputs:
- `docker_image_tag`
- `model_path`
- `models`
- `evaluations`
- `hardware_profiles_override`
- `result_upload_branch`

Example (bee eval on one model with explicit TP):

```bash
TARGET_REF=$(git branch --show-current)

gh workflow run build-and-eval.yaml \
  -R cohere-ai/vllm-cohere \
  --ref "$TARGET_REF" \
  --field gpu=h100 \
  --field evaluations=bee_eval \
  --field models=command-r7b_fp8:tp1
```

### `build-and-bench`

Required:
- `gpu`

Common optional inputs:
- `docker_image_tag`
- `model_path`
- `models`
- `benchmarks`
- `hardware_profiles_override`
- `result_upload_branch`

Example with internal model:

```bash
TARGET_REF=$(git branch --show-current)

gh workflow run build-and-bench.yaml \
  -R cohere-ai/vllm-cohere \
  --ref "$TARGET_REF" \
  --field gpu=all \
  --field benchmarks=perf_1000 \
  --field models=c4-25a218t_fp8:tp4
```

Example with public HF model:

```bash
TARGET_REF=$(git branch --show-current)

gh workflow run build-and-bench.yaml \
  -R cohere-ai/vllm-cohere \
  --ref "$TARGET_REF" \
  --field gpu=h100 \
  --field benchmarks=perf_100 \
  --field models=CohereLabs/c4ai-command-r7b-12-2024:tp1
```

## Triggering Checklist

Before running `gh workflow run`, confirm:

- workflow file name
- target ref
- repo slug `cohere-ai/vllm-cohere`
- all non-default inputs
- whether `hardware_profiles_override` is a full-file YAML replacement
- for public HF bench runs, whether a `max-model-len` override is needed

## Error Handling

- Git out of sync: stop and ask the user to sync
- Missing `gh`: stop, provide install instructions, and wait
- Unauthenticated `gh`: stop and ask for `gh auth login`
- Workflow trigger fails with 404: retry with explicit `-R cohere-ai/vllm-cohere`
- Public HF model without `:tpN` on bench: stop and ask for an explicit suffix such as `:tp1`
- Public HF model on `build-and-eval`: stop and explain that direct HF repo ids are only supported on `build-and-bench`
- Partial `hardware_profiles_override`: stop and ask for a full replacement YAML

## After Triggering

After a successful trigger:

1. Confirm the workflow was dispatched
2. Print back the effective workflow inputs you sent, including:
   - workflow file
   - target ref
   - repo slug `cohere-ai/vllm-cohere`
   - every non-default input field/value
   - whether `hardware_profiles_override` was provided, and if so, a concise summary of the edited keys
3. Fetch the newest matching run and share the run URL
4. Offer to monitor status with the `check-github-actions-status` skill
