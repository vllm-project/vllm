# build-and-push workflow overview

This repository includes a reusable GitHub Actions workflow at `.github/workflows/build-and-push.yaml` that builds and publishes vLLM Docker images. It is designed to be called by other workflows (for example, `.github/workflows/build-and-test.yaml`, `.github/workflows/build-and-eval.yaml`, `.github/workflows/build-and-bench.yaml`, and `.github/workflows/nightly-benchmark.yaml`).

## What it does

- Builds Docker images for multiple GPU providers (NVIDIA and AMD).
- Tags images with the appropriate git reference or commit SHA.
- Publishes images to the configured container registry.
- Produces Cohere images where vLLM is installed in editable mode for both NVIDIA and AMD builds.
- Extracts Python wheels from built images and uploads them to Google Artifact Registry.

## How it is triggered

- Workflow calls from other workflows (`workflow_call`).
- Manual invocations (`workflow_dispatch`).
- Tag pushes matching release conventions (for example, `v*.*.*`).

## Inputs

These are the supported inputs, grouped by trigger type. The workflow uses `inputs.git_ref` when provided, otherwise it falls back to `github.ref_name`. The `build-and-test`, `build-and-eval`, and `build-and-bench` entry workflows use their triggering ref when calling this workflow.

### `workflow_dispatch`

| Input | Type | Default | Details |
| --- | --- | --- | --- |
| `git_ref` | string | current branch | Git ref to check out (branch, tag, or SHA). |
| `force_build` | boolean | `false` | When `true`, rebuilds even if the image for the commit already exists. |
| `upload_wheels` | boolean | `false` | When `true`, runs the wheels upload job. |
| `use_precompiled` | boolean | `false` | When `true`, sets `VLLM_USE_PRECOMPILED=1` for the build. |
| `sccache_key_prefix` | string | `vllm-cohere-build-sccache-5` | Prefix for the sccache WebDAV key used by Depot builds. |
| `custom_tag` | string | (none) | Additional image tag to apply in addition to the commit SHA and optional push tag. |
| `incremental_build` | boolean | `true` | When `true`, uses the latest available base image for the current provider (NVIDIA or AMD/ROCm) to speed up builds. |
| `torch_cuda_arch_list` | string | `8.0 9.0 10.0+PTX` | Sets `TORCH_CUDA_ARCH_LIST` for NVIDIA builds. |
