# Code Notes: Build and Packaging

## 1) Two-Stage Image Strategy (Base vLLM -> Cohere Overlay)

The Cohere branch introduces a layered build model:

- build base vLLM image first (`build-vllm-*` targets),
- then build Cohere overlay image (`build-cohere-*`) on top.

From `Makefile` and `docker/Dockerfile.cohere`, this enables:

- immutable base images tagged by commit SHA,
- fast iteration on Cohere-only dependencies and scripts,
- editable install of the local source while reusing prebuilt native extensions.

## 2) Wheel Handoff Contract

Critical contract:

- base build emits wheel artifacts under `/app/cohere/dist`,
- `Dockerfile.cohere` expects at least one wheel in that directory,
- editable install uses `VLLM_PRECOMPILED_WHEEL_LOCATION` for extension reuse.

This applies to **all** platform Dockerfiles:

- `docker/Dockerfile` (NVIDIA/CUDA) — wheel retained via multi-stage `COPY --from`.
- `docker/Dockerfile.rocm` (AMD/ROCm) — same pattern in the `final` stage (and `test`).
- `docker/Dockerfile.cpu` — `COPY --from=vllm-build /vllm-workspace/dist /app/cohere/dist` persists the wheel so the CI wheel-upload pipeline (`build-and-push.yaml`) can extract it with crane.

Why this matters:

- avoids recompiling heavy CUDA/ROCm extensions in the overlay stage,
- gives development ergonomics (`pip install -e .`) with production-like binaries,
- enables the CI wheel-upload flow for all platforms (NVIDIA, AMD, and CPU).

Failure mode:

- if wheel retention path changes in upstream Dockerfile stages, overlay build fails with "No precompiled wheel found".
- if the `COPY --from` step is dropped during a rebase of `Dockerfile.cpu`, the CPU wheel upload silently produces an empty artifact.

## 3) SCCACHE Backend Shift for Depot

In `docker/Dockerfile`, Cohere variant configures sccache for WebDAV instead of S3 variables:

- `SCCACHE_WEBDAV_ENDPOINT`
- `SCCACHE_WEBDAV_TOKEN`
- `SCCACHE_WEBDAV_KEY_PREFIX`

Intent:

- align build cache with Depot environment and multi-arch workflows.

Rebase risk:

- upstream sccache updates often touch the same area; partial cherry-picks can leave mixed env var sets and silently disable cache reuse.
- do not re-add `SCCACHE_BUCKET` / `SCCACHE_REGION` / `SCCACHE_ENDPOINT` in the `csrc-build` wheel step: sccache prefers S3 over WebDAV and Depot CI gets `403` on `vllm-build-sccache` without vLLM AWS creds (v0.21 merge regression; v0.19.1b03 was WebDAV-only).

## 4) CPU Build Divergence Points

`docker/Dockerfile.cpu` is an upstream-maintained file with a single Cohere-specific addition:

- `COPY --from=vllm-build /vllm-workspace/dist /app/cohere/dist` — persists the built wheel for extraction and upload to Artifact Registry.

This mirrors the wheel-handoff pattern in the GPU Dockerfiles. The rest of the file is upstream-owned.

Rebase risk:

- upstream refactors to the `vllm-build` stage or changes to the final-stage `WORKDIR` can silently break the `COPY` path.

## 5) ROCm Divergence Points

`docker/Dockerfile.rocm` adds Cohere-specific operational deltas:

- pinned base image digest for deterministic base behavior,
- apt source hotfix handling for AMD repo issues,
- export/copy of `vllm/v1` and wheel artifacts for downstream use.

Additionally, the final stage installs the RIXL Python wheel (built in the `build_rixl` stage) and RDMA
userspace libraries (`librdmacm1`, `libibverbs1`, `ibverbs-providers`, `ibverbs-utils`). This enables
prefill/decode (P/D) disaggregated serving via vLLM's `NixlConnector`, which imports the `nixl` Python
module from the RIXL wheel. The `test` stage already had this; the final stage was missing it.

These are less about model logic and more about build reproducibility and runtime image consistency.

## 6) Dependency and Formatting Deltas

Notable repo-level packaging/config changes:

- `requirements/common.txt`: remove hard pin on upstream `transformers`.
- `pyproject.toml`: exclude Cohere fixture text files from formatter.
- `CMakeLists.txt`: drop selected Marlin CUDA source entries (conflict-prone with upstream kernel work).

Interpretation:

- Cohere branch prioritizes compatibility with internal model/tool stacks over strict upstream dependency lockstep.

## 7) Change Hotspots and Practical Checks

High-conflict files:

- `docker/Dockerfile`
- `docker/Dockerfile.cohere`
- `docker/Dockerfile.cpu`
- `docker/Dockerfile.rocm`
- `Makefile`
- `requirements/common.txt`

Quick validation after rebase:

1. Build base image and verify wheel exists in `/app/cohere/dist`.
2. Build Cohere overlay and verify editable install succeeds.
3. Confirm `vllm` CLI entrypoint still works in final image.
4. Check sccache stats are non-zero in build logs when enabled.
