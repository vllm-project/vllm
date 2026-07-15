# ci-rocm.hcl - CI-specific configuration for vLLM ROCm Docker builds
#
# This file lives in the vLLM repo at docker/ci-rocm.hcl so ROCm Docker
# build mechanics can evolve with Dockerfile.rocm and docker-bake-rocm.hcl.
# Used with: docker buildx bake -f docker/docker-bake-rocm.hcl -f docker/ci-rocm.hcl test-rocm-ci
#
# Registry cache: Docker Hub (rocm/vllm-ci-cache) is used exclusively.
# AMD build agents already have Docker Hub credentials (they push the test
# image to rocm/vllm-ci), so no additional credential setup is required.
# ROCm CI uses Docker Hub for BuildKit layer cache by default. A separate
# compiler cache can be enabled with USE_SCCACHE=1 when AMD provides a shared
# S3-compatible cache endpoint.

# CI metadata

variable "BUILDKITE_COMMIT" {
  default = ""
}

variable "BUILDKITE_BUILD_NUMBER" {
  default = ""
}

variable "BUILDKITE_BUILD_ID" {
  default = ""
}

variable "PARENT_COMMIT" {
  default = ""
}

# Merge-base of HEAD with main - provides a more stable cache fallback than
# parent commit for long-lived PRs. Mirrors the VLLM_MERGE_BASE_COMMIT
# pattern used in the shared ci.hcl file. Auto-computed by ci-bake-rocm.sh
# when unset.
variable "VLLM_MERGE_BASE_COMMIT" {
  default = ""
}

# Bridge to vLLM's COMMIT variable for OCI labels
variable "COMMIT" {
  default = BUILDKITE_COMMIT
}

# Image tags (set by CI)

variable "IMAGE_TAG" {
  default = ""
}

variable "IMAGE_TAG_LATEST" {
  default = ""
}

# ROCm-specific GPU architecture targets

variable "PYTORCH_ROCM_ARCH" {
  default = "gfx90a;gfx942;gfx950"
}

# Pre-built CI base image (Tier 1). Per-PR builds pull this instead of
# rebuilding RIXL/DeepEP/torchcodec from scratch. The ci_base stage in
# Dockerfile.rocm inherits from base, so CI_BASE_IMAGE only affects the test
# stage and is irrelevant when building --target ci_base itself.
variable "CI_BASE_IMAGE" {
  default = "rocm/vllm-dev:ci_base"
}

# Leave CI_MAX_JOBS empty so the Dockerfile falls back to $(nproc) and uses
# the full builder parallelism. Operators can still override this per build.
variable "CI_MAX_JOBS" {
  default = ""
}

# Upstream dependency commit pins -- extracted from Dockerfile.rocm by
# ci-bake-rocm.sh at build time. Empty defaults are safe: the cache
# functions produce no entries when the variable is empty.
variable "RIXL_BRANCH" {
  default = ""
}

variable "UCX_BRANCH" {
  default = ""
}

variable "ROCSHMEM_BRANCH" {
  default = ""
}

variable "DEEPEP_BRANCH" {
  default = ""
}

variable "RIXL_CACHE_KEY" {
  default = ""
}

variable "ROCSHMEM_CACHE_KEY" {
  default = ""
}

variable "DEEPEP_CACHE_KEY" {
  default = ""
}

# Docker Hub registry cache for AMD builds.
#
# A separate repo (rocm/vllm-ci-cache) is used for BuildKit layer cache.
# Final-image cache exports use mode=min to reduce the volume of data pushed.
# Source-scoped csrc cache exports default to mode=max so fresh workers can
# recover more of the native build graph when ROCm extension inputs change.
# NOTE: mode=min still includes all layers referenced by the final image
# manifest, including inherited base layers (~7.25GB ROCm runtime).
# Docker Hub auto-creates the repo on first push.
#
# Final-image cache stays commit-scoped. Branch-to-branch reuse for the test
# image comes from importing the parent and merge-base commit cache refs.
#
# The source-scoped native cache is exported both per-commit and per-branch so
# ROCm extension rebuilds are shareable within the same commit reruns and across
# consecutive commits on the same branch without depending on a single global
# latest tag.

variable "DOCKERHUB_CACHE_REPO" {
  default = "rocm/vllm-ci-cache"
}

variable "DOCKERHUB_CACHE_TO" {
  default = ""
}

variable "ROCM_CACHE_BRANCH_TAG" {
  default = ""
}

variable "ROCM_CACHE_UPSTREAM_BRANCH_TAG" {
  default = ""
}

variable "ROCM_CSRC_CACHE_TO_MODE" {
  default = "max"
}

variable "ROCM_RUST_CACHE_TO_MODE" {
  default = "max"
}

variable "ROCM_FINAL_CACHE_TO_MODE" {
  default = "min"
}

# Functions

function "get_cache_from_rocm" {
  params = []
  result = compact([
    # Exact commit hit - fastest cache on re-runs of the same commit
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-${BUILDKITE_COMMIT}" : "",
    # Parent commit - useful cache for incremental changes
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-${PARENT_COMMIT}" : "",
    # Merge-base with main - stable fallback for long-lived or rebased PRs;
    # maps to a real main-branch commit whose cache layers are likely warm
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    # Import the source-scoped native build cache as well so builds whose
    # Python/package layers changed can still reuse compiled ROCm objects.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${BUILDKITE_COMMIT}" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${PARENT_COMMIT}" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
    # Import the source-scoped Rust frontend cache so non-Rust changes do not
    # force a fresh cargo release build.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${BUILDKITE_COMMIT}" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${PARENT_COMMIT}" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
    # Branch-scoped full image cache - fallback when parent-commit cache is evicted
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
  ])
}

function "get_cache_to_rocm" {
  params = []
  result = compact([
    # Commit-scoped cache for exact re-runs.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-${BUILDKITE_COMMIT},mode=${ROCM_FINAL_CACHE_TO_MODE}" : "",
    # Branch-scoped cache so later commits on the same branch can reuse the full
    # image layers when the parent-commit cache is evicted. Unlike the old
    # rocm-latest tag (which caused duplicate exporter 400s), this is per-branch.
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${ROCM_FINAL_CACHE_TO_MODE}" : "",
  ])
}

function "get_cache_from_rocm_csrc" {
  params = []
  result = compact([
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${BUILDKITE_COMMIT}" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${PARENT_COMMIT}" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
  ])
}

function "get_cache_to_rocm_csrc" {
  params = []
  result = compact([
    # Export the exact-commit native cache for same-commit reruns.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-${BUILDKITE_COMMIT},mode=${ROCM_CSRC_CACHE_TO_MODE}" : "",
    # Export the branch-scoped native cache so later commits on the same branch
    # can reuse compiled ROCm objects even when the exact parent cache is absent.
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:csrc-rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${ROCM_CSRC_CACHE_TO_MODE}" : "",
  ])
}

function "get_cache_from_rocm_rust" {
  params = []
  result = compact([
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${BUILDKITE_COMMIT}" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${PARENT_COMMIT}" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${VLLM_MERGE_BASE_COMMIT}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_BRANCH_TAG}" : "",
    ROCM_CACHE_UPSTREAM_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_UPSTREAM_BRANCH_TAG}" : "",
  ])
}

function "get_cache_to_rocm_rust" {
  params = []
  result = compact([
    # Export exact-commit and branch-scoped Rust caches. A content-addressed
    # cache ref is appended by ci-bake-rocm.sh when that wrapper is used.
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-${BUILDKITE_COMMIT},mode=${ROCM_RUST_CACHE_TO_MODE}" : "",
    ROCM_CACHE_BRANCH_TAG != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rust-rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${ROCM_RUST_CACHE_TO_MODE}" : "",
  ])
}

# Cache functions for upstream dependency stages (RIXL/UCX, ROCShmem, DeepEP).
# These stages are pinned to specific upstream commit hashes, so cache keys use
# those hashes rather than the Buildkite commit. This means the cache persists
# across all vLLM commits as long as the upstream dependency pins don't change.

function "get_cache_from_rocm_deps" {
  params = []
  result = compact([
    RIXL_CACHE_KEY != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rixl-rocm-${RIXL_CACHE_KEY}" : (RIXL_BRANCH != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rixl-rocm-${RIXL_BRANCH}-ucx-${UCX_BRANCH}" : ""),
    ROCSHMEM_CACHE_KEY != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocshmem-rocm-${ROCSHMEM_CACHE_KEY}" : (ROCSHMEM_BRANCH != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocshmem-rocm-${ROCSHMEM_BRANCH}" : ""),
    DEEPEP_CACHE_KEY != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:deepep-rocm-${DEEPEP_CACHE_KEY}" : (DEEPEP_BRANCH != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:deepep-rocm-${DEEPEP_BRANCH}-rocshmem-${ROCSHMEM_BRANCH}" : ""),
  ])
}

function "get_cache_to_rocm_rixl" {
  params = []
  result = compact([
    RIXL_CACHE_KEY != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rixl-rocm-${RIXL_CACHE_KEY},mode=min" : (RIXL_BRANCH != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rixl-rocm-${RIXL_BRANCH}-ucx-${UCX_BRANCH},mode=min" : ""),
  ])
}

function "get_cache_to_rocm_rocshmem" {
  params = []
  result = compact([
    ROCSHMEM_CACHE_KEY != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocshmem-rocm-${ROCSHMEM_CACHE_KEY},mode=min" : (ROCSHMEM_BRANCH != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:rocshmem-rocm-${ROCSHMEM_BRANCH},mode=min" : ""),
  ])
}

function "get_cache_to_rocm_deepep" {
  params = []
  result = compact([
    DEEPEP_CACHE_KEY != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:deepep-rocm-${DEEPEP_CACHE_KEY},mode=min" : (DEEPEP_BRANCH != "" ? "type=registry,ref=${DOCKERHUB_CACHE_REPO}:deepep-rocm-${DEEPEP_BRANCH}-rocshmem-${ROCSHMEM_BRANCH},mode=min" : ""),
  ])
}

# CI targets

target "_ci-rocm" {
  annotations = [
    "manifest:vllm.buildkite.build_number=${BUILDKITE_BUILD_NUMBER}",
    "manifest:vllm.buildkite.build_id=${BUILDKITE_BUILD_ID}",
  ]
  args = {
    ARG_PYTORCH_ROCM_ARCH = PYTORCH_ROCM_ARCH
    CI_BASE_IMAGE         = CI_BASE_IMAGE
    max_jobs              = CI_MAX_JOBS
  }
}

target "test-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm", "_labels"]
  target     = "test"
  cache-from = get_cache_from_rocm()
  cache-to   = get_cache_to_rocm()
  tags = compact([
    IMAGE_TAG,
    IMAGE_TAG_LATEST,
  ])
  output = ["type=registry"]
}

# Cache-only target for the source-scoped ROCm native build stage.
# This persists the csrc-build stage in the registry cache even though the
# final test image only consumes it indirectly while packaging the wheel.
target "csrc-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "csrc-build"
  cache-from = get_cache_from_rocm_csrc()
  cache-to   = get_cache_to_rocm_csrc()
  output     = ["type=cacheonly"]
}

# Cache-only target for the Rust frontend build stage. Final-image cache
# exports use mode=min and do not reliably persist intermediate cargo layers,
# so Rust gets its own source-scoped cache target.
target "rust-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "rust-build"
  cache-from = get_cache_from_rocm_rust()
  cache-to   = get_cache_to_rocm_rust()
  output     = ["type=cacheonly"]
}

# Keep wheel export on the same CI graph as the test image build so the
# shared build_vllm/export_vllm stages resolve identically within one bake
# invocation. Without this, export-wheel-rocm uses the plain local target
# args while test-rocm-ci uses CI-only args, which can lead to separate
# cache lineages and inconsistent export_vllm results.
target "export-wheel-rocm" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "export_vllm"
  cache-from = get_cache_from_rocm()
  cache-to   = get_cache_to_rocm()
  output     = ["type=local,dest=./wheel-export"]
}

# Artifact-only vLLM build. GPU test jobs consume this artifact on top of
# ci_base, avoiding a per-commit multi-GB image push/pull.
group "test-rocm-ci-with-artifacts" {
  targets = ["rust-rocm-ci", "csrc-rocm-ci", "export-wheel-rocm"]
}

# Full test image + wheel export. Kept for fallback/debugging when a pushed
# per-commit image is useful.
group "test-rocm-ci-with-wheel" {
  targets = ["rust-rocm-ci", "csrc-rocm-ci", "test-rocm-ci", "export-wheel-rocm"]
}

# Image tags for the ci_base build. ci-bake-rocm.sh rewrites CI_BASE_IMAGE_TAG
# to the primary tag for this build. Builds always publish a content-scoped tag
# when the ci_base content hash is available. Builds with BUILDKITE_COMMIT also
# publish a commit-scoped tag, either as the primary tag or an additional alias.
# NIGHTLY=1 builds on the stable branch can additionally set
# CI_BASE_IMAGE_TAG_STABLE to refresh rocm/vllm-dev:ci_base.
variable "CI_BASE_IMAGE_TAG" {
  default = "rocm/vllm-dev:ci_base"
}

# Supplemental tags only. ci-bake-rocm.sh leaves these empty when the same ref
# is already the primary CI_BASE_IMAGE_TAG.
variable "CI_BASE_IMAGE_TAG_COMMIT_EXTRA" {
  default = ""
}

variable "CI_BASE_IMAGE_TAG_CONTENT_EXTRA" {
  default = ""
}

variable "CI_BASE_IMAGE_TAG_STABLE" {
  default = ""
}

# Cache-only targets for upstream dependency stages. These persist each stage
# in the registry cache keyed by its upstream commit hash. When ci_base rebuilds
# (e.g., requirements change), these stages are cache hits if their upstream
# pins haven't changed -- saving ~35min of compilation.
target "rixl-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "build_rixl"
  cache-from = get_cache_from_rocm_deps()
  cache-to   = get_cache_to_rocm_rixl()
  output     = ["type=cacheonly"]
}

target "rocshmem-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "build_rocshmem"
  cache-from = get_cache_from_rocm_deps()
  cache-to   = get_cache_to_rocm_rocshmem()
  output     = ["type=cacheonly"]
}

target "deepep-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm"]
  target     = "build_deepep"
  cache-from = get_cache_from_rocm_deps()
  cache-to   = get_cache_to_rocm_deepep()
  output     = ["type=cacheonly"]
}

# Builds only the ci_base stage (RIXL, DeepEP, torchcodec, etc.)
# Invoked by the ensure-ci-base step when the content hash of ci_base-affecting
# files drifts from the remote image label. Per-PR builds then pull the result
# as CI_BASE_IMAGE instead of rebuilding those slow layers on every commit.
# Uses inline cache metadata on the ci_base image itself instead of exporting a
# separate registry cache artifact.
target "ci-base-rocm-ci" {
  inherits   = ["_common-rocm", "_ci-rocm", "_labels"]
  target     = "ci_base"
  cache-from = concat(
    compact([
      CI_BASE_IMAGE_TAG != "" ? "type=registry,ref=${CI_BASE_IMAGE_TAG}" : "",
      CI_BASE_IMAGE_TAG_COMMIT_EXTRA != "" ? "type=registry,ref=${CI_BASE_IMAGE_TAG_COMMIT_EXTRA}" : "",
      CI_BASE_IMAGE_TAG_CONTENT_EXTRA != "" ? "type=registry,ref=${CI_BASE_IMAGE_TAG_CONTENT_EXTRA}" : "",
      CI_BASE_IMAGE_TAG_STABLE != "" ? "type=registry,ref=${CI_BASE_IMAGE_TAG_STABLE}" : "",
    ]),
    # Import upstream dependency caches so RIXL/ROCShmem/DeepEP stages
    # are cache hits even when ci_base itself needs rebuilding.
    get_cache_from_rocm_deps(),
  )
  cache-to = ["type=inline"]
  tags     = compact([CI_BASE_IMAGE_TAG, CI_BASE_IMAGE_TAG_COMMIT_EXTRA, CI_BASE_IMAGE_TAG_CONTENT_EXTRA, CI_BASE_IMAGE_TAG_STABLE])
  output   = ["type=registry"]
}

# Group for ci_base builds -- exports dependency stage caches alongside the
# ci_base image so future rebuilds can reuse them independently.
group "ci-base-rocm-ci-with-deps" {
  targets = ["rixl-rocm-ci", "rocshmem-rocm-ci", "deepep-rocm-ci", "ci-base-rocm-ci"]
}
