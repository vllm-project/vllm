# docker-bake-rocm.hcl - vLLM ROCm Docker build configuration
#
# This file lives in the vLLM repo at docker/docker-bake-rocm.hcl
# Equivalent of docker-bake.hcl for ROCm builds.
#
# Usage:
#   docker buildx bake -f docker/docker-bake-rocm.hcl              # Build test (default)
#   docker buildx bake -f docker/docker-bake-rocm.hcl final-rocm   # Build final image
#   docker buildx bake -f docker/docker-bake-rocm.hcl --print      # Show resolved config
#
# CI usage (with ci-rocm.hcl overlay from ci-infra):
#   docker buildx bake -f docker/docker-bake-rocm.hcl -f /tmp/ci-rocm.hcl test-rocm-ci

variable "MAX_JOBS" {
  # Empty string lets the Dockerfile fall back to $(nproc) via
  # MAX_JOBS="${MAX_JOBS:-$(nproc)}" in each RUN step, which uses all
  # available cores on whatever machine the build runs on.
  # Override with --set '*.args.max_jobs=8' for local builds on small machines.
  default = ""
}

variable "PYTORCH_ROCM_ARCH" {
  default = "gfx90a;gfx942;gfx950"
}

variable "COMMIT" {
  default = ""
}

# Content hash of ci_base-affecting files. Computed by ci-bake-rocm.sh and
# embedded as a label so future builds can compare without rebuilding.
variable "CI_BASE_CONTENT_HASH" {
  default = ""
}

# REMOTE_VLLM=0: use local source via Docker build context (ONBUILD COPY ./ vllm/)
# REMOTE_VLLM=1: clone from GitHub at VLLM_BRANCH (standalone builds without local source)
variable "REMOTE_VLLM" {
  default = "0"
}

variable "VLLM_BRANCH" {
  default = "main"
}

# CI_BASE_IMAGE: pre-built ci_base image for per-PR test builds.
# Defaults to the local "ci_base" stage for standalone/local builds.
# CI overrides this to "rocm/vllm-dev:ci_base" via environment variable.
variable "CI_BASE_IMAGE" {
  default = "rocm/vllm-dev:ci_base"
}

group "default" {
  targets = ["test-rocm"]
}

target "_common-rocm" {
  dockerfile = "docker/Dockerfile.rocm"
  context    = "."
  args = {
    max_jobs              = MAX_JOBS
    ARG_PYTORCH_ROCM_ARCH = PYTORCH_ROCM_ARCH
    REMOTE_VLLM           = REMOTE_VLLM
    VLLM_BRANCH           = VLLM_BRANCH
    CI_BASE_IMAGE         = CI_BASE_IMAGE
  }
}

target "_labels" {
  labels = {
    "org.opencontainers.image.source"      = "https://github.com/vllm-project/vllm"
    "org.opencontainers.image.vendor"      = "vLLM"
    "org.opencontainers.image.title"       = "vLLM ROCm"
    "org.opencontainers.image.description" = "vLLM: A high-throughput and memory-efficient inference and serving engine for LLMs (ROCm)"
    "org.opencontainers.image.licenses"    = "Apache-2.0"
    "org.opencontainers.image.revision"    = COMMIT
  }
  annotations = [
    "manifest:org.opencontainers.image.revision=${COMMIT}",
  ]
}

target "test-rocm" {
  inherits = ["_common-rocm", "_labels"]
  target   = "test"
  tags     = ["rocm/vllm:test"]
  output   = ["type=docker"]
}

# CI base image target — builds only the ci_base stage (RIXL, DeepEP,
# torchcodec, requirements, etc.). Used by the weekly scheduled build and
# the auto-rebuild trigger when requirements change in a PR.
target "ci-base-rocm" {
  inherits = ["_common-rocm", "_labels"]
  target   = "ci_base"
  labels   = {
    "vllm.ci_base.content_hash" = CI_BASE_CONTENT_HASH
  }
  tags     = ["rocm/vllm-dev:ci_base"]
  output   = ["type=docker"]
}

# Wheel export target - extracts the built vLLM wheel + test workspace
# to local disk. Used by CI to upload the wheel as a Buildkite artifact
# so test jobs can assemble images locally from ci_base + wheel instead
# of pulling the full large image from Docker Hub.
#
# Usage:
#   docker buildx bake -f docker/docker-bake-rocm.hcl export-wheel-rocm
#   # Creates ./wheel-export/*.whl, ./wheel-export/requirements/, etc.
#
# After a full bake build, BuildKit cache makes this nearly instant.
target "export-wheel-rocm" {
  inherits = ["_common-rocm"]
  target   = "export_vllm"
  output   = ["type=local,dest=./wheel-export"]
}

target "final-rocm" {
  inherits = ["_common-rocm", "_labels"]
  target   = "final"
  tags     = ["rocm/vllm:latest"]
  output   = ["type=docker"]
}
