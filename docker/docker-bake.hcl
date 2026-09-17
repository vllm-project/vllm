# docker-bake.hcl - vLLM Docker build configuration
#
# This file lives in vLLM repo at docker/docker-bake.hcl
#
# Usage:
#   cd docker && docker buildx bake        # Build default target (openai)
#   cd docker && docker buildx bake test   # Build test target
#   docker buildx bake --print             # Show resolved config
#
# Reference: https://docs.docker.com/build/bake/reference/

# Build configuration

variable "MAX_JOBS" {
  default = 16
}

variable "NVCC_THREADS" {
  default = 8
}

variable "TORCH_CUDA_ARCH_LIST" {
  default = "8.0 8.9 9.0 10.0 11.0 12.0"
}

variable "COMMIT" {
  default = ""
}

variable "VLLM_BUILD_COMMIT" {
  default = "unknown"
}

variable "VLLM_BUILD_PIPELINE" {
  default = "local"
}

variable "VLLM_BUILD_URL" {
  default = ""
}

variable "VLLM_IMAGE_TAG" {
  default = "local/vllm-openai:dev"
}

# Groups

group "default" {
  targets = ["openai"]
}

group "all" {
  targets = ["openai", "openai-ubuntu2404"]
}

# Base targets

target "_common" {
  dockerfile = "docker/Dockerfile"
  context    = "."
  args = {
    max_jobs             = MAX_JOBS
    nvcc_threads         = NVCC_THREADS
    torch_cuda_arch_list = TORCH_CUDA_ARCH_LIST
    VLLM_BUILD_COMMIT    = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    VLLM_BUILD_PIPELINE  = VLLM_BUILD_PIPELINE
    VLLM_BUILD_URL       = VLLM_BUILD_URL
    VLLM_IMAGE_TAG       = VLLM_IMAGE_TAG
  }
}

target "_labels" {
  labels = {
    "org.opencontainers.image.source"      = "https://github.com/vllm-project/vllm"
    "org.opencontainers.image.vendor"      = "vLLM"
    "org.opencontainers.image.title"       = "vLLM"
    "org.opencontainers.image.description" = "vLLM: A high-throughput and memory-efficient inference and serving engine for LLMs"
    "org.opencontainers.image.licenses"    = "Apache-2.0"
    "org.opencontainers.image.revision"    = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    "org.opencontainers.image.version"     = VLLM_IMAGE_TAG
    "org.opencontainers.image.url"         = VLLM_BUILD_URL
    "ai.vllm.build.commit"                 = VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")
    "ai.vllm.build.pipeline"               = VLLM_BUILD_PIPELINE
    "ai.vllm.build.url"                    = VLLM_BUILD_URL
    "ai.vllm.image.tag"                    = VLLM_IMAGE_TAG
  }
  annotations = [
    "index,manifest:org.opencontainers.image.revision=${VLLM_BUILD_COMMIT != "unknown" ? VLLM_BUILD_COMMIT : (COMMIT != "" ? COMMIT : "unknown")}",
  ]
}

# Build targets

target "test" {
  inherits = ["_common", "_labels"]
  target   = "test"
  tags     = ["vllm:test"]
  output   = ["type=docker"]
}

target "openai" {
  inherits = ["_common", "_labels"]
  target   = "vllm-openai"
  tags     = ["vllm:openai"]
  output   = ["type=docker"]
}

# Ubuntu 24.04 targets

target "test-ubuntu2404" {
  inherits = ["_common", "_labels"]
  target   = "test"
  tags     = ["vllm:test-ubuntu24.04"]
  args = {
    UBUNTU_VERSION          = "24.04"
    GDRCOPY_OS_VERSION      = "Ubuntu24_04"
  }
  output = ["type=docker"]
}

target "openai-ubuntu2404" {
  inherits = ["_common", "_labels"]
  target   = "vllm-openai"
  tags     = ["vllm:openai-ubuntu24.04"]
  args = {
    UBUNTU_VERSION          = "24.04"
    GDRCOPY_OS_VERSION      = "Ubuntu24_04"
  }
  output = ["type=docker"]
}
