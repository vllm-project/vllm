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
  default = "8.0 8.9 9.0 10.0"
}

variable "COMMIT" {
  default = ""
}

# Groups

group "default" {
  targets = ["openai"]
}

# Base targets

target "_common" {
  dockerfile = "docker/Dockerfile"
  context    = "."
  args = {
    max_jobs             = MAX_JOBS
    nvcc_threads         = NVCC_THREADS
    torch_cuda_arch_list = TORCH_CUDA_ARCH_LIST
  }
}

target "_labels" {
  labels = {
    "org.opencontainers.image.source"      = "https://github.com/vllm-project/vllm"
    "org.opencontainers.image.vendor"      = "vLLM"
    "org.opencontainers.image.title"       = "vLLM"
    "org.opencontainers.image.description" = "vLLM: A high-throughput and memory-efficient inference and serving engine for LLMs"
    "org.opencontainers.image.licenses"    = "Apache-2.0"
    "org.opencontainers.image.revision"    = COMMIT
  }
  annotations = [
      "index,manifest:org.opencontainers.image.revision=${COMMIT}",
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
