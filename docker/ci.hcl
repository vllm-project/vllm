# ci.hcl - CI-specific configuration for vLLM Docker builds
#
# This file lives in ci-infra repo at docker/ci.hcl
# Used with: docker buildx bake -f docker-bake.hcl -f ci.hcl test-ci
#
# Contains CI infrastructure config that shouldn't be in the vLLM repo:
# - Registry URLs
# - Cache locations
# - sccache config
# - CI-specific build args

# Registries

variable "REGISTRY" {
  default = "936637512419.dkr.ecr.us-east-1.amazonaws.com"
}

variable "PUBLIC_REGISTRY" {
  default = "public.ecr.aws/q9t5s3a7"
}

# sccache configuration

variable "USE_SCCACHE" {
  default = 1
}

variable "SCCACHE_BUCKET_NAME" {
  default = "vllm-build-sccache"
}

variable "SCCACHE_REGION_NAME" {
  default = "us-west-2"
}

variable "SCCACHE_S3_NO_CREDENTIALS" {
  default = 0
}

# CI build args

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

# Bridge to vLLM's COMMIT variable for OCI labels
variable "COMMIT" {
  default = BUILDKITE_COMMIT
}

variable "VLLM_USE_PRECOMPILED" {
  default = "0"
}

variable "VLLM_MERGE_BASE_COMMIT" {
  default = ""
}

# Image tags (set by CI)

variable "IMAGE_TAG" {
  default = ""
}

variable "IMAGE_TAG_LATEST" {
  default = ""
}

# Cache configuration

variable "CACHE_FROM" {
  default = ""
}

variable "CACHE_FROM_BASE" {
  default = ""
}

variable "CACHE_FROM_MAIN" {
  default = ""
}

variable "CACHE_TO" {
  default = ""
}

# Functions

function "get_cache_from" {
  params = []
  result = compact([
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${REGISTRY}/vllm-ci-test-cache:${BUILDKITE_COMMIT},mode=max" : "",
    PARENT_COMMIT != "" ? "type=registry,ref=${REGISTRY}/vllm-ci-test-cache:${PARENT_COMMIT},mode=max" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${REGISTRY}/vllm-ci-test-cache:${VLLM_MERGE_BASE_COMMIT},mode=max" : "",
    CACHE_FROM != "" ? "type=registry,ref=${CACHE_FROM},mode=max" : "",
    CACHE_FROM_BASE != "" ? "type=registry,ref=${CACHE_FROM_BASE},mode=max" : "",
    CACHE_FROM_MAIN != "" ? "type=registry,ref=${CACHE_FROM_MAIN},mode=max" : "",
  ])
}

function "get_cache_to" {
  params = []
  result = compact([
    CACHE_TO != "" ? "type=registry,ref=${CACHE_TO},mode=max,compression=zstd" : "",
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${REGISTRY}/vllm-ci-test-cache:${BUILDKITE_COMMIT},mode=max,compression=zstd" : "",
  ])
}

# CI targets

target "_ci" {
  annotations = [
    "index,manifest:vllm.buildkite.build_number=${BUILDKITE_BUILD_NUMBER}",
    "index,manifest:vllm.buildkite.build_id=${BUILDKITE_BUILD_ID}",
  ]
  args = {
    buildkite_commit          = BUILDKITE_COMMIT
    USE_SCCACHE               = USE_SCCACHE
    SCCACHE_BUCKET_NAME       = SCCACHE_BUCKET_NAME
    SCCACHE_REGION_NAME       = SCCACHE_REGION_NAME
    SCCACHE_S3_NO_CREDENTIALS = SCCACHE_S3_NO_CREDENTIALS
    VLLM_USE_PRECOMPILED      = VLLM_USE_PRECOMPILED
    VLLM_MERGE_BASE_COMMIT    = VLLM_MERGE_BASE_COMMIT
  }
}

target "test-ci" {
  inherits   = ["_common", "_ci", "_labels"]
  target     = "test"
  cache-from = get_cache_from()
  cache-to   = get_cache_to()
  tags = compact([
    IMAGE_TAG,
    IMAGE_TAG_LATEST,
  ])
  output = ["type=registry"]
}

target "cache-warm" {
  inherits = ["_common", "_ci", "_labels"]
  target   = "test"
  output   = ["type=cacheonly"]
  cache-from = compact([
    "type=registry,ref=${REGISTRY}/vllm-ci-postmerge-cache:latest,mode=max",
    BUILDKITE_COMMIT != "" ? "type=registry,ref=${REGISTRY}/vllm-ci-test-cache:${BUILDKITE_COMMIT},mode=max" : "",
    VLLM_MERGE_BASE_COMMIT != "" ? "type=registry,ref=${REGISTRY}/vllm-ci-test-cache:${VLLM_MERGE_BASE_COMMIT},mode=max" : "",
  ])
}
