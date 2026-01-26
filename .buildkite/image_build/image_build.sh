#!/bin/bash
set -euo pipefail

#################################
#         Helper Functions      #
#################################

# Function to replace invalid characters in Docker image tags and truncate to 128 chars
clean_docker_tag() {
    local input="$1"
    echo "$input" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-128
}

print_usage_and_exit() {
    echo "Usage: $0 <registry> <repo> <commit> <branch> <vllm_use_precompiled> <vllm_merge_base_commit> <cache_from> <cache_to>"
    exit 1
}

print_instance_info() {
    echo ""
    echo "=== Debug: Instance Information ==="
    # Get IMDSv2 token
    if TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null); then
        AMI_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
            http://169.254.169.254/latest/meta-data/ami-id 2>/dev/null || echo "unknown")
        INSTANCE_TYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
            http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
        INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
            http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
        AZ=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
            http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null || echo "unknown")
        echo "AMI ID:        ${AMI_ID}"
        echo "Instance Type: ${INSTANCE_TYPE}"
        echo "Instance ID:   ${INSTANCE_ID}"
        echo "AZ:            ${AZ}"
    else
        echo "Not running on EC2 or IMDS not available"
    fi
    # Check for warm cache AMI (marker file baked into custom AMI)
    if [[ -f /etc/vllm-ami-info ]]; then
        echo "Cache:         warm (custom vLLM AMI)"
        cat /etc/vllm-ami-info
    else
        echo "Cache:         cold (standard AMI)"
    fi
    echo "==================================="
    echo ""
}

setup_buildx_builder() {
    echo "--- :buildkite: Setting up buildx builder"
    if [[ -S "${BUILDKIT_SOCKET}" ]]; then
        # Custom AMI with standalone buildkitd - use remote driver for warm cache
        echo "✅ Found local buildkitd socket at ${BUILDKIT_SOCKET}"
        echo "Using remote driver to connect to buildkitd (warm cache available)"
        if docker buildx inspect baked-vllm-builder >/dev/null 2>&1; then
            echo "Using existing baked-vllm-builder"
            docker buildx use baked-vllm-builder
        else
            echo "Creating baked-vllm-builder with remote driver"
            docker buildx create \
                --name baked-vllm-builder \
                --driver remote \
                --use \
                "unix://${BUILDKIT_SOCKET}"
        fi
        docker buildx inspect --bootstrap
    elif docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        # Existing builder available
        echo "Using existing builder: ${BUILDER_NAME}"
        docker buildx use "${BUILDER_NAME}"
        docker buildx inspect --bootstrap
    else
        # No local buildkitd, no existing builder - create new docker-container builder
        echo "No local buildkitd found, using docker-container driver"
        docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
        docker buildx inspect --bootstrap
    fi

    # Show builder info
    echo "Active builder:"
    docker buildx ls | grep -E '^\*|^NAME' || docker buildx ls
}

check_and_skip_if_image_exists() {
    if [[ -n "${IMAGE_TAG:-}" ]]; then
        echo "--- :mag: Checking if image exists"
        if docker manifest inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
            echo "Image already exists: ${IMAGE_TAG}"
            echo "Skipping build"
            exit 0
        fi
        echo "Image not found, proceeding with build"
    fi
}

# Helper to authenticate with AWS ECR
ecr_login() {
    aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$REGISTRY"
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 936637512419.dkr.ecr.us-east-1.amazonaws.com
}

prepare_cache_tags() {
    # Expects and sets: CACHE_TO, CACHE_FROM, CACHE_FROM_BASE_BRANCH, CACHE_FROM_MAIN
    TEST_CACHE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com/vllm-ci-test-cache"
    MAIN_CACHE_ECR="936637512419.dkr.ecr.us-east-1.amazonaws.com/vllm-ci-postmerge-cache"

    if [[ "$BUILDKITE_PULL_REQUEST" == "false" ]]; then
        if [[ "$BUILDKITE_BRANCH" == "main" ]]; then
            cache="${MAIN_CACHE_ECR}:latest"
        else
            clean_branch=$(clean_docker_tag "$BUILDKITE_BRANCH")
            cache="${TEST_CACHE_ECR}:${clean_branch}"
        fi
        CACHE_TO="$cache"
        CACHE_FROM="$cache"
        CACHE_FROM_BASE_BRANCH="$cache"
    else
        CACHE_TO="${TEST_CACHE_ECR}:pr-${BUILDKITE_PULL_REQUEST}"
        CACHE_FROM="${TEST_CACHE_ECR}:pr-${BUILDKITE_PULL_REQUEST}"
        if [[ "$BUILDKITE_PULL_REQUEST_BASE_BRANCH" == "main" ]]; then
            CACHE_FROM_BASE_BRANCH="${MAIN_CACHE_ECR}:latest"
        else
            clean_base=$(clean_docker_tag "$BUILDKITE_PULL_REQUEST_BASE_BRANCH")
            CACHE_FROM_BASE_BRANCH="${TEST_CACHE_ECR}:${clean_base}"
        fi
    fi

    CACHE_FROM_MAIN="${MAIN_CACHE_ECR}:latest"
    export CACHE_TO CACHE_FROM CACHE_FROM_BASE_BRANCH CACHE_FROM_MAIN
}

resolve_parent_commit() {
    if [[ -z "${PARENT_COMMIT:-}" ]]; then
        PARENT_COMMIT=$(git rev-parse HEAD~1 2>/dev/null || echo "")
        if [[ -n "${PARENT_COMMIT}" ]]; then
            echo "Computed parent commit for cache fallback: ${PARENT_COMMIT}"
            export PARENT_COMMIT
        else
            echo "Could not determine parent commit (may be first commit in repo)"
        fi
    else
        echo "Using provided PARENT_COMMIT: ${PARENT_COMMIT}"
    fi
}

print_bake_config() {
    echo "--- :page_facing_up: Resolved bake configuration"
    BAKE_CONFIG_FILE="bake-config-build-${BUILDKITE_BUILD_NUMBER:-local}.json"
    docker buildx bake -f "${VLLM_BAKE_FILE}" -f "${CI_HCL_PATH}" --print "${TARGET}" | tee "${BAKE_CONFIG_FILE}" || true
    echo "Saved bake config to ${BAKE_CONFIG_FILE}"
    buildkite-agent artifact upload "${BAKE_CONFIG_FILE}"
}

#################################
#         Main Script           #
#################################
print_instance_info

# Argument check
if [[ $# -lt 7 ]]; then
    print_usage_and_exit
fi

# Input arguments
REGISTRY=$1
REPO=$2
BUILDKITE_COMMIT=$3
BRANCH=$4
VLLM_USE_PRECOMPILED=$5
VLLM_MERGE_BASE_COMMIT=$6
IMAGE_TAG=$7
IMAGE_TAG_LATEST=${8:-} # only used for main branch, optional

# Configuration with sensible defaults
TARGET="test-ci"
CI_HCL_URL="${CI_HCL_URL:-https://raw.githubusercontent.com/vllm-project/ci-infra/main/docker/ci.hcl}"
VLLM_BAKE_FILE="${VLLM_BAKE_FILE:-docker/docker-bake.hcl}"
BUILDER_NAME="${BUILDER_NAME:-vllm-builder}"
CI_HCL_PATH="/tmp/ci.hcl"
BUILDKIT_SOCKET="/run/buildkit/buildkitd.sock"

# Prepare cache tags based on PR/branch context
prepare_cache_tags

# Authenticate with AWS ECR
ecr_login

# Environment info (for docs and human readers)
#   CI_HCL_URL          - URL to ci.hcl (default: from ci-infra main branch)
#   VLLM_CI_BRANCH      - ci-infra branch to use (default: main)
#   VLLM_BAKE_FILE      - Path to vLLM's bake file (default: docker/docker-bake.hcl)
#   BUILDER_NAME        - Name for buildx builder (default: vllm-builder)
#
# Build configuration (exported as environment variables for bake):
export BUILDKITE_COMMIT
export PARENT_COMMIT
export IMAGE_TAG
export IMAGE_TAG_LATEST
export CACHE_FROM
export CACHE_FROM_BASE_BRANCH
export CACHE_FROM_MAIN
export CACHE_TO
export VLLM_USE_PRECOMPILED
export VLLM_MERGE_BASE_COMMIT

# print out all args
echo "--- :mag: Arguments"
echo "REGISTRY: ${REGISTRY}"
echo "REPO: ${REPO}"
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"
echo "BRANCH: ${BRANCH}"
echo "VLLM_USE_PRECOMPILED: ${VLLM_USE_PRECOMPILED}"
echo "VLLM_MERGE_BASE_COMMIT: ${VLLM_MERGE_BASE_COMMIT}"
echo "IMAGE_TAG: ${IMAGE_TAG}"
echo "IMAGE_TAG_LATEST: ${IMAGE_TAG_LATEST}"

# print out all build configuration
echo "--- :mag: Build configuration"
echo "TARGET: ${TARGET}"
echo "CI HCL URL: ${CI_HCL_URL}"
echo "vLLM bake file: ${VLLM_BAKE_FILE}"
echo "BUILDER_NAME: ${BUILDER_NAME}"
echo "CI_HCL_PATH: ${CI_HCL_PATH}"
echo "BUILDKIT_SOCKET: ${BUILDKIT_SOCKET}"

echo "--- :mag: Cache tags"
echo "CACHE_TO: ${CACHE_TO}"
echo "CACHE_FROM: ${CACHE_FROM}"
echo "CACHE_FROM_BASE_BRANCH: ${CACHE_FROM_BASE_BRANCH}"
echo "CACHE_FROM_MAIN: ${CACHE_FROM_MAIN}"

echo "--- :mag: Resolved config"
echo "PARENT_COMMIT: ${PARENT_COMMIT}"

# Short-circuit for existing image
check_and_skip_if_image_exists

echo "--- :docker: Setting up Docker buildx bake"
echo "Target: ${TARGET}"
echo "CI HCL URL: ${CI_HCL_URL}"
echo "vLLM bake file: ${VLLM_BAKE_FILE}"

# Check if vLLM bake file exists
if [[ ! -f "${VLLM_BAKE_FILE}" ]]; then
    echo "Error: vLLM bake file not found at ${VLLM_BAKE_FILE}"
    echo "Make sure you're running from the vLLM repository root"
    exit 1
fi

# Download ci.hcl file
echo "--- :arrow_down: Downloading ci.hcl"
curl -sSfL -o "${CI_HCL_PATH}" "${CI_HCL_URL}"
echo "Downloaded to ${CI_HCL_PATH}"

# Setup docker buildx builder
setup_buildx_builder

# Compute parent commit for cache fallback (if not already set)
resolve_parent_commit

# Print resolved config for diagnostic artifact
print_bake_config

# Building
echo "--- :docker: Building ${TARGET}"
docker --debug buildx bake -f "${VLLM_BAKE_FILE}" -f "${CI_HCL_PATH}" --progress plain "${TARGET}"

echo "--- :white_check_mark: Build complete"

buildkite-agent artifact upload "bake-config-build-${BUILDKITE_BUILD_NUMBER:-local}.json"
