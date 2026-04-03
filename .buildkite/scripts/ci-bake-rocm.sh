#!/bin/bash
# ci-bake-rocm.sh - Wrapper script for Docker buildx bake CI builds
#
# Canonical location: vllm repo at .buildkite/scripts/ci-bake-rocm.sh
# Kept in sync with ci-infra repo at buildkite/scripts/ci-bake-rocm.sh.
# Update both when making changes; the vllm copy is what actually runs in CI
# (pinned to the vllm commit under test).
#
# This script handles the common setup for running docker buildx bake:
# - Downloads ci.hcl from ci-infra
# - Detects and uses local buildkitd if available (custom AMI with warm cache)
# - Falls back to docker-container driver on regular instances
# - Runs bake with --print for debugging
# - Runs the actual build
#
# Usage:
#   ci-bake-rocm.sh [TARGET]
#
# Environment variables (all optional, with sensible defaults):
#   CI_HCL_URL          - URL to ci.hcl (default: from ci-infra main branch)
#   VLLM_CI_BRANCH      - ci-infra branch to use (default: main)
#   VLLM_BAKE_FILE      - Path to vLLM's bake file (default: docker/docker-bake.hcl)
#   BUILDER_NAME        - Name for buildx builder (default: vllm-builder)
#
# Build configuration (passed through to bake via environment):
#   BUILDKITE_COMMIT    - Git commit (auto-detected from Buildkite)
#   PARENT_COMMIT       - Parent commit (HEAD~1) for cache fallback (auto-computed)
#   IMAGE_TAG           - Primary image tag
#   IMAGE_TAG_LATEST    - Latest tag (optional)
#   CACHE_FROM          - Cache source
#   CACHE_FROM_BASE     - Base branch cache source
#   CACHE_FROM_MAIN     - Main branch cache source
#   CACHE_TO            - Cache destination
#   VLLM_USE_PRECOMPILED    - Use precompiled wheels
#   VLLM_MERGE_BASE_COMMIT  - Merge base commit for precompiled

set -euo pipefail

get_remote_image_label() {
    local image_ref="$1"
    local label_key="$2"

    docker buildx imagetools inspect "${image_ref}" --raw 2>/dev/null \
        | python3 -c '
import json
import subprocess
import sys

image_ref = sys.argv[1]
label_key = sys.argv[2]

try:
    data = json.load(sys.stdin)
    if data.get("manifests"):
        digest = data["manifests"][0]["digest"]
        result = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", image_ref + "@" + digest, "--raw"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError("digest inspect failed")
        data = json.loads(result.stdout)
    labels = data.get("config", {}).get("Labels", {})
    print(labels.get(label_key, ""))
except Exception:
    print("")
' "${image_ref}" "${label_key}" 2>/dev/null || echo ""
}

# Check if image already exists (skip build if it does)
#
# For commit-tagged images (rocm/vllm-ci:$COMMIT), the tag is unique per
# commit, so "exists = already built" is correct.
#
# For stable-tagged images (rocm/vllm-dev:ci_base), the tag always exists
# after the first weekly build. To detect staleness, we compare a hash of
# the ci_base-affecting source files against a label on the remote image.
# If the hashes match, the image is current and we skip. If they differ
# (or the label is missing), we rebuild.
if [[ -n "${IMAGE_TAG:-}" && "${FORCE_BUILD:-0}" != "1" ]]; then
    echo "--- :mag: Checking if image exists"
    if docker manifest inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
        # Image exists. Check content hash for stable tags.
        if [[ -n "${CI_BASE_CONTENT_FILES:-}" ]]; then
            LOCAL_HASH=$(cat ${CI_BASE_CONTENT_FILES} 2>/dev/null | sha256sum | cut -d' ' -f1)
            echo "Local ci_base content hash: ${LOCAL_HASH:0:16}..."

            REMOTE_HASH=$(get_remote_image_label "${IMAGE_TAG}" "vllm.ci_base.content_hash")

            if [[ -n "${REMOTE_HASH}" ]]; then
                echo "Remote ci_base content hash: ${REMOTE_HASH:0:16}..."
                if [[ "${LOCAL_HASH}" == "${REMOTE_HASH}" ]]; then
                    echo "Content hashes match -- ci_base is current"
                    echo "Skipping build"
                    exit 0
                else
                    echo "Content hashes DIFFER -- ci_base is stale, rebuilding"
                fi
            else
                echo "Remote image has no content hash label -- rebuilding to add it"
            fi
        else
            echo "Image already exists: ${IMAGE_TAG}"
            echo "Skipping build"
            exit 0
        fi
    else
        echo "Image not found, proceeding with build"
    fi
fi

# Configuration with defaults
TARGET="${1:-test-ci}"
CI_HCL_URL="${CI_HCL_URL:-https://raw.githubusercontent.com/vllm-project/ci-infra/main/docker/ci.hcl}"
VLLM_BAKE_FILE="${VLLM_BAKE_FILE:-docker/docker-bake.hcl}"
BUILDER_NAME="${BUILDER_NAME:-vllm-builder}"
CI_HCL_PATH="/tmp/ci.hcl"
BUILDKIT_SOCKET="/run/buildkit/buildkitd.sock"

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

# Download ci.hcl
echo "--- :arrow_down: Downloading ci.hcl"
curl -sSfL -o "${CI_HCL_PATH}" "${CI_HCL_URL}"
echo "Downloaded to ${CI_HCL_PATH}"

# Set up buildx builder
# Priority: 1) local buildkitd socket (custom AMI) 2) existing builder 3) new docker-container builder
echo "--- :buildkite: Setting up buildx builder"

if [[ -S "${BUILDKIT_SOCKET}" ]]; then
    # Custom AMI with standalone buildkitd - use remote driver for warm cache
    echo "Found local buildkitd socket at ${BUILDKIT_SOCKET}"
    echo "Using remote driver to connect to buildkitd (warm cache available)"

    # Check if ${BUILDER_NAME} already exists and is using the socket
    if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        echo "Using existing builder: ${BUILDER_NAME}"
        docker buildx use "${BUILDER_NAME}"
    else
        echo "Creating builder '${BUILDER_NAME}' with remote driver"
        docker buildx create \
            --name "${BUILDER_NAME}" \
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

# Deepen shallow clones so HEAD~1 and merge-base are available.
# Buildkite agents often clone with --depth=1; without deepening, git rev-parse
# HEAD~1 and git merge-base both silently fail, disabling the per-commit cache layers.
if git rev-parse --is-shallow-repository 2>/dev/null | grep -q "true"; then
    echo "Shallow clone detected - deepening for cache key computation"
    # --deepen=1 extends the current shallow clone by 1 commit along the
    # already-fetched branch, making HEAD~1 available.  Unlike --depth=2
    # with a refspec, it operates on the currently checked-out branch and
    # is safe in detached-HEAD (Buildkite) checkout state.
    git fetch --deepen=1 origin 2>/dev/null || true
fi

# Compute parent commit for cache fallback (if not already set)
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

# Compute merge-base with main for an additional cache fallback layer.
# Mirrors the VLLM_MERGE_BASE_COMMIT pattern used in the shared ci.hcl file.
# Useful for long-lived PRs where parent-commit cache may be missing but the
# merge-base (a real main commit) maps to a warm :rocm-latest snapshot.
if [[ -z "${VLLM_MERGE_BASE_COMMIT:-}" ]]; then
    # Fetch just the tip of main so merge-base can be resolved on shallow clones.
    git fetch --depth=1 origin main 2>/dev/null || true
    VLLM_MERGE_BASE_COMMIT=$(git merge-base HEAD origin/main 2>/dev/null || echo "")
    if [[ -n "${VLLM_MERGE_BASE_COMMIT}" ]]; then
        echo "Computed merge base commit for cache fallback: ${VLLM_MERGE_BASE_COMMIT}"
        export VLLM_MERGE_BASE_COMMIT
    else
        echo "Could not determine merge base (will skip that cache layer)"
    fi
else
    echo "Using provided VLLM_MERGE_BASE_COMMIT: ${VLLM_MERGE_BASE_COMMIT}"
fi

# Compute and export ci_base content hash (if content files are specified).
# This hash gets embedded as a label in the ci_base image via the bake file's
# CI_BASE_CONTENT_HASH variable, so future builds can compare without rebuilding.
if [[ -n "${CI_BASE_CONTENT_FILES:-}" ]]; then
    CI_BASE_CONTENT_HASH=$(cat ${CI_BASE_CONTENT_FILES} 2>/dev/null | sha256sum | cut -d' ' -f1)
    export CI_BASE_CONTENT_HASH
    echo "ci_base content hash: ${CI_BASE_CONTENT_HASH:0:16}... (will be embedded as image label)"
fi

# Print resolved configuration for debugging and upload as a Buildkite artifact
echo "--- :page_facing_up: Resolved bake configuration"
BAKE_CONFIG_FILE="bake-config-build-${BUILDKITE_BUILD_NUMBER:-local}.json"
docker buildx bake -f "${VLLM_BAKE_FILE}" -f "${CI_HCL_PATH}" --print "${TARGET}" | tee "${BAKE_CONFIG_FILE}" || true
if command -v buildkite-agent >/dev/null 2>&1 && [[ -n "${BUILDKITE_BUILD_NUMBER:-}" ]]; then
    buildkite-agent artifact upload "${BAKE_CONFIG_FILE}" || true
    echo "Uploaded ${BAKE_CONFIG_FILE} as Buildkite artifact"
else
    echo "Saved bake config to ${BAKE_CONFIG_FILE} (not in Buildkite, skipping upload)"
fi

# Run the actual build.
#
# BuildKit combines image push and cache export into one command. If the
# cache export fails (e.g., Docker Hub rejects a large layer blob with
# 400 Bad Request due to upload session timeout), the entire bake command
# returns non-zero even though the image was pushed successfully.
#
# To prevent cache export failures from blocking the pipeline, we:
# 1. Run bake normally (image push + cache export together)
# 2. If it fails, check whether the image was actually pushed
# 3. If the image exists on the registry, treat the cache failure as
#    non-fatal (warn, don't fail)
# 4. If the image does NOT exist, the failure is real (fail hard)
echo "--- :docker: Building ${TARGET}"
BUILD_RC=0
docker buildx bake -f "${VLLM_BAKE_FILE}" -f "${CI_HCL_PATH}" --progress plain "${TARGET}" || BUILD_RC=$?

if [[ ${BUILD_RC} -ne 0 ]]; then
    echo ""
    echo "WARNING: docker buildx bake exited with code ${BUILD_RC}"

    # Check if the build still produced the exact image we need.
    if [[ -n "${IMAGE_TAG:-}" ]]; then
        echo "Checking if image was pushed successfully..."
        IMAGE_CONFIRMED=0
        if docker manifest inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
            if [[ -n "${CI_BASE_CONTENT_HASH:-}" ]]; then
                REMOTE_HASH=$(get_remote_image_label "${IMAGE_TAG}" "vllm.ci_base.content_hash")
                if [[ -n "${REMOTE_HASH}" && "${REMOTE_HASH}" == "${CI_BASE_CONTENT_HASH}" ]]; then
                    IMAGE_CONFIRMED=1
                else
                    echo "Remote image exists but does not have the expected ci_base content hash."
                    echo "  expected: ${CI_BASE_CONTENT_HASH:0:16}..."
                    echo "  found:    ${REMOTE_HASH:0:16}..."
                fi
            elif [[ -n "${BUILDKITE_COMMIT:-}" ]]; then
                REMOTE_REVISION=$(get_remote_image_label "${IMAGE_TAG}" "org.opencontainers.image.revision")
                if [[ -n "${REMOTE_REVISION}" && "${REMOTE_REVISION}" == "${BUILDKITE_COMMIT}" ]]; then
                    IMAGE_CONFIRMED=1
                else
                    echo "Remote image exists but revision label does not match ${BUILDKITE_COMMIT}."
                    echo "  found revision: ${REMOTE_REVISION:-<missing>}"
                fi
            else
                IMAGE_CONFIRMED=1
            fi
        fi
        if [[ ${IMAGE_CONFIRMED} -eq 1 ]]; then
            echo ""
            echo "WARNING: Build reported failure (rc=${BUILD_RC}) but the"
            echo "         image was pushed successfully: ${IMAGE_TAG}"
            echo ""
            echo "         This is typically caused by a cache export failure"
            echo "         (Docker Hub rejecting a large layer blob upload)."
            echo "         The image is usable. Cache will be cold on next"
            echo "         build but the pipeline can proceed."
            echo ""
            echo "         Problematic layer (from build log):"
            echo "           sha256:6f62317ed92e... (~7.25GB ROCm runtime)"
            echo "         Root cause: Docker Hub upload session timeout on"
            echo "           PUT to /v2/rocm/vllm-ci-cache/blobs/uploads/"
            echo ""

            # Post a Buildkite annotation so the warning is visible.
            if command -v buildkite-agent >/dev/null 2>&1; then
                buildkite-agent annotate \
                    --style warning \
                    --context "cache-export-warning" \
                    "### :warning: Docker cache export failed (non-fatal)

Image was pushed successfully: \`${IMAGE_TAG}\`

The BuildKit registry cache export to \`rocm/vllm-ci-cache\` failed
(likely Docker Hub upload session timeout on a large layer). Next build
will not benefit from registry cache but will still work.

Build exit code: ${BUILD_RC}" 2>/dev/null || true
            fi

            # Override exit code: image exists, cache failure is non-fatal.
            BUILD_RC=0
        else
            echo ""
            echo "ERROR: Build failed and image was NOT pushed: ${IMAGE_TAG}"
            echo "       This is a real build failure, not a cache issue."
            echo ""
        fi
    fi
fi

if [[ ${BUILD_RC} -ne 0 ]]; then
    exit ${BUILD_RC}
fi

echo "--- :white_check_mark: Build complete"

# ---------------------------------------------------------------------------
# Wheel artifact upload.
#
# If the bake target included export-wheel-rocm (via a *-ci-with-wheel group),
# the wheel is already extracted to ./wheel-export/. Upload it
# as a Buildkite artifact so test jobs can assemble images locally from
# ci_base + wheel instead of pulling a large from Docker Hub.
#
# If ./wheel-export/ doesn't exist this section is a no-op.
#
# Artifact path:
#   artifacts/vllm-wheel-rocm/*.whl
# ---------------------------------------------------------------------------
WHEEL_DIR="./wheel-export"
if [[ -d "${WHEEL_DIR}" ]] && ls "${WHEEL_DIR}"/*.whl >/dev/null 2>&1; then
    echo "--- :package: Uploading vLLM wheel"

    ARTIFACT_DIR="artifacts/vllm-wheel-rocm"
    mkdir -p "${ARTIFACT_DIR}"

    for whl in "${WHEEL_DIR}"/*.whl; do
        [ -f "${whl}" ] || continue
        WHL_NAME=$(basename "${whl}")
        cp "${whl}" "${ARTIFACT_DIR}/${WHL_NAME}"
        echo "Copied ${WHL_NAME}: $(du -sh "${ARTIFACT_DIR}/${WHL_NAME}" | cut -f1)"
    done

    # Upload as Buildkite artifacts
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "${ARTIFACT_DIR}/*"
        echo "Wheel artifacts uploaded to ${ARTIFACT_DIR}/"
    else
        echo "Not in Buildkite, skipping artifact upload"
    fi

    rm -rf "${WHEEL_DIR}"
fi
