#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Promote ROCm release artifacts without rebuilding anything.
# The release pipeline's build step (build.sh release) pushed the source images to
# ECR; this script only copies digests and uploads wheels.
#
# Usage: publish.sh <stack> <mode>
#
# Modes:
#   wheels   Extract the vLLM wheel from the -wheels image and upload to S3.
#   images   Tag the release image and its runtime base for their variant in ECR.
#            With NIGHTLY=1, also push the nightly tags to Docker Hub.
#
# Required environment variables:
#   BUILDKITE_COMMIT   — the commit being released
#   ECR            — ECR repository prefix (release-pipeline.yaml's ROCM_CI_REPO)
#
# Optional:
#   HUB            — Docker Hub repo for nightly images (default: vllm/vllm-openai-rocm)
#   NIGHTLY        — set to 1 to push nightly Hub tags (default: 0)
#   DOCKERHUB_USERNAME — Hub username for cleanup script (default: vllmbot)
set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
STACK=$1
MODE=$2
COMMIT=${BUILDKITE_COMMIT:?BUILDKITE_COMMIT must be set}

ECR=${ECR:?ECR must be set (by release-pipeline.yaml)}
SRC=$ECR:${STACK}-${COMMIT}   # build.sh release, in the same build
HUB=${HUB:-vllm/vllm-openai-rocm}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# retry <cmd...>  — registry lookups: these steps have no retry rules
retry() { "$@" || { sleep 5; "$@"; } || { sleep 5; "$@"; }; }

# pin <ref>  — resolve <ref> to a pinned ref@digest (resolves once, stable for the rest of the script).
#   Parses the validated "Digest:" line, as build.sh does.
pin() {
  local d
  d=$(retry docker buildx imagetools inspect "$1" | awk '$1 == "Digest:" { print $2; exit }')
  [[ $d =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "cannot resolve the digest of $1" >&2; return 1; }
  echo "$1@$d"
}

# variant <ref>  — read the ai.vllm.rocm.variant image label
variant() {
  retry docker buildx imagetools inspect "$1" \
    --format '{{index .Image.Config.Labels "ai.vllm.rocm.variant"}}'
}

# ---------------------------------------------------------------------------
# Resolve source image refs (done once; images step uses them too)
# ---------------------------------------------------------------------------
# Logged in, the ECR lookups below are not subject to the anonymous rate limits.
aws ecr-public get-login-password --region us-east-1 \
  | docker login -u AWS --password-stdin "${ECR%%/*}"
IMG=$(pin "$SRC")
V=$(variant "$IMG")      # e.g. rocm100 — used in S3 paths, ECR tags and Hub tags
[[ $V =~ ^rocm[0-9]+$ ]] || { echo "$IMG has no valid ai.vllm.rocm.variant label: '$V'" >&2; exit 1; }
# For the steps after either mode: root-index generation and publish-release-images.sh.
buildkite-agent meta-data set "rocm-$STACK-variant" "$V"

# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
case $MODE in

  # -------------------------------------------------------------------------
  wheels)
  # -------------------------------------------------------------------------
  # Extract the vLLM wheel from this commit's -wheels image and upload it
  # to S3 so users can pip-install it directly.
  # -------------------------------------------------------------------------

    WHEELS_IMG=$(pin "$SRC-wheels")

    # Export wheel files out of the scratch image into ./dist/
    rm -rf dist
    echo "FROM $WHEELS_IMG" | docker buildx build -q -o dist -

    # Sanity: the wheel's variant suffix must match the stack label.
    # (setup.py's rocm_version_to_variant produces e.g. rocm100 for 10.0.x)
    actual_variant=$(basename dist/vllm-*.whl | grep -oE 'rocm[0-9]+' | tail -1)
    [[ $actual_variant == "$V" ]] \
      || { echo "Wheel variant $actual_variant != image variant $V" >&2; exit 1; }

    bash .buildkite/scripts/upload-rocm-wheels.sh "$STACK" dist

    # Buildkite annotation: pip install + direct S3 download commands.
    WHL_URL="https://wheels.vllm.ai/rocm/$COMMIT/$V"
    buildkite-agent annotate --style success --context "rocm-$STACK-wheels" << EOF
## :package: ROCm \`$V\` wheels published

| | |
|---|---|
| **Variant** | \`$V\` |
| **Commit**  | \`$COMMIT\` |
| **Branch**  | \`${BUILDKITE_BRANCH:-?}\` |

### Install (this commit)
\`\`\`bash
pip install vllm --extra-index-url $WHL_URL
\`\`\`

### Install (latest nightly)
\`\`\`bash
pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/nightly/$V
\`\`\`

### Download wheels directly
\`\`\`bash
aws s3 ls  s3://${S3_BUCKET:-vllm-wheels}/rocm/$COMMIT/$V-wheels/
aws s3 cp  --recursive --exclude "*" --include "*.whl" s3://${S3_BUCKET:-vllm-wheels}/rocm/$COMMIT/$V-wheels/ .
\`\`\`
EOF
    ;;

  # -------------------------------------------------------------------------
  images)
  # -------------------------------------------------------------------------
  # Tag the release image and its runtime base for their variant in ECR.
  # With NIGHTLY=1, also tag them on Docker Hub and prune old nightly tags.
  # -------------------------------------------------------------------------

    RT=$(pin "$SRC-base")    # runtime base alias (published as base-nightly*)

    # --- ECR -----------------------------------------------------------------
    docker buildx imagetools create -t "$ECR:$COMMIT-$V"       "$IMG"
    docker buildx imagetools create -t "$ECR:$COMMIT-$V-base"  "$RT"

    ECR_IMG="$ECR:$COMMIT-$V"
    ECR_BASE="$ECR:$COMMIT-$V-base"

    # --- Non-nightly annotation ----------------------------------------------
    if [[ ${NIGHTLY:-0} != 1 ]]; then
      buildkite-agent annotate --style success --context "rocm-$STACK-images" << EOF
## :docker: ROCm \`$V\` images published to ECR

| | |
|---|---|
| **Variant** | \`$V\` |
| **Commit**  | \`$COMMIT\` |
| **ECR image** | \`$ECR_IMG\` |
| **ECR base**  | \`$ECR_BASE\` |

\`\`\`bash
docker pull $ECR_IMG
\`\`\`
EOF
      exit 0
    fi

    # --- Nightly: Docker Hub -------------------------------------------------
    export DOCKERHUB_USERNAME=${DOCKERHUB_USERNAME:-vllmbot}

    # The default stack (rocm) owns the unqualified :nightly / :base-nightly tags.
    # All stacks own the variant-qualified :nightly-<variant> tags.
    suffixes=("-$V")
    [[ $STACK == rocm ]] && suffixes+=("")

    hub_tags=""
    for sfx in "${suffixes[@]}"; do
      docker buildx imagetools create \
        -t "$HUB:nightly$sfx" -t "$HUB:nightly$sfx-$COMMIT" "$IMG"
      docker buildx imagetools create \
        -t "$HUB:base-nightly$sfx" -t "$HUB:base-nightly$sfx-$COMMIT" "$RT"
      hub_tags="$hub_tags \`$HUB:nightly$sfx\` \`$HUB:nightly$sfx-$COMMIT\`"
    done

    # Prune old nightly tags (keep last 14 builds per tag family).
    clean=.buildkite/scripts/cleanup-nightly-builds.sh
    bash "$clean" "nightly-$V-"      "$HUB"
    bash "$clean" "base-nightly-$V-" "$HUB"
    if [[ $STACK == rocm ]]; then
      bash "$clean" "nightly-"      "$HUB" "nightly-rocm"
      bash "$clean" "base-nightly-" "$HUB" "base-nightly-rocm"
    fi

    buildkite-agent annotate --style success --context "rocm-$STACK-images" << EOF
## :docker: ROCm \`$V\` nightly images published

| | |
|---|---|
| **Variant**   | \`$V\` |
| **Commit**    | \`$COMMIT\` |
| **ECR image** | \`$ECR_IMG\` |
| **ECR base**  | \`$ECR_BASE\` |
| **Docker Hub** |$hub_tags |

\`\`\`bash
docker pull $HUB:nightly-$V
\`\`\`
EOF
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
