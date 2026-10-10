#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Generate Buildkite annotation for a ROCm stack's wheel release.
#
# Usage: annotate-rocm-release.sh <base-dockerfile> [--default]
#   base-dockerfile  e.g. docker/Dockerfile.rocm_base or docker/Dockerfile.rocm_72_base
#   --default        the stack also owns the plain :latest / :v<ver> Docker Hub tags
set -ex

# shellcheck source=.buildkite/scripts/rocm/stack.sh
source "$(dirname "${BASH_SOURCE[0]}")/rocm/stack.sh" "${1:?Usage: $0 <base-dockerfile> [--default]}"
TAG_VARIANTS=("-${ROCM_STACK_VARIANT}")
if [[ "${2:-}" == "--default" ]]; then
  TAG_VARIANTS=("" "-${ROCM_STACK_VARIANT}")
fi
ROCM_VERSION="$ROCM_STACK_VERSION"
VARIANT="$ROCM_STACK_VARIANT"
PYTHON_VERSION=$(sed -nE 's/^ARG PYTHON_VERSION="?([^" ]+)"?.*/\1/p' "$ROCM_STACK_BASE_DOCKERFILE")
PYTORCH_ROCM_ARCH=$(sed -nE 's/^ARG PYTORCH_ROCM_ARCH="?([^"]+)"?.*/\1/p' "$ROCM_STACK_BASE_DOCKERFILE")

# Get release version, default to 1.0.0.dev for nightly/per-commit builds
RELEASE_VERSION=$(buildkite-agent meta-data get release-version 2>/dev/null || echo "")
if [ -z "${RELEASE_VERSION}" ]; then
  RELEASE_VERSION="1.0.0.dev"
fi

# S3 URLs
S3_BUCKET="${S3_BUCKET:-vllm-wheels}"
S3_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
S3_HOST="${S3_BUCKET}.s3-website-${S3_REGION}.amazonaws.com"
S3_URL="http://${S3_HOST}"
WHEEL_PATH="rocm/${BUILDKITE_COMMIT}/${VARIANT}-wheels/"

DOCKER_CMDS="docker pull ${ROCM_STACK_ECR_BASE}
docker pull ${ROCM_STACK_ECR_IMAGE}"
for suffix in "${TAG_VARIANTS[@]}"; do
  for tag in "latest${suffix}" "v${RELEASE_VERSION}${suffix}"; do
    DOCKER_CMDS+="
docker tag ${ROCM_STACK_ECR_BASE} vllm/vllm-openai-rocm:${tag}-base
docker tag ${ROCM_STACK_ECR_IMAGE} vllm/vllm-openai-rocm:${tag}
docker push vllm/vllm-openai-rocm:${tag}-base
docker push vllm/vllm-openai-rocm:${tag}"
  done
done

buildkite-agent annotate --style 'success' --context "rocm-release-workflow-${VARIANT}" << EOF
## ROCm ${ROCM_VERSION} Wheel and Docker Image Releases
### Build Configuration
| Setting | Value |
|---------|-------|
| **ROCm Version** | ${ROCM_VERSION} |
| **Index Variant** | ${VARIANT} |
| **Python Version** | ${PYTHON_VERSION} |
| **GPU Architectures** | ${PYTORCH_ROCM_ARCH} |
| **Branch** | \`${BUILDKITE_BRANCH}\` |
| **Commit** | \`${BUILDKITE_COMMIT}\` |

### :package: Installation

\`\`\`bash
# This build (by commit)
pip install vllm --extra-index-url ${S3_URL}/rocm/${BUILDKITE_COMMIT}/${VARIANT}/ --trusted-host ${S3_HOST}
# Nightly (if published)
pip install vllm --extra-index-url ${S3_URL}/rocm/nightly/${VARIANT}/ --trusted-host ${S3_HOST}
\`\`\`

### :floppy_disk: Download Wheels Directly

\`\`\`bash
aws s3 ls s3://${S3_BUCKET}/${WHEEL_PATH}
aws s3 cp --recursive --exclude "*" --include "*.whl" s3://${S3_BUCKET}/${WHEEL_PATH} .
\`\`\`

### :warning: Notes
- These wheels are built for **ROCm ${ROCM_VERSION}** and will NOT work with CUDA GPUs
- Supported GPU architectures: ${PYTORCH_ROCM_ARCH}
- Platform: Linux x86_64 only

### :package: Docker Image Release

\`\`\`
${DOCKER_CMDS}
\`\`\`

EOF
