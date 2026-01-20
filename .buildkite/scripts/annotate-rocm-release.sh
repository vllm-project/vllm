#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Generate Buildkite annotation for ROCm wheel release

set -ex

# Get build configuration from meta-data
# Extract ROCm version dynamically from Dockerfile.rocm_base
# BASE_IMAGE format: rocm/dev-ubuntu-22.04:7.1-complete -> extracts "7.1"
ROCM_VERSION=$(grep -E '^ARG BASE_IMAGE=' docker/Dockerfile.rocm_base | sed -E 's/.*:([0-9]+\.[0-9]+).*/\1/' || echo "unknown")
PYTHON_VERSION=$(buildkite-agent meta-data get rocm-python-version 2>/dev/null || echo "3.12")
PYTORCH_ROCM_ARCH=$(buildkite-agent meta-data get rocm-pytorch-rocm-arch 2>/dev/null || echo "gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151")

# S3 URLs
S3_BUCKET="${S3_BUCKET:-vllm-wheels}"
S3_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
S3_URL="https://${S3_BUCKET}.s3.${S3_REGION}.amazonaws.com"
ROCM_PATH="rocm/${BUILDKITE_COMMIT}"

buildkite-agent annotate --style 'success' --context 'rocm-release-workflow' << EOF
## :rocm: ROCm Wheel Release

### Build Configuration
| Setting | Value |
|---------|-------|
| **ROCm Version** | ${ROCM_VERSION} |
| **Python Version** | ${PYTHON_VERSION} |
| **GPU Architectures** | ${PYTORCH_ROCM_ARCH} |
| **Branch** | \`${BUILDKITE_BRANCH}\` |
| **Commit** | \`${BUILDKITE_COMMIT}\` |

### :package: Installation

**Install from this build (by commit):**
\`\`\`bash
uv pip install vllm --extra-index-url ${S3_URL}/${ROCM_PATH}/{rocm_variant}/

# Example:
uv pip install vllm --extra-index-url ${S3_URL}/${ROCM_PATH}/rocm700/
\`\`\`

**Install from nightly (if published):**
\`\`\`bash
uv pip install vllm --extra-index-url ${S3_URL}/rocm/nightly/
\`\`\`

### :floppy_disk: Download Wheels Directly

\`\`\`bash
# List all ROCm wheels
aws s3 ls s3://${S3_BUCKET}/${ROCM_PATH}/

# Download specific wheels
aws s3 cp s3://${S3_BUCKET}/${ROCM_PATH}/vllm-*.whl .
aws s3 cp s3://${S3_BUCKET}/${ROCM_PATH}/torch-*.whl .
aws s3 cp s3://${S3_BUCKET}/${ROCM_PATH}/triton_rocm-*.whl .
aws s3 cp s3://${S3_BUCKET}/${ROCM_PATH}/torchvision-*.whl .
aws s3 cp s3://${S3_BUCKET}/${ROCM_PATH}/amdsmi-*.whl .
\`\`\`

### :gear: Included Packages
- **vllm**: vLLM with ROCm support
- **torch**: PyTorch built for ROCm ${ROCM_VERSION}
- **triton_rocm**: Triton built for ROCm
- **torchvision**: TorchVision for ROCm PyTorch
- **amdsmi**: AMD SMI Python bindings

### :warning: Notes
- These wheels are built for **ROCm ${ROCM_VERSION}** and will NOT work with CUDA GPUs
- Supported GPU architectures: ${PYTORCH_ROCM_ARCH}
- Platform: Linux x86_64 only
EOF
