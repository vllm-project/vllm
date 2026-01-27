#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Generate Buildkite annotation for ROCm wheel release
set -ex

# Get build configuration from meta-data
# Extract ROCm version dynamically from Dockerfile.rocm_base
# BASE_IMAGE format: rocm/dev-ubuntu-22.04:7.0-complete -> extracts "7.0"
ROCM_VERSION=$(grep -E '^ARG BASE_IMAGE=' docker/Dockerfile.rocm_base | sed -E 's/.*:([0-9]+\.[0-9]+).*/\1/' || echo "unknown")
PYTHON_VERSION=$(buildkite-agent meta-data get rocm-python-version 2>/dev/null || echo "3.12")
PYTORCH_ROCM_ARCH=$(buildkite-agent meta-data get rocm-pytorch-rocm-arch 2>/dev/null || echo "gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151")

# S3 URLs
S3_BUCKET="${S3_BUCKET:-vllm-wheels-dev}"
S3_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
S3_URL="http://${S3_BUCKET}.s3-website-${S3_REGION}.amazonaws.com"

# Format ROCm version for path (e.g., "7.1" -> "rocm710")
ROCM_VERSION_PATH="rocm$(echo ${ROCM_VERSION} | tr -d '.')"
ROCM_PATH="rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}"
buildkite-agent annotate --style 'success' --context 'rocm-release-workflow' << EOF
## ROCm Wheel and Docker Image Releases
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
pip install vllm --extra-index-url ${S3_URL}/${ROCM_PATH}/ --trusted-host ${S3_BUCKET}.s3-website-${S3_REGION}.amazonaws.com

# Example for ROCm ${ROCM_VERSION}:
pip install vllm --extra-index-url ${S3_URL}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/ --trusted-host ${S3_BUCKET}.s3-website-${S3_REGION}.amazonaws.com
\`\`\`

**Install from nightly (if published):**

\`\`\`bash
pip install vllm --extra-index-url ${S3_URL}/rocm/nightly/ --trusted-host ${S3_BUCKET}.s3-website-${S3_REGION}.amazonaws.com
\`\`\`

### :floppy_disk: Download Wheels Directly

\`\`\`bash
# List all ROCm wheels
aws s3 ls s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/
# Download specific wheels
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/vllm-*.whl .
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/torch-*.whl .
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/triton-*.whl .
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/triton-kernels-*.whl .
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/torchvision-*.whl .
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/torchaudio-*.whl .
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/amdsmi-*.whl .
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/aiter-*.whl .
aws s3 cp s3://${S3_BUCKET}/rocm/${BUILDKITE_COMMIT}/${ROCM_VERSION_PATH}/flash-attn-*.whl .
\`\`\`

### :gear: Included Packages
- **vllm**: vLLM with ROCm support
- **torch**: PyTorch built for ROCm ${ROCM_VERSION}
- **triton**: Triton
- **triton-kernels**: Triton kernels
- **torchvision**: TorchVision for ROCm PyTorch
- **torchaudio**: Torchaudio for ROCm PyTorch
- **amdsmi**: AMD SMI Python bindings
- **aiter**: Aiter for ROCm
- **flash-attn**: Flash Attention for ROCm

### :warning: Notes
- These wheels are built for **ROCm ${ROCM_VERSION}** and will NOT work with CUDA GPUs
- Supported GPU architectures: ${PYTORCH_ROCM_ARCH}
- Platform: Linux x86_64 only

### :package: Docker Image Release

To download and upload the image:

\`\`\`
docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-rocm vllm/vllm-openai-rocm:${BUILDKITE_COMMIT}
docker tag vllm/vllm-openai-rocm:${BUILDKITE_COMMIT} vllm/vllm-openai-rocm:latest
docker tag vllm/vllm-openai-rocm:${BUILDKITE_COMMIT} vllm/vllm-openai-rocm:v${RELEASE_VERSION}
docker push vllm/vllm-openai-rocm:latest
docker push vllm/vllm-openai-rocm:v${RELEASE_VERSION}
\`\`\`

EOF