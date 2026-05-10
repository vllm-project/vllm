#!/bin/bash

set -ex

# Get release version, default to 1.0.0.dev for nightly/per-commit builds
RELEASE_VERSION=$(buildkite-agent meta-data get release-version 2>/dev/null | sed 's/^v//')
if [ -z "${RELEASE_VERSION}" ]; then
  RELEASE_VERSION="1.0.0.dev"
fi

buildkite-agent annotate --style 'info' --context 'release-workflow' << EOF
To download the wheel (by commit):
\`\`\`
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}-cp38-abi3-manylinux_2_35_x86_64.whl .
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}-cp38-abi3-manylinux_2_35_aarch64.whl .

(Optional) For CUDA 12.9:
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}+cu129-cp38-abi3-manylinux_2_31_x86_64.whl .
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}+cu129-cp38-abi3-manylinux_2_31_aarch64.whl .

(Optional) For CPU:
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl .
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}+cpu-cp38-abi3-manylinux_2_35_aarch64.whl .
\`\`\`

Docker images are published automatically by the "Publish release images to DockerHub" pipeline step.
EOF
