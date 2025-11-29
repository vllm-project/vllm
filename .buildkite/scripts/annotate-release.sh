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
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}-cp38-abi3-manylinux1_x86_64.whl .
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}-cp38-abi3-manylinux2014_aarch64.whl .

aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}+cu129-cp38-abi3-manylinux1_x86_64.whl .
aws s3 cp s3://vllm-wheels/${BUILDKITE_COMMIT}/vllm-${RELEASE_VERSION}+cu129-cp38-abi3-manylinux1_x86_64.whl .
\`\`\`

To download the wheel (by version):
\`\`\`
aws s3 cp s3://vllm-wheels/${RELEASE_VERSION}/vllm-${RELEASE_VERSION}-cp38-abi3-manylinux1_x86_64.whl .
aws s3 cp s3://vllm-wheels/${RELEASE_VERSION}/vllm-${RELEASE_VERSION}-cp38-abi3-manylinux2014_aarch64.whl .

aws s3 cp s3://vllm-wheels/${RELEASE_VERSION}+cu129/vllm-${RELEASE_VERSION}+cu129-cp38-abi3-manylinux1_x86_64.whl .
aws s3 cp s3://vllm-wheels/${RELEASE_VERSION}+cu130/vllm-${RELEASE_VERSION}+cu130-cp38-abi3-manylinux1_x86_64.whl .
\`\`\`

To download and upload the image:

\`\`\`
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-x86_64 vllm/vllm-openai:x86_64
docker tag vllm/vllm-openai:x86_64 vllm/vllm-openai:latest-x86_64
docker tag vllm/vllm-openai:x86_64 vllm/vllm-openai:v${RELEASE_VERSION}-x86_64
docker push vllm/vllm-openai:latest-x86_64
docker push vllm/vllm-openai:v${RELEASE_VERSION}-x86_64

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}-aarch64 vllm/vllm-openai:aarch64
docker tag vllm/vllm-openai:aarch64 vllm/vllm-openai:latest-aarch64
docker tag vllm/vllm-openai:aarch64 vllm/vllm-openai:v${RELEASE_VERSION}-aarch64
docker push vllm/vllm-openai:latest-aarch64
docker push vllm/vllm-openai:v${RELEASE_VERSION}-aarch64

docker manifest rm vllm/vllm-openai:latest
docker manifest create vllm/vllm-openai:latest vllm/vllm-openai:latest-x86_64 vllm/vllm-openai:latest-aarch64
docker manifest create vllm/vllm-openai:v${RELEASE_VERSION} vllm/vllm-openai:v${RELEASE_VERSION}-x86_64 vllm/vllm-openai:v${RELEASE_VERSION}-aarch64
docker manifest push vllm/vllm-openai:latest
docker manifest push vllm/vllm-openai:v${RELEASE_VERSION}
\`\`\`
EOF 
