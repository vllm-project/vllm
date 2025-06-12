#!/bin/bash

set -ex

# Get release version and strip leading 'v' if present
RELEASE_VERSION=$(buildkite-agent meta-data get release-version | sed 's/^v//')

if [ -z "$RELEASE_VERSION" ]; then
  echo "Error: RELEASE_VERSION is empty. 'release-version' metadata might not be set or is invalid."
  exit 1
fi

buildkite-agent annotate --style 'info' --context 'release-workflow' << EOF
To download the wheel:
\`\`\`
aws s3 cp s3://vllm-wheels/${RELEASE_VERSION}/vllm-${RELEASE_VERSION}-cp38-abi3-manylinux1_x86_64.whl .
aws s3 cp s3://vllm-wheels/${RELEASE_VERSION}+cu126/vllm-${RELEASE_VERSION}+cu126-cp38-abi3-manylinux1_x86_64.whl .
aws s3 cp s3://vllm-wheels/${RELEASE_VERSION}+cu118/vllm-${RELEASE_VERSION}+cu118-cp38-abi3-manylinux1_x86_64.whl . 
\`\`\`

To download and upload the image:

\`\`\`
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT}
docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:${BUILDKITE_COMMIT} vllm/vllm-openai
docker tag vllm/vllm-openai vllm/vllm-openai:latest
docker tag vllm/vllm-openai vllm/vllm-openai:v${RELEASE_VERSION}
docker push vllm/vllm-openai:latest
docker push vllm/vllm-openai:v${RELEASE_VERSION}
\`\`\`
EOF 