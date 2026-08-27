#!/bin/bash

set -ex

ORIG_TAG_NAME="$BUILDKITE_COMMIT"
REPO="vllm/vllm-openai-xpu"

echo "Pushing original XPU tag ${ORIG_TAG_NAME}-xpu to nightly tags in ${REPO}"

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:"$ORIG_TAG_NAME"-x86_64-xpu

docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:"$ORIG_TAG_NAME"-x86_64-xpu ${REPO}:nightly-x86_64
docker push ${REPO}:nightly-x86_64

docker manifest rm ${REPO}:nightly || true
docker manifest rm ${REPO}:nightly-"$BUILDKITE_COMMIT" || true
docker manifest create ${REPO}:nightly ${REPO}:nightly-x86_64 --amend
docker manifest create ${REPO}:nightly-"$BUILDKITE_COMMIT" ${REPO}:nightly-x86_64 --amend
docker manifest push ${REPO}:nightly
docker manifest push ${REPO}:nightly-"$BUILDKITE_COMMIT"