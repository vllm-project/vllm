#!/bin/bash

set -ex

# Get tag variant from argument, default to empty if not provided, should be something like "cu130".
# Due to limits in cleanup script, we must move variants to use separate tags like "cu130-nightly",
# otherwise they will be cleaned up together with the main "nightly" tags.

TAG_VARIANT="$1"
if [ -n "$TAG_VARIANT" ]; then
    ORIG_TAG_SUFFIX="-$TAG_VARIANT"
    TAG_NAME="$TAG_VARIANT-nightly"
else
    ORIG_TAG_SUFFIX=""
    TAG_NAME="nightly"
fi

ORIG_TAG_NAME="$BUILDKITE_COMMIT"

echo "Pushing original tag $ORIG_TAG_NAME$ORIG_TAG_SUFFIX to new nightly tag name: $TAG_NAME"

# pull original arch-dependent images from AWS ECR Public
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:$ORIG_TAG_NAME-x86_64$ORIG_TAG_SUFFIX
docker pull public.ecr.aws/q9t5s3a7/vllm-release-repo:$ORIG_TAG_NAME-aarch64$ORIG_TAG_SUFFIX
# tag arch-dependent images
docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:$ORIG_TAG_NAME-x86_64$ORIG_TAG_SUFFIX vllm/vllm-openai:$TAG_NAME-x86_64
docker tag public.ecr.aws/q9t5s3a7/vllm-release-repo:$ORIG_TAG_NAME-aarch64$ORIG_TAG_SUFFIX vllm/vllm-openai:$TAG_NAME-aarch64
# push arch-dependent images to DockerHub
docker push vllm/vllm-openai:$TAG_NAME-x86_64
docker push vllm/vllm-openai:$TAG_NAME-aarch64
# push arch-independent manifest to DockerHub
docker manifest create vllm/vllm-openai:$TAG_NAME vllm/vllm-openai:$TAG_NAME-x86_64 vllm/vllm-openai:$TAG_NAME-aarch64 --amend
docker manifest create vllm/vllm-openai:$TAG_NAME-$BUILDKITE_COMMIT vllm/vllm-openai:$TAG_NAME-x86_64 vllm/vllm-openai:$TAG_NAME-aarch64 --amend
docker manifest push vllm/vllm-openai:$TAG_NAME
docker manifest push vllm/vllm-openai:$TAG_NAME-$BUILDKITE_COMMIT
