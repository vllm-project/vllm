#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Promote one stack's release artifacts, built and GPU-smoked by amd-release, without rebuilding:
#   publish.sh <stack> wheels   rocm/vllm-ci:<stack>-<commit>-wheels -> s3://vllm-wheels/rocm/... + indices
#   publish.sh <stack> images   rocm/vllm-ci:<stack>-<commit>{,-base} -> ECR; with NIGHTLY=1 vllm/vllm-openai-rocm
set -euo pipefail
S=$1 C=${BUILDKITE_COMMIT:?} SRC=rocm/vllm-ci:$1-${BUILDKITE_COMMIT:?}
ECR=${ECR:-public.ecr.aws/q9t5s3a7/vllm-release-repo} HUB=${HUB:-vllm/vllm-openai-rocm}
pin() { echo "$1@$(docker buildx imagetools inspect "$1" --format '{{.Manifest.Digest}}')"; }  # resolve once
variant() { docker buildx imagetools inspect "$1" --format '{{index .Image.Config.Labels "ai.vllm.rocm.variant"}}'; }
IMG=$(pin "$SRC"); V=$(variant "$IMG")       # the stack's label; images do not depend on the wheels step
case $2 in
wheels)
  rm -rf dist && echo "FROM $(pin "$SRC-wheels")" | docker buildx build -q -o dist -
  [[ $(basename dist/vllm-*.whl | grep -oE 'rocm[0-9]+' | tail -1) == "$V" ]]   # setup.py's rocm_version_to_variant
  bash .buildkite/scripts/upload-rocm-wheels.sh "$S" dist
  buildkite-agent annotate --style info --context "rocm-$S" \
    "ROCm $V: \`pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/$C/$V\`" ;;
images)
  RT=$(pin "$SRC-base")
  aws ecr-public get-login-password --region us-east-1 | docker login -u AWS --password-stdin "${ECR%%/*}"
  docker buildx imagetools create -t "$ECR:$C-$V" "$IMG"
  docker buildx imagetools create -t "$ECR:$C-$V-base" "$RT"
  buildkite-agent meta-data set "rocm-$S-variant" "$V"   # root index, publish-release-images.sh
  [[ ${NIGHTLY:-0} == 1 ]] || exit 0
  export DOCKERHUB_USERNAME=${DOCKERHUB_USERNAME:-vllmbot}           # cleanup-nightly-builds.sh needs it
  sfx=("-$V"); if [[ $S == rocm ]]; then sfx+=(""); fi                # the default stack owns the plain tags
  for s in "${sfx[@]}"; do
    docker buildx imagetools create -t "$HUB:nightly$s" -t "$HUB:nightly$s-$C" "$IMG"
    docker buildx imagetools create -t "$HUB:base-nightly$s" -t "$HUB:base-nightly$s-$C" "$RT"
  done
  clean=.buildkite/scripts/cleanup-nightly-builds.sh
  bash "$clean" "nightly-$V-" "$HUB"; bash "$clean" "base-nightly-$V-" "$HUB"
  if [[ $S == rocm ]]; then
    bash "$clean" "nightly-" "$HUB" "nightly-rocm"
    bash "$clean" "base-nightly-" "$HUB" "base-nightly-rocm"
  fi ;;
*) exit 2 ;;
esac
