#!/bin/bash
# Upload a trigger step that starts a perf-eval build for this commit.
# Generated at runtime so the build title can carry the run date, e.g.
#   [Nvidia] Nightly run 2026-10-07: commit <sha>
#
# Usage: trigger-perf-eval.sh cuda|rocm
set -euo pipefail

case "${1:-}" in
  cuda)
    vendor="Nvidia"
    image_env="VLLM_IMAGE_CUDA"
    image_tag="${BUILDKITE_COMMIT}-x86_64"
    ;;
  rocm)
    vendor="AMD"
    image_env="VLLM_IMAGE_ROCM"
    image_tag="${BUILDKITE_COMMIT}-rocm"
    ;;
  *)
    echo "Usage: $0 cuda|rocm" >&2
    exit 1
    ;;
esac

if [[ "${NIGHTLY:-0}" == "1" ]]; then
  run_type="Nightly run"
else
  run_type="Manual run"
fi

message="[${vendor}] ${run_type} $(date -u +%Y-%m-%d): commit ${BUILDKITE_COMMIT}"

buildkite-agent pipeline upload --no-interpolation <<EOF
steps:
  - trigger: "perf-eval"
    label: "perf-eval (${vendor})"
    async: true
    soft_fail: true
    build:
      branch: "main"
      message: "${message}"
      env:
        NIGHTLY: "1" # run all perf-eval workloads
        VLLM_COMMIT: "${BUILDKITE_COMMIT}"
        ${image_env}: "public.ecr.aws/q9t5s3a7/vllm-release-repo:${image_tag}"
EOF
