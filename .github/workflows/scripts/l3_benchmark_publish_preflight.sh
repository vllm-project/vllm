#!/bin/bash
set -euo pipefail

PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}
BENCHMARK_REPO_SLUG=${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}
SNAPSHOT_TARGET_BRANCH=${SNAPSHOT_TARGET_BRANCH:-main}
BENCHMARK_REPO_GH_TOKEN=${BENCHMARK_REPO_GH_TOKEN:-}
BENCHMARK_REPO_SSH_KEY=${BENCHMARK_REPO_SSH_KEY:-}

write_github_env() {
  local key=$1
  local value=$2
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s=%s\n' "$key" "$value" >>"$GITHUB_ENV"
  fi
}

if [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT skipped
  echo "L3 benchmark repository publish preflight: skipped"
  exit 0
fi

write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT running
write_github_env L3_BENCHMARK_PUBLISH_TARGET "${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}"

if [[ -z "$BENCHMARK_REPO_GH_TOKEN" && -z "$BENCHMARK_REPO_SSH_KEY" ]]; then
  write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT credential-missing
  echo "L3 benchmark repository publication is enabled, but no cross-repository write credential is available." >&2
  echo "Target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
  echo "Configure one of these secrets before enabling benchmark repo publish:" >&2
  echo "  - VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" >&2
  echo "  - VLLM_HUST_BENCHMARK_GH_TOKEN" >&2
  exit 2
fi

if [[ -n "$BENCHMARK_REPO_SSH_KEY" ]]; then
  write_github_env L3_BENCHMARK_PUBLISH_CREDENTIAL ssh-key
elif [[ -n "$BENCHMARK_REPO_GH_TOKEN" ]]; then
  write_github_env L3_BENCHMARK_PUBLISH_CREDENTIAL token
fi

write_github_env L3_BENCHMARK_PUBLISH_PREFLIGHT ok
echo "L3 benchmark repository publish preflight: ok"
echo "Target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}"
