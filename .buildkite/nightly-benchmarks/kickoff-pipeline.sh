#!/usr/bin/env bash

set -euo pipefail

# Install system packages
apt update
apt install -y curl jq

# Install minijinja for templating
curl -sSfL https://github.com/mitsuhiko/minijinja/releases/latest/download/minijinja-cli-installer.sh | sh
source $HOME/.cargo/env

# If BUILDKITE_PULL_REQUEST != "false", then we check the PR labels using curl and jq
if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
  PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/vllm/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')

  if [[ $PR_LABELS == *"perf-benchmarks"* ]]; then
    echo "This PR has the 'perf-benchmarks' label. Proceeding with the nightly benchmarks."
  else
    echo "This PR does not have the 'perf-benchmarks' label. Skipping the nightly benchmarks."
    exit 0
  fi
fi

# Upload sample.yaml
buildkite-agent pipeline upload .buildkite/nightly-benchmarks/sample.yaml
