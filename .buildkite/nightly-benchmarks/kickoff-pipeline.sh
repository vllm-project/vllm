#!/usr/bin/env bash

# NOTE(simon): this script runs inside a buildkite agent with CPU only access.
set -euo pipefail

# Install system packages
apt update
apt install -y curl jq
# install yq
add-apt-repository ppa:rmescandon/yq -y && apt update && apt install yq -y

# Install minijinja for templating
curl -sSfL https://github.com/mitsuhiko/minijinja/releases/latest/download/minijinja-cli-installer.sh | sh
source $HOME/.cargo/env

# If BUILDKITE_PULL_REQUEST != "false", then we check the PR labels using curl and jq
if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
  PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/vllm/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')

  touch final.yaml

  if [[ $PR_LABELS == *"perf-benchmarks"* ]]; then
    echo "This PR has the 'perf-benchmarks' label. Proceeding with the performance benchmarks."
    yq -n 'load("final.yaml") *+ load(".buildkite/nightly-benchmarks/benchmark-pipeline.yaml")' > final.yaml
  fi

  cat final.yaml

  if [[ $PR_LABELS == *"nightly-benchmarks"* ]]; then
    echo "This PR has the 'nightly-benchmark' label. Proceeding with the nightly benchmarks."
    yq -n 'load("final.yaml") *+ load(".buildkite/nightly-benchmarks/nightly-pipeline.yaml")' > final.yaml
  fi

  cat final.yaml

  buildkite-agent pipeline upload final.yaml

fi
