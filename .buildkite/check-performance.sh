#!/bin/bash
set -euox pipefail

if [[ -z "$BUILDKITE_PULL_REQUEST" ]]; then
  echo "Skipping performance regression check: not a PR."
  exit 0
fi
(which wget && which curl) || (apt-get update && apt-get install -y wget curl)

run_benchmark() {
  python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-3B-Instruct $@ &
  server_pid=$!

  # Wait for server to start, timeout after 600 seconds
  timeout 180 bash -c 'until curl localhost:8000/v1/models; do sleep 4; done' || exit 1
  python3 ../benchmarks/benchmark_serving.py \
      --backend vllm \
      --dataset-name sharegpt \
      --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
      --model meta-llama/Llama-3.2-3B-Instruct \
      --num-prompts 100 \
      --save-result \
      2>&1 | tee benchmark_serving.txt
  # Wait for graceful exit
  kill $server_pid
}

# Compare against *latest* PR base commit. If a rebase happens in the PR,
# compare against current "rebased" base commit. 
BASE_BRANCH="$BUILDKITE_PULL_REQUEST_BASE_BRANCH"
# Strip fork username from BUILDKITE_BRANCH
CURRENT_BRANCH=${BUILDKITE_BRANCH#*:}

# Avoid hard links to vllm-project that would break forks.
git branch | cat
echo "$BUILDKITE_COMMIT"
# Initial state of the repo is dirty with main branch changes (?)
git restore .
# TODO auto-rebase if no conflicts are detected?
git fetch origin "$BASE_BRANCH" >/dev/null 2>&1
# Buildkite detached head state prevents 'merge-base' from finding common ancestor.
git remote add pr "$BUILDKITE_PULL_REQUEST_REPO" || echo "Remote already present: testing V1."
git fetch pr >/dev/null 2>&1
git switch "$CURRENT_BRANCH" >/dev/null 2>&1

# Find the common ancestor between PR and base/main
BASE_COMMIT=$(git merge-base "origin/$BASE_BRANCH" "$BUILDKITE_COMMIT" || echo "") 

if [[ -z "$BASE_COMMIT" ]]; then
  echo "Unable to determine PR base commit! Make sure 'origin' is set and pointing to the right remote." >&2
  exit 1
fi
echo "Using merge-base commit ($BASE_COMMIT) as base reference"

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Base Commit vs. PR Commit
echo "--- Testing BASE commit ($BASE_COMMIT)"
git checkout -q "$BASE_COMMIT"

# Extra arguments passed to the script are used here to spin up the server.
run_benchmark $@ && mv benchmark_serving.txt benchmark_base.txt

# Test PR commit
echo "--- Testing PR commit ($BUILDKITE_COMMIT)"
git switch "$CURRENT_BRANCH" >/dev/null 2>&1

run_benchmark $@ && mv benchmark_serving.txt benchmark_pr.txt
rm ShareGPT_V3_unfiltered_cleaned_split.json

# Compare results. Run the comparison 3 times to avoid jitter of a single run.
FAILURES=0
ATTEMPTS=3

set +e # do not quit on first error
for ((i=1; i<=ATTEMPTS; i++)); do
  echo "Attempt $i/$ATTEMPTS:"
  if ! python3 compare_benchmarks.py benchmark_base.txt benchmark_pr.txt; then
    ((FAILURES++))
  fi
done

# Final decision
if [[ "$FAILURES" -eq "$ATTEMPTS" ]]; then
  echo "🚨 Performance regression detected in all $ATTEMPTS attempts!"
  exit 1
elif [[ "$FAILURES" -gt 0 ]]; then
  echo "⚠️  Regression seen in $FAILURES/$ATTEMPTS attempts (possible flakiness)"
  exit 0
else
  echo "✅ No performance regression detected!"
  exit 0
fi