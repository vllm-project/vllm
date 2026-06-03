#!/bin/bash
set -euo pipefail

COMMIT=${1:-${FORK_POINT:-${GITHUB_SHA:-}}}
BASELINE_BRANCH=${PERFGATE_BASELINE_BRANCH:-benchmark-baselines}
OUTPUT_DIR=${PERFGATE_BASELINE_OUTPUT_DIR:-${RUNNER_TEMP:-/tmp}/perfgate-baselines}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}

if [[ -z "$COMMIT" ]]; then
  echo "Usage: $0 <commit-sha> or set FORK_POINT/GITHUB_SHA" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR/branch"
if ! git ls-remote --exit-code --heads origin "$BASELINE_BRANCH" >/dev/null 2>&1; then
  echo "Perfgate baseline branch not found: $BASELINE_BRANCH" >&2
  exit 2
fi

git clone --depth 1 --branch "$BASELINE_BRANCH" "$(git remote get-url origin)" "$OUTPUT_DIR/branch"

baseline_file="$OUTPUT_DIR/branch/baselines/$COMMIT/run_leaderboard.json"
baseline_commit="$COMMIT"
if [[ ! -f "$baseline_file" ]]; then
  baseline_file="$OUTPUT_DIR/branch/latest-main.json"
  baseline_commit="latest-main"
fi

if [[ ! -f "$baseline_file" ]]; then
  echo "No perfgate baseline found for $COMMIT and latest-main is missing" >&2
  exit 2
fi

resolved_file="$OUTPUT_DIR/baseline-${COMMIT:0:8}.json"
cp "$baseline_file" "$resolved_file"
{
  echo "PERFGATE_BASELINE_FILE=$resolved_file"
  echo "PERFGATE_BASELINE_COMMIT=$baseline_commit"
} >> "$GITHUB_ENV"

echo "Fetched perfgate baseline: $baseline_commit -> $resolved_file"
