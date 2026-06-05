#!/bin/bash
set -euo pipefail

COMMIT=${1:-${FORK_POINT:-${GITHUB_SHA:-}}}
BASELINE_BRANCH=${PERFGATE_BASELINE_BRANCH:-benchmark-baselines}
OUTPUT_DIR=${PERFGATE_BASELINE_OUTPUT_DIR:-${RUNNER_TEMP:-/tmp}/perfgate-baselines}
ALLOW_BASELINE_FALLBACK=${PERFGATE_ALLOW_BASELINE_FALLBACK:-0}
MODE=${PERFGATE_MODE:-report}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}

write_env() {
  local name=$1
  local value=$2
  local delimiter="EOF_${name}_$$_${RANDOM}"
  {
    echo "${name}<<${delimiter}"
    printf '%s\n' "$value"
    echo "$delimiter"
  } >> "$GITHUB_ENV"
}

baseline_unavailable() {
  local reason=$1
  echo "$reason" >&2
  if [[ "$MODE" == "report" ]]; then
    write_env PERFGATE_BASELINE_AVAILABLE 0
    write_env PERFGATE_BASELINE_COMMIT "$COMMIT"
    write_env PERFGATE_BASELINE_SOURCE unavailable
    write_env PERFGATE_BASELINE_UNAVAILABLE_REASON "$reason"
    echo "Perfgate baseline unavailable in report mode; continuing without baseline."
    exit 0
  fi
  exit 2
}

if [[ -z "$COMMIT" ]]; then
  echo "Usage: $0 <commit-sha> or set FORK_POINT/GITHUB_SHA" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR/branch"
if ! git ls-remote --exit-code --heads origin "$BASELINE_BRANCH" >/dev/null 2>&1; then
  baseline_unavailable "Perfgate baseline branch not found: $BASELINE_BRANCH"
fi

git clone --depth 1 --branch "$BASELINE_BRANCH" "$(git remote get-url origin)" "$OUTPUT_DIR/branch"

baseline_file="$OUTPUT_DIR/branch/baselines/$COMMIT/run_leaderboard.json"
baseline_commit="$COMMIT"
baseline_source="exact"
if [[ ! -f "$baseline_file" ]]; then
  if [[ "$ALLOW_BASELINE_FALLBACK" != "1" ]]; then
    baseline_unavailable "No exact perfgate baseline found for $COMMIT"
  fi
  baseline_file="$OUTPUT_DIR/branch/latest-main.json"
  baseline_commit="latest-main"
  baseline_source="latest-main-fallback"
fi

if [[ ! -f "$baseline_file" ]]; then
  baseline_unavailable "No perfgate baseline found for $COMMIT and latest-main is missing"
fi

resolved_file="$OUTPUT_DIR/baseline-${COMMIT:0:8}.json"
cp "$baseline_file" "$resolved_file"
write_env PERFGATE_BASELINE_FILE "$resolved_file"
write_env PERFGATE_BASELINE_AVAILABLE 1
write_env PERFGATE_BASELINE_COMMIT "$baseline_commit"
write_env PERFGATE_BASELINE_SOURCE "$baseline_source"

echo "Fetched perfgate baseline: $baseline_commit ($baseline_source) -> $resolved_file"
