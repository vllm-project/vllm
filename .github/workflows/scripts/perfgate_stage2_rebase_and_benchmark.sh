#!/bin/bash
set -euo pipefail

GITHUB_ENV=${GITHUB_ENV:-/dev/null}
ORIGINAL_REF=$(git rev-parse HEAD)
FORK_POINT=${FORK_POINT:-}
M2_COMMIT=$(git rev-parse origin/main 2>/dev/null || true)
RUN_ID_BASE=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${GITHUB_SHA:-local}}
STAGE2_RUN_ID=${PERFGATE_STAGE2_RUN_ID:-${RUN_ID_BASE}-stage2}
STAGE2_RESULT_ROOT=${PERFGATE_STAGE2_RESULT_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.benchmarks/ci/$STAGE2_RUN_ID}
CONFLICT_FILE=${PERFGATE_STAGE2_REBASE_CONFLICT_FILE:-$STAGE2_RESULT_ROOT/rebase-conflict.txt}
TEMP_BRANCH=${PERFGATE_STAGE2_BRANCH:-perfgate-stage2-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}}

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

read_env_value() {
  local name=$1
  local env_file=$2
  awk -v key="$name" '
    $0 == key {
      # Defensive no-op for malformed entries.
      next
    }
    index($0, key "=") == 1 {
      print substr($0, length(key) + 2)
      found = 1
      exit
    }
    index($0, key "<<") == 1 {
      delimiter = substr($0, length(key) + 3)
      value = ""
      while ((getline line) > 0) {
        if (line == delimiter) {
          print value
          found = 1
          exit
        }
        value = value (value == "" ? "" : "\n") line
      }
    }
    END { exit found ? 0 : 1 }
  ' "$env_file"
}

cleanup() {
  set +e
  if git rev-parse --verify REBASE_HEAD >/dev/null 2>&1 || [[ -d .git/rebase-merge || -d .git/rebase-apply ]]; then
    git rebase --abort >/dev/null 2>&1 || true
  fi
  git checkout --force "$ORIGINAL_REF" >/dev/null 2>&1 || true
  git branch -D "$TEMP_BRANCH" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$STAGE2_RESULT_ROOT"
git fetch origin main
M2_COMMIT=$(git rev-parse origin/main)
write_env PERFGATE_M2_COMMIT "$M2_COMMIT"

if [[ -n "$FORK_POINT" && "$FORK_POINT" == "$M2_COMMIT" ]]; then
  write_env PERFGATE_STAGE2_SKIPPED 1
  write_env PERFGATE_STAGE2_SKIP_REASON "fork-point is already latest main"
  echo "Stage 2 skipped: fork-point is already latest main"
  exit 0
fi

git checkout -B "$TEMP_BRANCH" "$ORIGINAL_REF"
set +e
git rebase origin/main >"$CONFLICT_FILE" 2>&1
rebase_rc=$?
set -e
if [[ "$rebase_rc" -ne 0 ]]; then
  {
    echo
    echo "Conflicting files:"
    git diff --name-only --diff-filter=U || true
  } >> "$CONFLICT_FILE"
  write_env PERFGATE_STAGE2_REBASE_CONFLICT 1
  write_env PERFGATE_STAGE2_REBASE_CONFLICT_FILE "$CONFLICT_FILE"
  echo "Stage 2 rebase conflict recorded at $CONFLICT_FILE"
  exit 0
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" "wheel" "packaging>=24.2"
  VLLM_TARGET_DEVICE=empty VLLM_USE_PRECOMPILED=0 "$PYTHON_BIN" -m pip install -e . --no-build-isolation --no-deps
fi

run_stage2_benchmark() {
  local max_attempts=${NODE_ENV_RETRY_MAX_ATTEMPTS:-3}
  local retry_delay_seconds=${NODE_ENV_RETRY_DELAY_SECONDS:-30}
  local attempt=1
  while [[ "$attempt" -le "$max_attempts" ]]; do
    echo "[perfgate-stage2] benchmark attempt ${attempt}/${max_attempts}"
    set +e
    RUN_ID="$STAGE2_RUN_ID" RESULT_ROOT="$STAGE2_RESULT_ROOT" PUBLISH_TO_HF=0 PUBLISH_TO_BENCHMARK_REPO=0 \
      bash .github/workflows/scripts/run_ascend_benchmark_ci.sh
    local rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      return 0
    fi
    if [[ "$rc" -eq 86 && "$attempt" -lt "$max_attempts" ]]; then
      echo "Detected node-level Ascend runtime failure in Stage 2; retrying after ${retry_delay_seconds}s."
      bash .github/workflows/scripts/cleanup_ascend_ci_processes.sh || true
      sleep "$retry_delay_seconds"
      attempt=$((attempt + 1))
      continue
    fi
    return "$rc"
  done
}

run_stage2_benchmark

b1prime_file="$STAGE2_RESULT_ROOT/submissions/$STAGE2_RUN_ID/run_leaderboard.json"
if [[ ! -f "$b1prime_file" ]]; then
  echo "Stage 2 run_leaderboard.json not found: $b1prime_file" >&2
  exit 2
fi

stage2_env_file="$GITHUB_ENV.stage2"
rm -f "$stage2_env_file"
PERFGATE_BASELINE_OUTPUT_DIR="${RUNNER_TEMP:-/tmp}/perfgate-stage2-baseline" \
  PERFGATE_ALLOW_BASELINE_FALLBACK="${PERFGATE_ALLOW_STAGE2_BASELINE_FALLBACK:-0}" \
  GITHUB_ENV="$stage2_env_file" \
  bash .github/workflows/scripts/perfgate_fetch_baseline.sh "$M2_COMMIT"

stage2_baseline_available=$(read_env_value PERFGATE_BASELINE_AVAILABLE "$stage2_env_file" || echo "1")
if [[ "$stage2_baseline_available" != "1" ]]; then
  stage2_unavailable_reason=$(read_env_value PERFGATE_BASELINE_UNAVAILABLE_REASON "$stage2_env_file" || echo "Stage 2 M2 baseline is unavailable")
  write_env PERFGATE_STAGE2_NOT_RUN_REASON "Stage 2 M2 baseline is unavailable: $stage2_unavailable_reason"
  echo "Stage 2 comparison not run: $stage2_unavailable_reason"
  exit 0
fi

stage2_baseline_file=$(read_env_value PERFGATE_BASELINE_FILE "$stage2_env_file")
stage2_baseline_commit=$(read_env_value PERFGATE_BASELINE_COMMIT "$stage2_env_file")
stage2_baseline_source=$(read_env_value PERFGATE_BASELINE_SOURCE "$stage2_env_file")
write_env PERFGATE_STAGE2_B1PRIME_FILE "$b1prime_file"
write_env PERFGATE_STAGE2_M2_BASELINE_FILE "$stage2_baseline_file"
write_env PERFGATE_STAGE2_M2_BASELINE_COMMIT "$stage2_baseline_commit"
write_env PERFGATE_STAGE2_M2_BASELINE_SOURCE "$stage2_baseline_source"
write_env PERFGATE_STAGE2_REBASE_CONFLICT 0
write_env PERFGATE_STAGE2_SKIPPED 0

echo "Stage 2 benchmark ready: $b1prime_file vs $stage2_baseline_file"
