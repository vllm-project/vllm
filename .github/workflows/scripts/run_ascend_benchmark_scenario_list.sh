#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT}
RUN_ID_BASE=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-$(printf '%s' "${TARGET_REPO_SHA:-${GITHUB_SHA:-local}}" | cut -c1-8)}
RESULT_ROOT_BASE=${RESULT_ROOT:-$VLLM_HUST_REPO/.benchmarks/ci/$RUN_ID_BASE}
SCENARIOS_RAW=${BENCH_SCENARIOS:-${BENCH_SCENARIO:-random-online}}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}

trim() {
  local value=$1
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

scenario_slug() {
  printf '%s' "$1" | tr -c '[:alnum:]._-' '-'
}

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

scenarios=()
SCENARIOS_RAW=${SCENARIOS_RAW//$'\n'/,}
IFS=',' read -r -a raw_scenarios <<< "$SCENARIOS_RAW"
for raw_scenario in "${raw_scenarios[@]}"; do
  scenario=$(trim "$raw_scenario")
  if [[ -n "$scenario" ]]; then
    scenarios+=("$scenario")
  fi
done

if [[ "${#scenarios[@]}" -le 1 ]]; then
  bash .github/workflows/scripts/run_ascend_benchmark_ci.sh
  exit $?
fi

if [[ -n "${SAME_SPEC_SPEC_FILE:-}" ]]; then
  echo "SAME_SPEC_SPEC_FILE cannot be combined with BENCH_SCENARIOS because each scenario must resolve its own perfgate spec." >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT_BASE"
summary_file="$RESULT_ROOT_BASE/multi_scenario_results.tsv"
printf 'scenario\trun_id\tresult_root\traw_result\tsubmission_dir\texit_code\n' > "$summary_file"

overall_exit_code=0
for scenario in "${scenarios[@]}"; do
  slug=$(scenario_slug "$scenario")
  scenario_run_id="${RUN_ID_BASE}-${slug}"
  scenario_result_root="${RESULT_ROOT_BASE}/${slug}"
  scenario_submission_dir="${scenario_result_root}/submissions/${scenario_run_id}"
  scenario_raw_result="${scenario_result_root}/raw_benchmark.json"

  echo "::group::Ascend benchmark scenario: ${scenario}"
  set +e
  BENCH_SCENARIO="$scenario" \
    BENCH_SCENARIOS="$scenario" \
    BENCH_SCENARIO_COUNT=1 \
    RUN_ID="$scenario_run_id" \
    RESULT_ROOT="$scenario_result_root" \
    RAW_RESULT_FILE="$scenario_raw_result" \
    SUBMISSIONS_ROOT="${scenario_result_root}/submissions" \
    SUBMISSION_DIR="$scenario_submission_dir" \
    AGGREGATE_OUTPUT_DIR="${scenario_result_root}/leaderboard-data" \
    SERVER_LOG="${scenario_result_root}/server.log" \
    RUNNER_PREFLIGHT_FAILURE_FILE="${scenario_result_root}/runner_preflight_failure.txt" \
    DIAGNOSTICS_DIR="${scenario_result_root}/diagnostics" \
    NODE_ENV_FAILURE_FILE="${scenario_result_root}/node_env_failure.txt" \
    SAME_SPEC_SPEC_FILE="" \
    bash .github/workflows/scripts/run_ascend_benchmark_ci.sh
  scenario_exit_code=$?
  set -e
  echo "::endgroup::"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scenario" \
    "$scenario_run_id" \
    "$scenario_result_root" \
    "$scenario_raw_result" \
    "$scenario_submission_dir" \
    "$scenario_exit_code" >> "$summary_file"

  if [[ "$scenario_exit_code" -ne 0 && "$overall_exit_code" -eq 0 ]]; then
    overall_exit_code=$scenario_exit_code
  fi
done

write_env BENCHMARK_MULTI_SCENARIO_SUMMARY_FILE "$summary_file"
echo "Multi-scenario benchmark summary: $summary_file"
exit "$overall_exit_code"
