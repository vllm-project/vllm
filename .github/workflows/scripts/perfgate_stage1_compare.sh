#!/bin/bash
set -euo pipefail

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${GITHUB_SHA:-local}}
RESULT_ROOT=${RESULT_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.benchmarks/ci/$RUN_ID}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}
MODE=${PERFGATE_MODE:-report}
REPORT_FILE=${PERFGATE_REPORT_FILE:-$RESULT_ROOT/perfgate_report.md}
STAGE1_CURRENT=${PERFGATE_STAGE1_CURRENT_FILE:-$RESULT_ROOT/submissions/$RUN_ID/run_leaderboard.json}

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

read_expected_spec_id() {
  if [[ -n "${PERFGATE_EXPECTED_SPEC_ID:-}" ]]; then
    printf '%s\n' "$PERFGATE_EXPECTED_SPEC_ID"
    return 0
  fi
  if [[ -n "${SAME_SPEC_SPEC_FILE:-}" && -f "$SAME_SPEC_SPEC_FILE" ]]; then
    "${PYTHON_BIN:-python}" - "$SAME_SPEC_SPEC_FILE" <<'PY'
import json
import sys
from pathlib import Path

print(str(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("id") or ""))
PY
    return 0
  fi
  printf '\n'
}

if [[ "${PERFGATE_BASELINE_AVAILABLE:-1}" != "1" || -z "${PERFGATE_BASELINE_FILE:-}" ]]; then
  reason=${PERFGATE_BASELINE_UNAVAILABLE_REASON:-Stage 1 baseline is unavailable}
  mkdir -p "$(dirname "$REPORT_FILE")"
  {
    echo "## Performance Gate Report"
    echo
    echo "**Overall: UNKNOWN**"
    echo
    echo "Stage 1 baseline is unavailable: $reason"
  } > "$REPORT_FILE"
  write_env PERFGATE_STAGE1_RESULT unknown
  write_env PERFGATE_REPORT_FILE "$REPORT_FILE"
  write_env PERFGATE_STAGE2_NOT_RUN_REASON "Stage 1 baseline is unavailable; Stage 2 was not run"
  echo "Stage 1 performance gate skipped: $reason"
  exit 0
fi

set +e
expected_spec_id=$(read_expected_spec_id)
args=(
  --current "$STAGE1_CURRENT"
  --baseline "$PERFGATE_BASELINE_FILE"
  --fork-point "${FORK_POINT:-}"
  --report-file "$REPORT_FILE"
  --mode enforce
)
if [[ -n "$expected_spec_id" ]]; then
  args+=(--expected-spec-id "$expected_spec_id")
fi
"${PYTHON_BIN:-python}" -m vllm_hust_benchmark.perfgate compare \
  "${args[@]}"
rc=$?
set -e

if grep -q '\*\*Overall: PASS\*\*' "$REPORT_FILE" 2>/dev/null; then
  result=pass
elif grep -q '\*\*Overall: FAIL\*\*' "$REPORT_FILE" 2>/dev/null; then
  result=fail
else
  result=unknown
fi
write_env PERFGATE_STAGE1_RESULT "$result"
write_env PERFGATE_REPORT_FILE "$REPORT_FILE"

if [[ "$rc" -ne 0 ]]; then
  echo "Stage 1 performance gate result: $result (exit $rc); final perfgate comparison/report step will decide job status."
fi
exit 0
