#!/bin/bash
set -euo pipefail

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${GITHUB_SHA:-local}}
RESULT_ROOT=${RESULT_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.benchmarks/ci/$RUN_ID}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}
MODE=${PERFGATE_MODE:-report}
REPORT_FILE=${PERFGATE_REPORT_FILE:-$RESULT_ROOT/perfgate_report.md}
STAGE1_CURRENT=${PERFGATE_STAGE1_CURRENT_FILE:-$RESULT_ROOT/submissions/$RUN_ID/run_leaderboard.json}

set +e
"${PYTHON_BIN:-python}" -m vllm_hust_benchmark.perfgate compare \
  --current "$STAGE1_CURRENT" \
  --baseline "$PERFGATE_BASELINE_FILE" \
  --fork-point "${FORK_POINT:-}" \
  --report-file "$REPORT_FILE" \
  --mode enforce
rc=$?
set -e

if grep -q '\*\*Overall: PASS\*\*' "$REPORT_FILE" 2>/dev/null; then
  result=pass
elif grep -q '\*\*Overall: FAIL\*\*' "$REPORT_FILE" 2>/dev/null; then
  result=fail
else
  result=unknown
fi
{
  echo "PERFGATE_STAGE1_RESULT=$result"
  echo "PERFGATE_REPORT_FILE=$REPORT_FILE"
} >> "$GITHUB_ENV"

if [[ "$rc" -ne 0 ]]; then
  echo "Stage 1 performance gate result: $result (exit $rc); final perfgate comparison/report step will decide job status."
fi
exit 0
