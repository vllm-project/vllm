#!/bin/bash
set -euo pipefail

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${GITHUB_SHA:-local}}
RESULT_ROOT=${RESULT_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.benchmarks/ci/$RUN_ID}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}
MODE=${PERFGATE_MODE:-report}
REPORT_FILE=${PERFGATE_REPORT_FILE:-$RESULT_ROOT/perfgate_report.md}
STAGE1_CURRENT=${PERFGATE_STAGE1_CURRENT_FILE:-$RESULT_ROOT/submissions/$RUN_ID/run_leaderboard.json}

args=(
  compare2
  --stage1-current "$STAGE1_CURRENT"
  --stage1-baseline "$PERFGATE_BASELINE_FILE"
  --fork-point "${FORK_POINT:-}"
  --m2-commit "${PERFGATE_M2_COMMIT:-}"
  --report-file "$REPORT_FILE"
  --mode "$MODE"
)

if [[ "${PERFGATE_STAGE2_REBASE_CONFLICT:-0}" == "1" ]]; then
  args+=(--stage2-rebase-conflict)
  if [[ -n "${PERFGATE_STAGE2_REBASE_CONFLICT_FILE:-}" ]]; then
    args+=(--stage2-rebase-conflict-file "$PERFGATE_STAGE2_REBASE_CONFLICT_FILE")
  fi
elif [[ "${PERFGATE_STAGE2_SKIPPED:-0}" == "1" ]]; then
  args+=(--stage2-skipped --stage2-skip-reason "${PERFGATE_STAGE2_SKIP_REASON:-fork-point is already latest main}")
else
  args+=(--stage2-current "$PERFGATE_STAGE2_B1PRIME_FILE" --stage2-baseline "$PERFGATE_STAGE2_M2_BASELINE_FILE")
fi

set +e
"${PYTHON_BIN:-python}" -m vllm_hust_benchmark.perfgate "${args[@]}"
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
  echo "PERFGATE_RESULT=$result"
  echo "PERFGATE_REPORT_FILE=$REPORT_FILE"
} >> "$GITHUB_ENV"

if [[ "$MODE" == "report" ]]; then
  exit 0
fi
exit "$rc"
