#!/usr/bin/env bash
# Ornith XPU graphs canary ladder: A (eager) -> B (PIECEWISE) -> C (FA-in-graph
# FULL), stop-on-fail. Each arm runs full correctness smokes + perf via
# smoke_ornith.sh. On a graph-arm failure, re-verifies the node with a short
# eager smoke (never leave a broken graph server as the last state), then
# exits non-zero. Writes AB_COMPARE_<stamp>.{json,md} under results/.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${DIR}/results}"
mkdir -p "${RESULTS_DIR}"
export STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
ARMS="${ARMS:-A B C}"

declare -A ARM_STATUS
failed_arm=""

for arm in ${ARMS}; do
  echo "=== ladder: arm ${arm} ==="
  if ARM="${arm}" PERF=1 bash "${DIR}/smoke_ornith.sh"; then
    ARM_STATUS[${arm}]="PASS"
  else
    ARM_STATUS[${arm}]="FAIL"
    failed_arm="${arm}"
    echo "=== arm ${arm} FAILED — stopping ladder ==="
    break
  fi
done

# Eager re-verify after a graph-arm failure: prove the node still serves
# correctly in the known-good config before finishing.
if [[ -n "${failed_arm}" && "${failed_arm}" != "A" ]]; then
  echo "=== eager re-verify after arm ${failed_arm} failure ==="
  if ARM=A PERF=0 STAMP="${STAMP}_reverify" bash "${DIR}/smoke_ornith.sh"; then
    echo "eager re-verify PASS — node healthy, graphs arm ${failed_arm} is the problem"
  else
    echo "eager re-verify FAIL — NODE/WEIGHTS PROBLEM, investigate before anything else"
  fi
fi

# Comparison report over whatever arms completed.
"${PYTHON_BIN}" "${DIR}/gen_report.py" "${RESULTS_DIR}" "${STAMP}"

echo "=== ladder summary ==="
for arm in ${ARMS}; do
  echo "arm ${arm}: ${ARM_STATUS[${arm}]:-SKIPPED}"
done
[[ -z "${failed_arm}" ]] || exit 1
