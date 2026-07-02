#!/bin/bash
# =============================================================================
# run-slurm-disagg-test.sh — foreground (login-node) submitter for the disagg
# P/D gate. The single command the Buildkite step runs.
# -----------------------------------------------------------------------------
# Thin CI glue: submits run_xPyD_disagg.slurm, waits for it, and exits with the
# job's exit code so the step pass/fails. stdout carries ONLY the SLURM job info
# (image/nodes line, job id, log path); after sbatch nothing else is written to
# stdout — the full job log persists at the --output path and pass/fail
# diagnostics go to stderr. The actual work lives in run_xPyD_disagg.slurm: it
# selects nodes,
# fans out one container per node via a single srun, and hands off to
# vllm_disagg.sh (rank-based prefill/decode self-select).
#
# Default target: 1P1D TP8 (NODES=2). For 2P2D EP:
#   NODES=4 WIDE_EP_MODE=1 xP=2 yD=2 bash run-slurm-disagg-test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/run_xPyD_disagg.slurm}"

# ---- knobs (override from the Buildkite step env) --------------------------
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly}"
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PARTITION="${SLURM_PARTITION:-amd-rccl}"
TIME_LIMIT="${SLURM_TIME_LIMIT:-01:30:00}"
WIDE_EP_MODE="${WIDE_EP_MODE:-0}"
xP="${xP:-1}"
yD="${yD:-1}"
RUN_AFTER_HEALTH="${RUN_AFTER_HEALTH:-accuracy}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-5400}"   # P/D bring-up budget (big models load slowly)
SHARED_MOUNT="${SHARED_MOUNT:-/shared_inference}"
LOG_ROOT="${LOG_ROOT:-${SHARED_MOUNT}/${USER:-$(whoami)}/disagg_logs}"
DRY_RUN="${DRY_RUN:-0}"
MORIIO_READ_MODE="${MORIIO_READ_MODE:-0}"

# Front door: toy (in-container proxy, default) | vllm-router (external container).
ROUTER_TYPE="${ROUTER_TYPE:-toy}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly}"
# Dry-run only validates wiring; cap its walltime low so it never holds the queue.
[[ "${DRY_RUN}" == "1" ]] && TIME_LIMIT="${SLURM_TIME_LIMIT:-00:10:00}"

mkdir -p "${LOG_ROOT}"

# Forward CI-relevant env into the SLURM job environment (run_xPyD_disagg.slurm
# reads these). DISAGG_SCRIPTS_DIR is CRITICAL: the Buildkite step runs this
# wrapper from the repo root, so the job's SLURM_SUBMIT_DIR fallback would be the
# repo root, not this scripts dir. We pin it to SCRIPT_DIR so the job bind-mounts
# the right host scripts (launcher, cluster.sh, models.yaml, proxy) into the
# container at CONTAINER_SCRIPTS.
EXPORTS="ALL"
EXPORTS+=",IMAGE=${IMAGE}"
EXPORTS+=",WIDE_EP_MODE=${WIDE_EP_MODE},xP=${xP},yD=${yD},GPUS_PER_NODE=${GPUS_PER_NODE}"
EXPORTS+=",RUN_AFTER_HEALTH=${RUN_AFTER_HEALTH},HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S}"
EXPORTS+=",SHARED_MOUNT=${SHARED_MOUNT},LOG_ROOT=${LOG_ROOT}"
EXPORTS+=",DISAGG_SCRIPTS_DIR=${SCRIPT_DIR}"
EXPORTS+=",DRY_RUN=${DRY_RUN}"
EXPORTS+=",MORIIO_READ_MODE=${MORIIO_READ_MODE}"
EXPORTS+=",ROUTER_TYPE=${ROUTER_TYPE},ROUTER_PORT=${ROUTER_PORT},VLLM_ROUTER_IMAGE=${VLLM_ROUTER_IMAGE}"

SBATCH_ARGS=(
    --parsable
    --nodes="${NODES}"
    --ntasks-per-node=1
    --gres="gpu:${GPUS_PER_NODE}"
    --time="${TIME_LIMIT}"
    --job-name="vllm-disagg-${BUILDKITE_BUILD_NUMBER:-local}"
    --output="${LOG_ROOT}/slurm-%j.log"
    --export="${EXPORTS}"
)
[[ -n "${PARTITION}" ]] && SBATCH_ARGS+=(--partition="${PARTITION}")

echo "[slurm-submit] image=${IMAGE} nodes=${NODES} gpus/node=${GPUS_PER_NODE} mode=$([[ ${WIDE_EP_MODE} == 0 ]] && echo tp || echo ep) router=${ROUTER_TYPE}"
JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${JOB_SCRIPT}")
echo "[slurm-submit] submitted job ${JOB_ID}${PARTITION:+ on ${PARTITION}}"

LOG_FILE="${LOG_ROOT}/slurm-${JOB_ID}.log"
echo "[slurm-submit] log: ${LOG_FILE}"

# Nothing past this point is written to stdout: only the SLURM job info above is
# printed there. We do NOT stream the job log (it persists at ${LOG_FILE}); the
# pass/fail diagnostics below go to stderr so stdout stays clean.
#
# Poll SLURM until the job leaves the queue.
while :; do
    STATE=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || echo "")
    [[ -z "${STATE}" ]] && break
    case "${STATE}" in COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL) break ;; esac
    sleep 15
done

# ---- determine pass/fail -----------------------------------------------------
# Derive the result from BOTH the final SLURM state AND the exit code. sacct
# reports the *main* job row's ExitCode as 0:0 even for a CANCELLED/TIMEOUT job
# (the signal only shows on the .batch/.<step> rows), so trusting ExitCode alone
# would mask an aborted run as success — a false green. We therefore:
#   1) read the main row's State + ExitCode (code:signal),
#   2) promote a signal-kill (code 0, signal !=0) to a non-zero rc, and
#   3) treat any non-COMPLETED terminal state as a failure.
# The GSM8K gate itself rides the exit code: vllm_disagg.sh's
# run_accuracy returns 1 when exact_match < ACCURACY_THRESHOLD, which propagates
# srun -> batch `exit` -> State=FAILED ExitCode=1:0 -> RC=1 here.
STATE=$(sacct -j "${JOB_ID}" -n -o State 2>/dev/null | awk 'NR==1{print $1}')
read -r RC SIG < <(sacct -j "${JOB_ID}" -n -o ExitCode 2>/dev/null \
    | awk -F: 'NR==1{gsub(/[^0-9]/,"",$1); gsub(/[^0-9]/,"",$2); print $1+0, $2+0; exit}')
RC="${RC:-0}"; SIG="${SIG:-0}"
[[ "${RC}" == "0" && "${SIG}" != "0" ]] && RC=$((128 + SIG))

case "${STATE:-}" in
    COMPLETED) : ;;                                  # genuine success (incl. accuracy PASS)
    "")        echo "[slurm-submit] WARN: no sacct state for ${JOB_ID}; treating as failure" >&2
               [[ "${RC}" == "0" ]] && RC=1 ;;
    *)         echo "[slurm-submit] state=${STATE} is not COMPLETED -> failing" >&2
               [[ "${RC}" == "0" ]] && RC=1 ;;
esac

# Surface the accuracy gate verdict (if any) from the job log — to stderr.
GATE_LINE=$(grep -aE '(PASS|FAIL): ' "${LOG_FILE}" 2>/dev/null | tail -n1 || true)
[[ -n "${GATE_LINE}" ]] && echo "[slurm-submit] gate: ${GATE_LINE}" >&2

echo "[slurm-submit] job ${JOB_ID} finished: state=${STATE:-unknown} exit=${RC}" >&2
exit "${RC}"
