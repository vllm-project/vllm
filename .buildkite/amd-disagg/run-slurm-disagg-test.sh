#!/bin/bash
# =============================================================================
# run-slurm-disagg-test.sh — foreground (login-node) submitter for the disagg
# P/D gate. The single command the Buildkite step runs.
# -----------------------------------------------------------------------------
# Thin glue: submits run_xPyD_disagg.slurm. By default (WAIT=0, Spur-safe) it
# fire-and-forgets and returns 0 on a good submit; with WAIT=1 it polls the
# scheduler and exits with the job's exit code so a CI step pass/fails.
# stdout carries ONLY the SLURM job info
# (image/nodes line, job id, log path); after sbatch nothing else is written to
# stdout — the full job log persists at the --output path and pass/fail
# diagnostics go to stderr. The actual work lives in run_xPyD_disagg.slurm: it
# selects nodes,
# fans out one container per node via a single srun, and hands off to
# vllm_disagg.sh (rank-based prefill/decode self-select).
#
# Default target: 1P1D TP8 (NODES=2), pinned image v0.23.0, accuracy gate.
#
# Spur usage (fire-and-forget; the default here):
#   bash run-slurm-disagg-test.sh                       # 1P1D TP8
#   WIDE_EP_MODE=1 bash run-slurm-disagg-test.sh         # 1P1D wide-EP
#   NODES=4 WIDE_EP_MODE=1 xP=2 yD=2 bash run-slurm-disagg-test.sh   # 2P2D EP
# Prints the job id + NFS log paths and returns immediately (does NOT poll
# squeue/sacct, which hang on Spur). Track the run via the printed `tail -f`.
#
# Classic-Slurm / CI usage (block until done, exit with the gate's pass/fail):
#   WAIT=1 bash run-slurm-disagg-test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/run_xPyD_disagg.slurm}"

# ---- knobs (override from the Buildkite step env) --------------------------
# Defaults tuned for the Spur AMD MI350X cluster: pinned v0.23.0 image (avoids
# :nightly build skew), /data NFS mount, and the Spur default partition (leave
# PARTITION empty so we don't pass --partition, which Spur rejects). Everything
# stays env-overridable for CI / other clusters.
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:v0.23.0}"
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PARTITION="${SLURM_PARTITION:-}"               # empty -> Spur default partition
TIME_LIMIT="${SLURM_TIME_LIMIT:-01:30:00}"
WIDE_EP_MODE="${WIDE_EP_MODE:-0}"              # 0 -> 1P1D TP8 (default); 1 -> wide-EP
xP="${xP:-1}"
yD="${yD:-1}"
RUN_AFTER_HEALTH="${RUN_AFTER_HEALTH:-accuracy}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-5400}"   # P/D bring-up budget (big models load slowly)
SHARED_MOUNT="${SHARED_MOUNT:-/data}"
LOG_ROOT="${LOG_ROOT:-${SHARED_MOUNT}/${USER:-$(whoami)}/disagg_logs}"
DRY_RUN="${DRY_RUN:-0}"
MORIIO_READ_MODE="${MORIIO_READ_MODE:-0}"

# Spur has no working srun/squeue/sacct in the login shell (they hang), so we do
# NOT block on them by default (WAIT=0); just submit and return. Set WAIT=1 to
# opt into the classic CI poll-until-done behavior (timeout-guarded so it
# degrades gracefully if the scheduler tools are unavailable).
# NB: --output/--error are NOT set here on purpose -- see the submit block below.
WAIT="${WAIT:-0}"

# Front door: toy (in-container proxy, default) | vllm-router (external container).
ROUTER_TYPE="${ROUTER_TYPE:-toy}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly}"
# Dry-run only validates wiring; cap its walltime low so it never holds the queue.
[[ "${DRY_RUN}" == "1" ]] && TIME_LIMIT="${SLURM_TIME_LIMIT:-00:10:00}"

mkdir -p "${LOG_ROOT}"

# IMPORTANT (Spur): passing sbatch CLI resource flags (--nodes/--gres/--time/
# --output/... ) makes the job PENDING forever even though Spur accepts them and
# prints a job id. The proven-working invocation on this cluster passes NO CLI
# resource flags and relies entirely on the #SBATCH directives baked into
# run_xPyD_disagg.slurm (nodes/gres/time/chdir + /tmp output/error, all
# Spur-correct) plus env inherited from THIS shell. So we:
#   * EXPORT the knobs (Spur forwards the shell env to the job; there is no
#     --export flag), and
#   * submit the job script bare, parsing the id from "Submitted batch job N".
#
# DISAGG_SCRIPTS_DIR is CRITICAL: it's the host scripts dir bind-mounted into the
# container (launcher, cluster.sh, models.yaml, proxy). Default to this script's
# own dir so the wrapper is self-contained regardless of the caller's CWD.
export DISAGG_SCRIPTS_DIR="${DISAGG_SCRIPTS_DIR:-${SCRIPT_DIR}}"
export IMAGE WIDE_EP_MODE xP yD GPUS_PER_NODE RUN_AFTER_HEALTH HEALTH_TIMEOUT_S
export SHARED_MOUNT LOG_ROOT DRY_RUN MORIIO_READ_MODE
export ROUTER_TYPE ROUTER_PORT VLLM_ROUTER_IMAGE

# Non-default resource requests can't ride CLI flags (they wedge Spur), so we
# patch the relevant #SBATCH directive(s) on a throwaway /tmp copy of the job
# script instead. Node count is the common one (EP wants 4). PARTITION has no
# Spur equivalent and is ignored.
[[ -n "${PARTITION}" ]] && echo "[slurm-submit] NOTE: PARTITION='${PARTITION}' ignored (Spur sbatch has no --partition)" >&2

SUBMIT_SCRIPT="${JOB_SCRIPT}"
FILE_NODES="$(grep -oE '^#SBATCH[[:space:]]+--nodes=[0-9]+' "${JOB_SCRIPT}" | grep -oE '[0-9]+' | head -n1)"
if [[ -n "${NODES}" && -n "${FILE_NODES}" && "${NODES}" != "${FILE_NODES}" ]]; then
    SUBMIT_SCRIPT="/tmp/$(basename "${JOB_SCRIPT%.slurm}")-n${NODES}-$$.slurm"
    sed -E "s/^#SBATCH([[:space:]]+)--nodes=[0-9]+/#SBATCH\\1--nodes=${NODES}/" "${JOB_SCRIPT}" > "${SUBMIT_SCRIPT}"
    echo "[slurm-submit] node count ${FILE_NODES}->${NODES}: submitting patched copy ${SUBMIT_SCRIPT}" >&2
fi

# Job name comes from the script's #SBATCH --job-name (SLURM sets SLURM_JOB_NAME
# from it), which is what the in-job `exec > $LOG_ROOT/${SLURM_JOB_NAME}-...`
# redirect uses -- so parse it here to compute the right NFS log path.
JOB_NAME="$(grep -oE '^#SBATCH[[:space:]]+--job-name=[^[:space:]]+' "${JOB_SCRIPT}" | sed -E 's/.*--job-name=//' | head -n1)"
JOB_NAME="${JOB_NAME:-vllm-disagg-pd}"

echo "[slurm-submit] image=${IMAGE} nodes=${NODES} gpus/node=${GPUS_PER_NODE} mode=$([[ ${WIDE_EP_MODE} == 0 ]] && echo tp || echo ep) router=${ROUTER_TYPE}"
SUBMIT_OUT="$(sbatch "${SUBMIT_SCRIPT}")"
echo "${SUBMIT_OUT}"
# "Submitted batch job 114" -> 114 (last integer on the line).
JOB_ID="$(printf '%s\n' "${SUBMIT_OUT}" | grep -oE '[0-9]+' | tail -n1 || true)"
if [[ -z "${JOB_ID}" ]]; then
    echo "[slurm-submit] ERROR: could not parse a job id from sbatch output above" >&2
    exit 1
fi
echo "[slurm-submit] submitted job ${JOB_ID}"

# The real per-job log is written to NFS from inside the job body by
# run_xPyD_disagg.slurm (`exec > "$LOG_ROOT/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"`).
# SLURM_JOB_NAME resolves to our --job-name, so we can compute the path here. The
# per-role server/proxy/bench logs land under ${LOG_ROOT}/${JOB_ID}/.
LOG_FILE="${LOG_ROOT}/${JOB_NAME}-${JOB_ID}.log"
LOG_DIR="${LOG_ROOT}/${JOB_ID}"
echo "[slurm-submit] job log:  ${LOG_FILE}"
echo "[slurm-submit] role logs: ${LOG_DIR}/"

# --- Spur-safe by default: fire-and-forget --------------------------------------
# squeue/sacct/srun hang in the Spur login shell, so we do NOT poll unless the
# caller opts in with WAIT=1 (classic CI behavior). In fire-and-forget mode we
# exit 0 on a successful submit; check the job log for the PASS/FAIL gate verdict.
if [[ "${WAIT}" != "1" ]]; then
    echo "[slurm-submit] submitted (WAIT=0, not polling). Track with:" >&2
    echo "  tail -f ${LOG_FILE}" >&2
    echo "  grep -aE '(PASS|FAIL): |exact_match' ${LOG_FILE}" >&2
    exit 0
fi

# --- WAIT=1: poll until the job leaves the queue (timeout-guarded) --------------
# Every scheduler call is wrapped in `timeout` so a hung/absent squeue/sacct
# (e.g. on Spur) degrades to a break instead of blocking the caller forever.
echo "[slurm-submit] WAIT=1: polling scheduler until job ${JOB_ID} finishes" >&2
while :; do
    STATE=$(timeout 15 squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || echo "")
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
STATE=$(timeout 15 sacct -j "${JOB_ID}" -n -o State 2>/dev/null | awk 'NR==1{print $1}')
read -r RC SIG < <(timeout 15 sacct -j "${JOB_ID}" -n -o ExitCode 2>/dev/null \
    | awk -F: 'NR==1{gsub(/[^0-9]/,"",$1); gsub(/[^0-9]/,"",$2); print $1+0, $2+0; exit}')
RC="${RC:-0}"; SIG="${SIG:-0}"
[[ "${RC}" == "0" && "${SIG}" != "0" ]] && RC=$((128 + SIG))

# If sacct is unavailable (Spur), fall back to the gate verdict in the job log.
if [[ -z "${STATE:-}" ]]; then
    if grep -aqE '^PASS: |PASS: ' "${LOG_FILE}" 2>/dev/null; then STATE="COMPLETED"; RC=0
    elif grep -aqE '^FAIL: |FAIL: ' "${LOG_FILE}" 2>/dev/null; then STATE="FAILED"; RC=1; fi
fi

case "${STATE:-}" in
    COMPLETED) : ;;                                  # genuine success (incl. accuracy PASS)
    "")        echo "[slurm-submit] WARN: no scheduler state for ${JOB_ID}; treating as failure" >&2
               [[ "${RC}" == "0" ]] && RC=1 ;;
    *)         echo "[slurm-submit] state=${STATE} is not COMPLETED -> failing" >&2
               [[ "${RC}" == "0" ]] && RC=1 ;;
esac

# Surface the accuracy gate verdict (if any) from the job log — to stderr.
GATE_LINE=$(grep -aE '(PASS|FAIL): ' "${LOG_FILE}" 2>/dev/null | tail -n1 || true)
[[ -n "${GATE_LINE}" ]] && echo "[slurm-submit] gate: ${GATE_LINE}" >&2

echo "[slurm-submit] job ${JOB_ID} finished: state=${STATE:-unknown} exit=${RC}" >&2
exit "${RC}"
