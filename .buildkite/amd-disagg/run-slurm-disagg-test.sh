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
# Defaults tuned for the Spur AMD MI350X cluster
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:v0.23.0}"
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PARTITION="${SLURM_PARTITION:-}"
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

WAIT="${WAIT:-0}"

# ROUTER Type - defaults to vllm-router
ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly}"
# Dry-run only validates wiring; cap its walltime low so it never holds the queue.
[[ "${DRY_RUN}" == "1" ]] && TIME_LIMIT="${SLURM_TIME_LIMIT:-00:10:00}"

mkdir -p "${LOG_ROOT}"

# Spur - sbatch scheduler.
DISAGG_SCRIPTS_STAGE="${DISAGG_SCRIPTS_STAGE:-/data/scratch/buildkite-agent}"
STAGED_DIR="${DISAGG_SCRIPTS_STAGE}/${BUILDKITE_COMMIT:-local}"
mkdir -p "${STAGED_DIR}"
cp -rL --no-preserve=ownership,timestamps "${SCRIPT_DIR}/." "${STAGED_DIR}/"
chmod -R u+rwX "${STAGED_DIR}" 2>/dev/null || true
export DISAGG_SCRIPTS_DIR="${STAGED_DIR}"
echo "[slurm-submit] staged scripts for compute nodes: ${DISAGG_SCRIPTS_DIR}" >&2
export IMAGE MODEL_NAME WIDE_EP_MODE xP yD GPUS_PER_NODE RUN_AFTER_HEALTH HEALTH_TIMEOUT_S
export SHARED_MOUNT LOG_ROOT DRY_RUN MORIIO_READ_MODE
export ROUTER_TYPE ROUTER_PORT VLLM_ROUTER_IMAGE

# Model selection.
[[ -n "${MODEL_NAME:-}" ]] && export MODEL_NAME
[[ -n "${MODEL_DIR:-}" ]] && export MODEL_DIR

[[ -n "${PARTITION}" ]] && echo "[slurm-submit] NOTE: PARTITION='${PARTITION}' ignored (Spur sbatch has no --partition)" >&2

SUBMIT_SCRIPT="${JOB_SCRIPT}"
FILE_NODES="$(grep -oE '^#SBATCH[[:space:]]+--nodes=[0-9]+' "${JOB_SCRIPT}" | grep -oE '[0-9]+' | head -n1)"
if [[ -n "${NODES}" && -n "${FILE_NODES}" && "${NODES}" != "${FILE_NODES}" ]]; then
    SUBMIT_SCRIPT="/tmp/$(basename "${JOB_SCRIPT%.slurm}")-n${NODES}-$$.slurm"
    sed -E "s/^#SBATCH([[:space:]]+)--nodes=[0-9]+/#SBATCH\\1--nodes=${NODES}/" "${JOB_SCRIPT}" > "${SUBMIT_SCRIPT}"
    echo "[slurm-submit] node count ${FILE_NODES}->${NODES}: submitting patched copy ${SUBMIT_SCRIPT}" >&2
fi

# Job name comes from the script's #SBATCH --job-name (SLURM sets SLURM_JOB_NAME
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
if [[ "${WAIT}" != "1" ]]; then
    echo "[slurm-submit] submitted (WAIT=0, not polling). Track with:" >&2
    echo "  tail -f ${LOG_FILE}" >&2
    echo "  grep -aE '(PASS|FAIL): |exact_match' ${LOG_FILE}" >&2
    exit 0
fi

# --- WAIT=1: phase-aware poll of the NFS job log (scontrol-based) -------------
# Detection is phased so failures surface fast instead of waiting out the full
# walltime:
#   1) submitted -> running : scontrol catches infra/scheduler failures
#                             (NODE_FAIL/BOOT_FAIL/CANCELLED/TIMEOUT/...) within
#                             one poll, or a stuck-PENDING/never-started job.
#   2) running   -> healthy : advance once every endpoint reports healthy (+ the
#                             vllm-router, when used); fail on bring-up errors or
#                             the health budget.
#   3) healthy   -> verdict : PASS/FAIL from the accuracy gate, capped.
# scontrol is the scheduler authority on this cluster
echo "[slurm-submit] WAIT=1: phase-aware poll of ${LOG_FILE} (timeout ${TIME_LIMIT})" >&2

_h=0 _m=0 _s=0
IFS=: read -r _h _m _s <<< "${TIME_LIMIT}"
WAIT_DEADLINE=$(( $(date +%s) + 10#${_h}*3600 + 10#${_m}*60 + 10#${_s:-0} ))
unset -v _h _m _s

# Per-phase budgets (all overridable from the Buildkite step env).
POLL_INTERVAL="${POLL_INTERVAL:-20}"
SUBMIT_GRACE_S="${SUBMIT_GRACE_S:-900}"                        # reach RUNNING within 15m
PENDING_MAX_S="${PENDING_MAX_S:-3600}"                          # tolerate 60m queued
HEALTH_PHASE_TIMEOUT_S="${HEALTH_PHASE_TIMEOUT_S:-$(( HEALTH_TIMEOUT_S + 900 ))}"
WORKLOAD_TIMEOUT_S="${WORKLOAD_TIMEOUT_S:-3600}"               # accuracy/bench cap

SENTINEL="${LOG_DIR}/.disagg_done"

# scontrol field extractor (authoritative here; timeout-guarded so a momentary
# scheduler stall can't wedge the poll). Returns the value or "".
job_field() {  # $1=jobid  $2=field  ->  value | ""
    timeout 15 scontrol show job "$1" 2>/dev/null \
        | grep -oE "$2=[^ ]+" | head -n1 | cut -d= -f2- || true
}
have() { grep -aqE "$1" "${LOG_FILE}" 2>/dev/null; }

STATE=""
RC=1
REASON=""
PHASE="submitted"
T_PHASE=$(date +%s)

while [[ $(date +%s) -lt ${WAIT_DEADLINE} ]]; do
    NOW=$(date +%s)

    # (1) Ultimate authority: terminal sentinel (holds rank-0 rc), then the
    #     explicit accuracy gate line. Honored regardless of phase.
    if [[ -f "${SENTINEL}" ]]; then
        RC="$(tr -dc '0-9' < "${SENTINEL}" 2>/dev/null || true)"; RC="${RC:-1}"
        if [[ "${RC}" == "0" ]]; then STATE="COMPLETED"; else STATE="FAILED"; fi
        REASON="sentinel"; break
    fi
    if have '(PASS|FAIL): '; then
        if have 'FAIL: '; then STATE="FAILED"; RC=1; else STATE="COMPLETED"; RC=0; fi
        REASON="gate"; break
    fi

    # (2) Scheduler state via scontrol: drives phase transitions and catches
    #     infra/scheduler failures fast.
    ST="$(job_field "${JOB_ID}" JobState)"
    case "${ST}" in
        RUNNING|COMPLETING)
            if [[ "${PHASE}" == "submitted" ]]; then PHASE="bringup"; T_PHASE="${NOW}"; fi
            ;;
        NODE_FAIL|BOOT_FAIL|CANCELLED|TIMEOUT|OUT_OF_MEMORY|DEADLINE|PREEMPTED)
            STATE="infra-${ST}"; RC=1
            REASON="scontrol Reason=$(job_field "${JOB_ID}" Reason)"; break
            ;;
        FAILED)
            # Ambiguous (infra crash vs a legit accuracy exit=1). A gate/sentinel
            # line would have won above, so classify by the phase we reached.
            case "${PHASE}" in
                submitted) STATE="infra-FAILED" ;;
                bringup)   STATE="server-failed" ;;
                *)         STATE="workload-failed" ;;
            esac
            RC=1; REASON="scontrol JobState=FAILED phase=${PHASE}"; break
            ;;
        COMPLETED)
            STATE="COMPLETED"; RC=0; REASON="scontrol COMPLETED"; break
            ;;
        PENDING|CONFIGURING|RESV_DEL_HOLD|REQUEUED)
            if (( NOW - T_PHASE > PENDING_MAX_S )); then
                STATE="infra-stuck-${ST}"; RC=1; REASON="queued > ${PENDING_MAX_S}s"; break
            fi
            ;;
        "")
            # scontrol lost the job (purged past MinJobAge) with no log verdict:
            # rely on the terminal markers above + the per-phase deadlines below.
            :
            ;;
    esac

    # (3) Log-driven phase progress + per-phase deadlines. Fast-fails hangs where
    #     the job is still RUNNING (so scontrol won't help) but bring-up/workload
    #     is stuck.
    case "${PHASE}" in
        submitted)
            # Job body appearing in the log is a second "it started" signal for
            # when scontrol is briefly empty.
            if have 'Selected node IPs|health-gate:|\[disagg-pd\]'; then
                PHASE="bringup"; T_PHASE="${NOW}"
            elif [[ -z "${ST}" ]] && (( NOW - T_PHASE > SUBMIT_GRACE_S )); then
                # scontrol can't confirm the job exists (empty state) AND no log
                # output within the grace window -> treat as a lost/failed launch.
                # NB: a genuinely-queued job reports PENDING via scontrol above and
                # is governed by PENDING_MAX_S (default 1h), not this grace window.
                STATE="infra-nostart"; RC=1; REASON="no scheduler state or log within ${SUBMIT_GRACE_S}s"; break
            fi
            ;;
        bringup)
            if have 'FAIL:|TIMEOUT waiting for|exited while waiting|Traceback \(most recent'; then
                STATE="server-failed"; RC=1; REASON="bring-up failure in log"; break
            fi
            NEED="$(grep -aoE 'waiting on [0-9]+' "${LOG_FILE}" 2>/dev/null | grep -oE '[0-9]+' | tail -n1 || true)"
            GOT="$(grep -acE '\] healthy: ' "${LOG_FILE}" 2>/dev/null || true)"; GOT="${GOT:-0}"
            ROUTER_OK=1
            if [[ "${ROUTER_TYPE}" == "vllm-router" ]]; then
                if ! have 'vllm-router healthy on'; then ROUTER_OK=0; fi
            fi
            if [[ -n "${NEED}" ]] && (( GOT >= NEED )) && (( ROUTER_OK == 1 )); then
                PHASE="workload"; T_PHASE="${NOW}"
            elif (( NOW - T_PHASE > HEALTH_PHASE_TIMEOUT_S )); then
                STATE="bringup-timeout"; RC=1; REASON="no healthy within ${HEALTH_PHASE_TIMEOUT_S}s"; break
            fi
            ;;
        workload)
            # PASS/FAIL handled at the top; only enforce the cap here.
            if (( NOW - T_PHASE > WORKLOAD_TIMEOUT_S )); then
                STATE="workload-timeout"; RC=1; REASON="no verdict within ${WORKLOAD_TIMEOUT_S}s"; break
            fi
            ;;
    esac

    sleep "${POLL_INTERVAL}"
done

if [[ -z "${STATE}" ]]; then
    echo "[slurm-submit] WARN: no verdict before ${TIME_LIMIT}; failing" >&2
    STATE="deadline"; RC=1; REASON="poll deadline"
fi

# Surface the accuracy gate verdict (if any) from the job log — to stderr.
GATE_LINE=$(grep -aE '(PASS|FAIL): ' "${LOG_FILE}" 2>/dev/null | tail -n1 || true)
[[ -n "${GATE_LINE}" ]] && echo "[slurm-submit] gate: ${GATE_LINE}" >&2

echo "[slurm-submit] job ${JOB_ID} finished: state=${STATE:-unknown} phase=${PHASE} exit=${RC} reason=${REASON:-}" >&2
exit "${RC}"
