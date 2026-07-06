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
# DISAGG_SCRIPTS_DIR is bind-mounted on compute nodes. The Buildkite checkout
# lives on the login node only — stage scripts onto shared NFS before sbatch.
DISAGG_SCRIPTS_STAGE="${DISAGG_SCRIPTS_STAGE:-/data/scratch/buildkite-agent}"
STAGED_DIR="${DISAGG_SCRIPTS_STAGE}/${BUILDKITE_COMMIT:-local}"
mkdir -p "${STAGED_DIR}"
cp -a "${SCRIPT_DIR}/." "${STAGED_DIR}/"
export DISAGG_SCRIPTS_DIR="${STAGED_DIR}"
echo "[slurm-submit] staged scripts for compute nodes: ${DISAGG_SCRIPTS_DIR}" >&2
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

# --- WAIT=1: poll NFS job log for PASS/FAIL (Spur-safe) -----------------------
# squeue/sacct are unreliable on Spur login nodes (empty/hang). Do not treat an
# empty squeue as "job finished" while the log has no gate verdict yet.
echo "[slurm-submit] WAIT=1: polling ${LOG_FILE} for gate (timeout ${TIME_LIMIT})" >&2
_h=0 _m=0 _s=0
IFS=: read -r _h _m _s <<< "${TIME_LIMIT}"
WAIT_DEADLINE=$(( $(date +%s) + 10#${_h}*3600 + 10#${_m}*60 + 10#${_s:-0} ))
unset -v _h _m _s

STATE=""
RC=1
while [[ $(date +%s) -lt ${WAIT_DEADLINE} ]]; do
    if grep -aqE '(PASS|FAIL): ' "${LOG_FILE}" 2>/dev/null; then
        if grep -aqE 'FAIL: ' "${LOG_FILE}" 2>/dev/null; then
            STATE="FAILED"
            RC=1
        else
            STATE="COMPLETED"
            RC=0
        fi
        break
    fi
    _sq=$(timeout 15 squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
    case "${_sq}" in
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL)
            STATE="${_sq}"
            RC=1
            break
            ;;
    esac
    unset -v _sq
    sleep 30
done

if [[ -z "${STATE}" ]]; then
    echo "[slurm-submit] WARN: no gate verdict before ${TIME_LIMIT}; failing" >&2
    RC=1
fi

# Surface the accuracy gate verdict (if any) from the job log — to stderr.
GATE_LINE=$(grep -aE '(PASS|FAIL): ' "${LOG_FILE}" 2>/dev/null | tail -n1 || true)
[[ -n "${GATE_LINE}" ]] && echo "[slurm-submit] gate: ${GATE_LINE}" >&2

echo "[slurm-submit] job ${JOB_ID} finished: state=${STATE:-unknown} exit=${RC}" >&2
exit "${RC}"
