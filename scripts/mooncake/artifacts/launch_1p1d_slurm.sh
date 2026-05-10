#!/usr/bin/env bash
set -euo pipefail

# Submit and run a manual 1P/1D bring-up on three Slurm nodes.
# P uses 1 node x 4 GPUs. D is DEP8, so it uses 2 nodes x 4 GPUs.
# The actual vLLM commands live in start_1p1d_prefill.sh and
# start_1p1d_decode.sh, which are expected to be visible on every node via
# the shared filesystem.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFILL_SCRIPT="${PREFILL_SCRIPT:-${SCRIPT_DIR}/start_1p1d_prefill.sh}"
DECODE_SCRIPT="${DECODE_SCRIPT:-${SCRIPT_DIR}/start_1p1d_decode.sh}"

PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-3:00:00}"
TOTAL_NODES="${TOTAL_NODES:-3}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
JOB_NAME="${JOB_NAME:-kimi-1p1d}"
PORT="${PORT:-8000}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "${LOG_DIR}"

  sbatch_args=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --nodes="${TOTAL_NODES}"
    --exclusive
    --gpus-per-node="${GPUS_PER_NODE}"
    --ntasks-per-node=1
    --time="${TIME_LIMIT}"
    --output="${LOG_DIR}/slurm-%j.out"
    --error="${LOG_DIR}/slurm-%j.out"
    --export=ALL,LOG_DIR="${LOG_DIR}",PREFILL_SCRIPT="${PREFILL_SCRIPT}",DECODE_SCRIPT="${DECODE_SCRIPT}",PORT="${PORT}"
  )

  if [[ -n "${ACCOUNT:-}" ]]; then
    sbatch_args+=(--account="${ACCOUNT}")
  fi
  if [[ -n "${NODELIST:-}" ]]; then
    sbatch_args+=(--nodelist="${NODELIST}")
  fi
  if [[ -n "${EXCLUDE:-}" ]]; then
    sbatch_args+=(--exclude="${EXCLUDE}")
  fi

  echo "Submitting ${JOB_NAME}; logs: ${LOG_DIR}"
  exec sbatch "${sbatch_args[@]}" "$0"
fi

mkdir -p "${LOG_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
if (( ${#NODES[@]} < 3 )); then
  echo "Need at least 3 allocated nodes for 1P/1D DEP8, got ${#NODES[@]}" >&2
  exit 1
fi

resolve_host() {
  local host=$1
  local addr
  addr=$(getent hosts "${host}" | awk '{print $1; exit}' || true)
  if [[ -n "${addr}" && "${addr}" != 127.* ]]; then
    echo "${addr}"
  else
    echo "${host}"
  fi
}

PREFILL_NODE="${PREFILL_NODE:-${NODES[0]}}"
DECODE_NODE_0="${DECODE_NODE_0:-${NODES[1]}}"
DECODE_NODE_1="${DECODE_NODE_1:-${NODES[2]}}"
DECODE_MASTER_ADDR="${DECODE_MASTER_ADDR:-$(resolve_host "${DECODE_NODE_0}")}"

cat > "${LOG_DIR}/nodes.env" <<EOF
PREFILL_NODE=${PREFILL_NODE}
DECODE_NODE_0=${DECODE_NODE_0}
DECODE_NODE_1=${DECODE_NODE_1}
DECODE_MASTER_ADDR=${DECODE_MASTER_ADDR}
PREFILL_URL=http://${PREFILL_NODE}:${PORT}
DECODE_URL_0=http://${DECODE_NODE_0}:${PORT}
DECODE_URL_1=http://${DECODE_NODE_1}:${PORT}
EOF

echo "=== 1P/1D Slurm Bring-up ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${SLURM_NODELIST}"
echo "Prefill node: ${PREFILL_NODE}"
echo "Decode node0: ${DECODE_NODE_0}"
echo "Decode node1: ${DECODE_NODE_1}"
echo "Decode addr:  ${DECODE_MASTER_ADDR}"
echo "Log dir:      ${LOG_DIR}"
echo "Started:      $(date -Iseconds)"
echo "============================"

cleanup() {
  echo "Stopping Slurm steps..."
  if [[ -n "${PREFILL_PID:-}" ]]; then
    kill "${PREFILL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${DECODE_PID:-}" ]]; then
    kill "${DECODE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${DECODE_PID_1:-}" ]]; then
    kill "${DECODE_PID_1}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting prefill on ${PREFILL_NODE}"
srun \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="${PREFILL_NODE}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --exclusive \
  /usr/bin/env PORT="${PORT}" "${PREFILL_SCRIPT}" \
  > "${LOG_DIR}/prefill-${PREFILL_NODE}.log" 2>&1 &
PREFILL_PID=$!

echo "Starting decode rank 0 on ${DECODE_NODE_0}"
srun \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="${DECODE_NODE_0}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --exclusive \
  /usr/bin/env PORT="${PORT}" D_NNODES=2 NODE_RANK=0 DECODE_MASTER_ADDR="${DECODE_MASTER_ADDR}" "${DECODE_SCRIPT}" \
  > "${LOG_DIR}/decode-${DECODE_NODE_0}.log" 2>&1 &
DECODE_PID=$!

echo "Starting decode rank 1 on ${DECODE_NODE_1}"
srun \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="${DECODE_NODE_1}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --exclusive \
  /usr/bin/env PORT="${PORT}" D_NNODES=2 NODE_RANK=1 DECODE_MASTER_ADDR="${DECODE_MASTER_ADDR}" "${DECODE_SCRIPT}" \
  > "${LOG_DIR}/decode-${DECODE_NODE_1}.log" 2>&1 &
DECODE_PID_1=$!

echo "Logs:"
echo "  prefill: ${LOG_DIR}/prefill-${PREFILL_NODE}.log"
echo "  decode0: ${LOG_DIR}/decode-${DECODE_NODE_0}.log"
echo "  decode1: ${LOG_DIR}/decode-${DECODE_NODE_1}.log"
echo "Endpoints written to ${LOG_DIR}/nodes.env"

set +e
wait -n "${PREFILL_PID}" "${DECODE_PID}" "${DECODE_PID_1}"
exit_code=$?
set -e

echo "A Slurm step exited with code ${exit_code}; stopping the other step."
exit "${exit_code}"
