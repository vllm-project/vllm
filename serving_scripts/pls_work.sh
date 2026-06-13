#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=r32_sp128_sd128_pp2_tp2_qwen3_30b_raydebug_v4
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=00:45:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority

set -euo pipefail

SCRIPT_VERSION="ray-debug-v4-qwen3-30b-tp2-pp2-export-launcher-env"

# This script is intentionally a minimal Ray/vLLM connectivity test:
#   2 nodes x 2 H100s/node = 4 total GPUs
#   TP=2 within each node
#   PP=2 across nodes
#
# Main differences from v2:
#   - No `ray stop --force`. That was touching stale /tmp/ray sessions from other jobs/users.
#   - Dynamic Ray port by default.
#   - Unique Ray temp dir per job and node.
#   - Ray dashboard disabled.
#   - 50GB object store by default for Ray-startup debugging.
#   - Python Ray probes are bounded with timeout.
#   - Requires 2 alive Ray nodes and 4 Ray GPUs before starting vLLM.
#   - Nsight is off by default. Re-enable only after Ray/vLLM startup is stable.
#   - v4 exports launcher variables so srun children can see VENV_DIR/RAY_BIN/TRACE_RUN_DIR.

DEBUG_SLURM_SCRIPT="${DEBUG_SLURM_SCRIPT:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

SP="${SP:-128}"
SD="${SD:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
REQUEST_RATE="${REQUEST_RATE:-1}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8000}"

GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
TP="${TP:-2}"
PP="${PP:-2}"
EP="${EP:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-64}}"

RAY_HEAD_READY_TIMEOUT_S="${RAY_HEAD_READY_TIMEOUT_S:-240}"
RAY_CLUSTER_READY_TIMEOUT_S="${RAY_CLUSTER_READY_TIMEOUT_S:-300}"
RAY_READY_POLL_S="${RAY_READY_POLL_S:-5}"
RAY_PROBE_TIMEOUT="${RAY_PROBE_TIMEOUT:-20s}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-900}"
SERVER_SHUTDOWN_TIMEOUT_S="${SERVER_SHUTDOWN_TIMEOUT_S:-180}"

RAY_PLASMA_DIRECTORY="${RAY_PLASMA_DIRECTORY:-/dev/shm}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-50000000000}"

RAY_CGRAPH_GET_TIMEOUT="${RAY_CGRAPH_GET_TIMEOUT:-1400}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-${RAY_CGRAPH_GET_TIMEOUT}}"
export RAY_CGRAPH_submit_timeout="${RAY_CGRAPH_submit_timeout:-1800}"

# Dynamic by default so repeated jobs do not collide with stale fixed-port Ray.
export RAY_PORT="${RAY_PORT:-$((6300 + (${SLURM_JOB_ID:-0} % 1000)))}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

export NSYS_ENABLE="${NSYS_ENABLE:-0}"
export NSYS_PROFILE_WORKERS="${NSYS_PROFILE_WORKERS:-0}"

run_with_timeout() {
  local limit="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "${limit}" "$@"
  else
    "$@"
  fi
}

resolve_host_ip() {
  local nodename="$1"
  local ip=""
  ip=$(dig +short "${nodename}" 2>/dev/null | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}' || true)
  if [ -z "${ip}" ]; then
    ip=$(getent hosts "${nodename}" 2>/dev/null | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/ {print $1; exit}' || true)
  fi
  if [ -z "${ip}" ]; then
    ip=$(srun --nodelist="${nodename}" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G \
      bash -lc "hostname -I | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true)
  fi
  printf '%s' "${ip}"
}

ray_tmp_link_for_node() {
  local node="$1"
  printf '/tmp/vray-%s-%s' "${SLURM_JOB_ID}" "${node}"
}

dump_ray_step_tails() {
  echo "=== Ray step tails ===" >&2
  for f in \
    "${TRACE_RUN_DIR}/slurm_ray_head_${HEAD_NODE}.out" \
    "${TRACE_RUN_DIR}/slurm_ray_head_${HEAD_NODE}.err" \
    "${TRACE_RUN_DIR}"/slurm_ray_worker_*.out \
    "${TRACE_RUN_DIR}"/slurm_ray_worker_*.err; do
    [ -f "${f}" ] || continue
    echo "--- ${f} ---" >&2
    tail -120 "${f}" >&2 || true
  done

  echo "=== Ray internal log hints ===" >&2
  for node in ${HEAD_NODE} ${WORKER_NODES:-}; do
    local link
    link="$(ray_tmp_link_for_node "${node}")"
    for f in \
      "${link}"/session_latest/logs/gcs_server.err \
      "${link}"/session_latest/logs/gcs_server.out \
      "${link}"/session_latest/logs/raylet.err \
      "${link}"/session_latest/logs/raylet.out \
      "${link}"/session_latest/logs/dashboard.err \
      "${link}"/session_latest/logs/dashboard_agent.err \
      "${link}"/session_latest/logs/ray_process_exit.log; do
      [ -f "${f}" ] || continue
      echo "--- ${f} ---" >&2
      tail -80 "${f}" >&2 || true
    done
  done
}

cleanup() {
  set +e
  if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    sleep 10
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi

  for pid in ${WORKER_RAY_PIDS:-} ${HEAD_RAY_PID:-}; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Main setup
# -----------------------------------------------------------------------------

export HEAD_NODE
HEAD_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"

export WORKER_NODES
WORKER_NODES="$(scontrol show hostnames "${SLURM_NODELIST}" | tail -n+2)"

export HEAD_NODE_IP
HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
if [ -z "${HEAD_NODE_IP}" ]; then
  echo "ERROR: could not resolve head node IP for ${HEAD_NODE}" >&2
  exit 1
fi

NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}"
TOTAL_GPUS="$((GPUS_PER_NODE * NUM_NODES))"

export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
export VLLM_HOST_IP="${HEAD_NODE_IP}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(pwd)"
fi
VENV_DIR="${REPO_ROOT}/.venv"

TRACE_BASE="${TRACE_BASE:-${REPO_ROOT}/results}"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"
mkdir -p "${TRACE_RUN_DIR}/ray_logs" "${TRACE_RUN_DIR}/nccl_logs"

source "${VENV_DIR}/bin/activate"
RAY_BIN="${VENV_DIR}/bin/ray"

if [ ! -x "${RAY_BIN}" ]; then
  echo "ERROR: Ray binary not found at ${RAY_BIN}" >&2
  exit 1
fi

if [ "${INSTALL_DEPS}" = "1" ]; then
  echo "INSTALL_DEPS=1; refreshing editable vLLM install."
  python -m pip install -U pip
  python -m pip install -r "${REPO_ROOT}/requirements/cuda.txt"
  python -m pip install -r "${REPO_ROOT}/requirements/build/cuda.txt"
  python -m pip install "${RAY_REQUIREMENT:-ray[cgraph]>=2.48.0}"
  (
    cd "${REPO_ROOT}"
    export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM:-0.19.2.dev0}"
    python -m pip install -e .
  )
fi

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export RAY_PLASMA_DIRECTORY RAY_OBJECT_STORE_MEMORY GPUS_PER_NODE CPUS_PER_TASK
export NCCL_IB_DISABLE NCCL_NET NCCL_IB_HCA NCCL_SOCKET_FAMILY NCCL_DEBUG NCCL_DEBUG_SUBSYS TORCH_DISTRIBUTED_DEBUG RAY_DEDUP_LOGS
export RAY_CGRAPH_get_timeout RAY_CGRAPH_submit_timeout

SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT_multi_node.sh"
LAUNCHER="${TRACE_RUN_DIR}/ray_node_start.sh"

# These must be exported because the Ray launcher is executed by srun as a
# separate process. Bash shell variables are not inherited unless exported.
export REPO_ROOT VENV_DIR RAY_BIN TRACE_RUN_DIR SERVE_SCRIPT LAUNCHER
export MODEL_ID PORT GPUS_PER_NODE TP PP EP MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION
export RAY_PORT RAY_ADDRESS HEAD_NODE HEAD_NODE_IP WORKER_NODES NUM_NODES TOTAL_GPUS
export SP SD NUM_PROMPTS REQUEST_RATE NSYS_ENABLE NSYS_PROFILE_WORKERS

cat > "${LAUNCHER}" <<'NODE_LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

ROLE="$1"
NODE_NAME="$2"
NODE_IP="$3"
HEAD_IP="$4"
RAY_PORT="$5"

source "${VENV_DIR}/bin/activate"

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

REAL_ROOT="/tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${NODE_NAME}"
SHORT_LINK="/tmp/vray-${SLURM_JOB_ID}-${NODE_NAME}"

rm -rf "${REAL_ROOT}" "${SHORT_LINK}" 2>/dev/null || true
mkdir -p "${REAL_ROOT}/spill" "${REAL_ROOT}/py_tmp"
ln -sfn "${REAL_ROOT}" "${SHORT_LINK}"

export RAY_TMPDIR="${SHORT_LINK}"
export RAY_SPILL_DIR="${SHORT_LINK}/spill"
export TMPDIR="${SHORT_LINK}/py_tmp"
export VLLM_HOST_IP="${NODE_IP}"
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

IFACE="$(ip -o -4 addr show 2>/dev/null | awk -v target="${NODE_IP}" '{split($4,a,"/"); if (a[1] == target) {print $2; exit}}')"
if [ -z "${IFACE}" ]; then
  echo "ERROR: could not find local interface for NODE_IP=${NODE_IP} on $(hostname)" >&2
  ip -o -4 addr show >&2 || true
  exit 1
fi

export GLOO_SOCKET_IFNAME="${IFACE}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${IFACE}}"
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

mkdir -p "${TRACE_RUN_DIR}/ray_logs/${NODE_NAME}" "${TRACE_RUN_DIR}/nccl_logs"

{
  echo "=== ray node launcher ==="
  echo "date=$(date -Is 2>/dev/null || date)"
  echo "host=$(hostname -f 2>/dev/null || hostname)"
  echo "ROLE=${ROLE} NODE_NAME=${NODE_NAME} NODE_IP=${NODE_IP} HEAD_IP=${HEAD_IP} RAY_PORT=${RAY_PORT}"
  echo "python=$(command -v python)"
  python - <<'PY'
import ray, sys
print("python_version:", sys.version.replace("\n", " "))
print("ray_version:", ray.__version__)
print("ray_path:", ray.__file__)
PY
  echo "REAL_ROOT=${REAL_ROOT}"
  echo "SHORT_LINK=${SHORT_LINK} -> $(readlink -f "${SHORT_LINK}" 2>/dev/null || echo MISSING)"
  echo "RAY_TMPDIR=${RAY_TMPDIR}"
  echo "TMPDIR=${TMPDIR}"
  echo "RAY_PLASMA_DIRECTORY=${RAY_PLASMA_DIRECTORY}"
  echo "RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY}"
  echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
  echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  echo "NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>} NCCL_NET=${NCCL_NET:-<unset>} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-<unset>}"
  echo "Listening sockets before ray start:"
  ss -ltnp 2>/dev/null | grep -E ":${RAY_PORT}\b|ray|gcs" || true
  df -h /tmp "${REAL_ROOT}" "${RAY_PLASMA_DIRECTORY}" 2>/dev/null || true
} | tee "${TRACE_RUN_DIR}/ray_logs/${NODE_NAME}/ray_start_preamble_${ROLE}.log"

if [ "${ROLE}" = "head" ]; then
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${RAY_PORT}$"; then
    echo "ERROR: port ${RAY_PORT} is already listening on head node before ray start." >&2
    ss -ltnp 2>/dev/null | grep -E ":${RAY_PORT}\b" >&2 || true
    exit 1
  fi

  exec "${RAY_BIN}" start --block \
    --head \
    --node-ip-address="${NODE_IP}" \
    --port="${RAY_PORT}" \
    --include-dashboard=false \
    --disable-usage-stats \
    --temp-dir="${RAY_TMPDIR}" \
    --plasma-directory="${RAY_PLASMA_DIRECTORY}" \
    --object-store-memory="${RAY_OBJECT_STORE_MEMORY}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --num-cpus="${CPUS_PER_TASK}"
else
  exec "${RAY_BIN}" start --block \
    --address="${HEAD_IP}:${RAY_PORT}" \
    --node-ip-address="${NODE_IP}" \
    --include-dashboard=false \
    --disable-usage-stats \
    --temp-dir="${RAY_TMPDIR}" \
    --plasma-directory="${RAY_PLASMA_DIRECTORY}" \
    --object-store-memory="${RAY_OBJECT_STORE_MEMORY}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --num-cpus="${CPUS_PER_TASK}"
fi
NODE_LAUNCHER

chmod +x "${LAUNCHER}"

cat <<EOF_SUMMARY
=== ${SCRIPT_VERSION} ===
Date: $(date -Is 2>/dev/null || date)
SLURM_JOB_ID=${SLURM_JOB_ID}
SLURM_NODELIST=${SLURM_NODELIST}
HEAD_NODE=${HEAD_NODE}
WORKER_NODES=${WORKER_NODES}
HEAD_NODE_IP=${HEAD_NODE_IP}
RAY_ADDRESS=${RAY_ADDRESS}
REPO_ROOT=${REPO_ROOT}
VENV_DIR=${VENV_DIR}
TRACE_RUN_DIR=${TRACE_RUN_DIR}
MODEL_ID=${MODEL_ID}
TP=${TP} PP=${PP} GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES} TOTAL_GPUS=${TOTAL_GPUS} CPUS_PER_TASK=${CPUS_PER_TASK}
SP=${SP} SD=${SD} NUM_PROMPTS=${NUM_PROMPTS} REQUEST_RATE=${REQUEST_RATE}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY}
NCCL_IB_HCA=${NCCL_IB_HCA} NCCL_NET=${NCCL_NET} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}
NSYS_ENABLE=${NSYS_ENABLE} NSYS_PROFILE_WORKERS=${NSYS_PROFILE_WORKERS}
EOF_SUMMARY

python - <<'PY'
import ray, vllm, torch, sys
print("python:", sys.executable)
print("ray:", ray.__version__, ray.__file__)
print("vllm:", getattr(vllm, "__version__", "unknown"), vllm.__file__)
print("torch:", torch.__version__)
PY

wait_for_ray_cluster() {
  local expected_nodes="$1"
  local expected_gpus="$2"
  local timeout_s="$3"
  local poll_s="$4"
  local label="$5"
  local elapsed=0

  echo "Waiting for Ray ${label}: expected_nodes=${expected_nodes}, expected_gpus=${expected_gpus}, address=${RAY_ADDRESS}"

  while [ "${elapsed}" -lt "${timeout_s}" ]; do
    if [ -n "${HEAD_RAY_PID:-}" ] && ! kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
      echo "ERROR: Ray head srun step died while waiting for ${label}." >&2
      dump_ray_step_tails || true
      return 1
    fi

    local dead_worker=0
    for pid in ${WORKER_RAY_PIDS:-}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        dead_worker=1
      fi
    done
    if [ "${dead_worker}" = "1" ]; then
      echo "ERROR: at least one Ray worker srun step died while waiting for ${label}." >&2
      dump_ray_step_tails || true
      return 1
    fi

    if EXPECTED_RAY_NODES="${expected_nodes}" EXPECTED_RAY_GPUS="${expected_gpus}" \
      run_with_timeout "${RAY_PROBE_TIMEOUT}" python - <<'PY'
import os
import sys
import ray

addr = os.environ["RAY_ADDRESS"]
expected_nodes = int(os.environ["EXPECTED_RAY_NODES"])
expected_gpus = float(os.environ["EXPECTED_RAY_GPUS"])

try:
    ray.init(address=addr, logging_level="ERROR", ignore_reinit_error=True)
    nodes = ray.nodes()
    alive = [n for n in nodes if n.get("Alive")]
    total_gpus = sum(float(n.get("Resources", {}).get("GPU", 0.0)) for n in alive)
    print(f"Ray alive nodes: {len(alive)}/{expected_nodes}")
    print(f"Ray GPUs: {total_gpus}/{expected_gpus}")
    for n in alive:
        print(f"  {n.get('NodeManagerAddress')} resources={n.get('Resources')}")
    ray.shutdown()
    if len(alive) >= expected_nodes and total_gpus >= expected_gpus:
        sys.exit(0)
    sys.exit(1)
except Exception as e:
    print(f"Ray probe failed: {type(e).__name__}: {e}")
    try:
        ray.shutdown()
    except Exception:
        pass
    sys.exit(1)
PY
    then
      echo "Ray ${label} is ready."
      return 0
    fi

    elapsed=$((elapsed + poll_s))
    echo "Ray ${label} not ready yet; retrying in ${poll_s}s (${elapsed}/${timeout_s}s)..."
    sleep "${poll_s}"
  done

  echo "ERROR: Ray ${label} did not become ready within ${timeout_s}s." >&2
  dump_ray_step_tails || true
  return 1
}

SERVER_PID=""
HEAD_RAY_PID=""
WORKER_RAY_PIDS=""

echo "=== Starting Ray head on ${HEAD_NODE} ==="
srun --nodelist="${HEAD_NODE}" --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" --cpus-per-task="${CPUS_PER_TASK}" \
  --output="${TRACE_RUN_DIR}/slurm_ray_head_${HEAD_NODE}.out" \
  --error="${TRACE_RUN_DIR}/slurm_ray_head_${HEAD_NODE}.err" \
  "${LAUNCHER}" head "${HEAD_NODE}" "${HEAD_NODE_IP}" "${HEAD_NODE_IP}" "${RAY_PORT}" &
HEAD_RAY_PID=$!
echo "HEAD_RAY_PID=${HEAD_RAY_PID}"

wait_for_ray_cluster 1 "${GPUS_PER_NODE}" "${RAY_HEAD_READY_TIMEOUT_S}" "${RAY_READY_POLL_S}" "head"

echo "=== Starting Ray workers ==="
for WORKER in ${WORKER_NODES}; do
  WORKER_IP="$(resolve_host_ip "${WORKER}")"
  if [ -z "${WORKER_IP}" ]; then
    echo "ERROR: could not resolve worker IP for ${WORKER}" >&2
    exit 1
  fi
  echo "Starting Ray worker ${WORKER} (${WORKER_IP})"
  srun --nodelist="${WORKER}" --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gpus-per-task="${GPUS_PER_NODE}" --cpus-per-task="${CPUS_PER_TASK}" \
    --output="${TRACE_RUN_DIR}/slurm_ray_worker_${WORKER}.out" \
    --error="${TRACE_RUN_DIR}/slurm_ray_worker_${WORKER}.err" \
    "${LAUNCHER}" worker "${WORKER}" "${WORKER_IP}" "${HEAD_NODE_IP}" "${RAY_PORT}" &
  WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
done

wait_for_ray_cluster "${NUM_NODES}" "${TOTAL_GPUS}" "${RAY_CLUSTER_READY_TIMEOUT_S}" "${RAY_READY_POLL_S}" "cluster"

echo "=== Starting vLLM api_server TP=${TP} PP=${PP} ==="
export RAY_TMPDIR="$(ray_tmp_link_for_node "${HEAD_NODE}")"
export RAY_SPILL_DIR="${RAY_TMPDIR}/spill"
export TMPDIR="${RAY_TMPDIR}/py_tmp"
mkdir -p "${TMPDIR}" "${RAY_SPILL_DIR}" 2>/dev/null || true

VLLM_TRACE_FLAGS=()
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  VLLM_TRACE_FLAGS+=(--ray-workers-use-nsight --enable-layerwise-nvtx-tracing --enable-logging-iteration-details)
fi

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HEAD_NODE_IP}" \
  --port "${PORT}" \
  --distributed-executor-backend ray \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  "${VLLM_TRACE_FLAGS[@]}" \
  --disable-custom-all-reduce &
SERVER_PID=$!

echo "Started vLLM server pid=${SERVER_PID}; waiting for /health ..."
elapsed=0
until curl -fsS "http://${HEAD_NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: vLLM server exited before becoming healthy." >&2
    dump_ray_step_tails || true
    wait "${SERVER_PID}" || true
    exit 1
  fi

  if [ "${elapsed}" -ge "${SERVER_READY_TIMEOUT_S}" ]; then
    echo "ERROR: vLLM server did not become healthy within ${SERVER_READY_TIMEOUT_S}s." >&2
    dump_ray_step_tails || true
    exit 1
  fi

  sleep 5
  elapsed=$((elapsed + 5))
  if [ "$((elapsed % 60))" -eq 0 ]; then
    echo "Still waiting for health (${elapsed}/${SERVER_READY_TIMEOUT_S}s)"
  fi
done

echo "Server is healthy. Running benchmark."
HOST="${HEAD_NODE_IP}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" \
  SP="${SP}" SD="${SD}" NUM_PROMPTS="${NUM_PROMPTS}" REQUEST_RATE="${REQUEST_RATE}" \
  HEAD_NODE_IP="${HEAD_NODE_IP}" GPUS_PER_NODE="${GPUS_PER_NODE}" CPUS_PER_TASK="${CPUS_PER_TASK}" \
  RAY_PORT="${RAY_PORT}" bash "${SERVE_SCRIPT}" "${SLURM_JOB_ID}" "${HEAD_NODE}"

echo "Benchmark finished. Stopping vLLM server."
if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
  kill -INT "${SERVER_PID}" 2>/dev/null || true
  sleep 10
  kill -TERM "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
fi
SERVER_PID=""

echo "Ray internal logs after run:"
dump_ray_step_tails || true

echo "Done."
