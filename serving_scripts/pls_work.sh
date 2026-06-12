#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=r32_sp128_sd128_pp2_tp2_qwen3_30b_raydebug
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

SCRIPT_VERSION="ray-debug-qwen3-30b-tp2-pp2-dashboard-off-readiness-timeouts-v2-cpus-fix"

# This is a connectivity-first script.
# It intentionally defaults Nsight worker profiling off so we can first prove:
#   Ray head starts -> Ray worker joins -> vLLM starts with TP=2, PP=2.
# Once this works, submit with NSYS_ENABLE=1 to re-enable worker-side Nsight.

DEBUG_SLURM_SCRIPT="${DEBUG_SLURM_SCRIPT:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

# Workload.
SP="${SP:-128}"
SD="${SD:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
REQUEST_RATE="${REQUEST_RATE:-1}"

# Ray/vLLM layout: 2 nodes x 2 GPUs/node = 4 GPUs total, TP2 within node, PP2 across nodes.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
# Slurm sets SLURM_CPUS_PER_TASK, not necessarily CPUS_PER_TASK.
# Define our own variable before any srun/ray start command so set -u cannot kill the job.
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-64}}"
TP="${TP:-2}"
PP="${PP:-2}"
EP="${EP:-1}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Startup/debugging knobs.
RAY_HEAD_READY_TIMEOUT_S="${RAY_HEAD_READY_TIMEOUT_S:-240}"
RAY_CLUSTER_READY_TIMEOUT_S="${RAY_CLUSTER_READY_TIMEOUT_S:-300}"
RAY_READY_POLL_S="${RAY_READY_POLL_S:-5}"
RAY_PROBE_TIMEOUT="${RAY_PROBE_TIMEOUT:-20s}"
SERVER_HEALTH_TIMEOUT_S="${SERVER_HEALTH_TIMEOUT_S:-1200}"
SERVER_SHUTDOWN_TIMEOUT_S="${SERVER_SHUTDOWN_TIMEOUT_S:-420}"
SRUN_COPY_TIMEOUT="${SRUN_COPY_TIMEOUT:-240s}"

# Keep object store modest while debugging Ray startup. 50 GB is plenty for this test.
RAY_PLASMA_DIRECTORY="${RAY_PLASMA_DIRECTORY:-/dev/shm}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-50000000000}"
RAY_TMP_LINK_PARENT="${RAY_TMP_LINK_PARENT:-/tmp}"
RAY_TMP_PREFIX="${RAY_TMP_PREFIX:-vray-${SLURM_JOB_ID}}"
CLEAN_RAY_TMP_ON_EXIT="${CLEAN_RAY_TMP_ON_EXIT:-1}"

# Avoid compiled-DAG timeout while cold-starting/profiling.
RAY_CGRAPH_GET_TIMEOUT="${RAY_CGRAPH_GET_TIMEOUT:-1400}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-${RAY_CGRAPH_GET_TIMEOUT}}"
export RAY_CGRAPH_submit_timeout="${RAY_CGRAPH_submit_timeout:-1800}"

# Default profiling off for connectivity debugging. Set NSYS_ENABLE=1 after this script works.
export NSYS_ENABLE="${NSYS_ENABLE:-0}"
export NSYS_PROFILE_WORKERS="${NSYS_PROFILE_WORKERS:-${NSYS_ENABLE}}"
export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
export NSYS_DELAY="${NSYS_DELAY:-0}"
export VLLM_ITERATION_NVTX="${VLLM_ITERATION_NVTX:-1}"
export VLLM_KV_CACHE_METRICS="${VLLM_KV_CACHE_METRICS:-1}"
export VLLM_KV_CACHE_METRICS_SAMPLE="${VLLM_KV_CACHE_METRICS_SAMPLE:-0.01}"

# NCCL/RDMA. Use broad mlx5 by default instead of over-constraining to mlx5_0.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

slurm_debug() {
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[slurm-debug] $*" >&2
  fi
}

run_with_timeout() {
  local limit="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "${limit}" "$@"
  else
    "$@"
  fi
}

wait_for_pid_or_kill() {
  local pid="$1"
  local label="$2"
  local timeout_s="${3:-420}"
  local elapsed=0

  if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "${label}: pid ${pid:-<empty>} is not running"
    return 0
  fi

  while kill -0 "${pid}" 2>/dev/null && [ "${elapsed}" -lt "${timeout_s}" ]; do
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if kill -0 "${pid}" 2>/dev/null; then
    echo "${label}: still alive after ${timeout_s}s; sending TERM"
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 10
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    echo "${label}: still alive after TERM; sending KILL"
    kill -KILL "${pid}" 2>/dev/null || true
  fi

  wait "${pid}" 2>/dev/null || true
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
      bash -lc "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true)
  fi

  printf '%s' "${ip}"
}

ray_tmp_link_for_node() {
  local node="$1"
  printf '%s/%s-%s' "${RAY_TMP_LINK_PARENT}" "${RAY_TMP_PREFIX}" "${node}"
}

ray_tmp_real_dir_for_node() {
  local node="$1"
  printf '/tmp/slurm-%s/tmp/ray-%s-%s' "${SLURM_JOB_ID}" "${SLURM_JOB_ID}" "${node}"
}

write_ray_node_launcher() {
  local launcher="$1"

  cat > "${launcher}" <<'NODE_LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

ROLE="$1"          # head or worker
NODE_NAME="$2"
NODE_IP="$3"
HEAD_IP="$4"
RAY_PORT="$5"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0
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
# Let NCCL use the same socket interface for bootstrap; IB HCA still controls RDMA.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${IFACE}}"

mkdir -p "${TRACE_RUN_DIR}/ray_logs/${NODE_NAME}" "${TRACE_RUN_DIR}/nccl_logs"
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

# Clean stale Ray processes from this user on this allocated node.
ray stop --force || true
sleep 2

{
  echo "=== ray node launcher ==="
  echo "date=$(date -Is 2>/dev/null || date)"
  echo "host=$(hostname -f 2>/dev/null || hostname)"
  echo "ROLE=${ROLE} NODE_NAME=${NODE_NAME} NODE_IP=${NODE_IP} HEAD_IP=${HEAD_IP} RAY_PORT=${RAY_PORT}"
  echo "python=$(command -v python)"
  python - <<'PY'
import ray, sys
print('python_version:', sys.version.replace('\n', ' '))
print('ray_version:', ray.__version__)
print('ray_path:', ray.__file__)
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
  exec "${RAY_BIN}" start --block \
    --head \
    --node-ip-address="${NODE_IP}" \
    --port="${RAY_PORT}" \
    --include-dashboard=false \
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
    --temp-dir="${RAY_TMPDIR}" \
    --plasma-directory="${RAY_PLASMA_DIRECTORY}" \
    --object-store-memory="${RAY_OBJECT_STORE_MEMORY}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --num-cpus="${CPUS_PER_TASK}"
fi
NODE_LAUNCHER

  chmod +x "${launcher}"
}

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

  echo "ERROR: Ray ${label} was not ready within ${timeout_s}s." >&2
  dump_ray_step_tails || true
  return 1
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
}

collect_ray_logs_from_node() {
  local node="$1"
  local out="${TRACE_RUN_DIR}/ray_logs/${node}"
  mkdir -p "${out}"
  run_with_timeout "${SRUN_COPY_TIMEOUT}" srun --nodelist="${node}" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G \
    bash -lc "set +e; mkdir -p '${out}'; for s in /tmp/vray-${SLURM_JOB_ID}-${node}/session_* /tmp/vray-${SLURM_JOB_ID}-${node}/session_latest /tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${node}/session_*; do [ -d \"\$s/logs\" ] || continue; b=\$(basename \"\$s\"); tar -C \"\$s\" -czf '${out}'/\"${node}_\${b}_logs.tgz\" logs --exclude='logs/nsight/*.qdstrm' --exclude='logs/nsight/*.nsys-rep' --exclude='logs/events/*' 2>/dev/null || true; done; find '${out}' -type f -printf '%p %s bytes\n' 2>/dev/null | sort" || true
}

cleanup() {
  set +e
  if [ -n "${SERVER_STEP_PID:-}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill -INT "${SERVER_STEP_PID}" 2>/dev/null || true
    wait_for_pid_or_kill "${SERVER_STEP_PID}" "vLLM API server" "${SERVER_SHUTDOWN_TIMEOUT_S}"
  fi
  if [ -n "${HEAD_RAY_PID:-}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
    kill -TERM "${HEAD_RAY_PID}" 2>/dev/null || true
    wait "${HEAD_RAY_PID}" 2>/dev/null || true
  fi
  for pid in ${WORKER_RAY_PIDS:-}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [ "${CLEAN_RAY_TMP_ON_EXIT}" = "1" ]; then
    for node in ${HEAD_NODE:-} ${WORKER_NODES:-}; do
      [ -n "${node}" ] || continue
      srun --nodelist="${node}" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G \
        bash -lc "rm -rf /tmp/vray-${SLURM_JOB_ID}-${node} /tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${node} 2>/dev/null || true" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

mapfile -t ALLOC_NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
HEAD_NODE="${ALLOC_NODES[0]}"
WORKER_NODES="${ALLOC_NODES[*]:1}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${#ALLOC_NODES[@]}}"
TOTAL_GPUS="$((GPUS_PER_NODE * NUM_NODES))"

HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
if [ -z "${HEAD_NODE_IP}" ]; then
  echo "ERROR: could not resolve head node IP for ${HEAD_NODE}" >&2
  exit 1
fi
export HEAD_NODE_IP
export HOST="${HEAD_NODE_IP}"
export RAY_PORT="${RAY_PORT:-$((6300 + (${SLURM_JOB_ID:-0} % 1000)))}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

TRACE_BASE="${TRACE_BASE:-/data/engs-glass/catz0932/inference-traces/vllm/results}"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"
mkdir -p "${TRACE_RUN_DIR}/ray_logs" "${TRACE_RUN_DIR}/nccl_logs" "${TRACE_RUN_DIR}/nsight"
export TRACE_RUN_DIR
export NSYS_DIR="${TRACE_RUN_DIR}/nsight"
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(pwd)"
fi
VENV_DIR="${REPO_ROOT}/.venv"
export REPO_ROOT VENV_DIR

source "${VENV_DIR}/bin/activate"
RAY_BIN="${VENV_DIR}/bin/ray"
export RAY_BIN

if [ ! -x "${RAY_BIN}" ]; then
  echo "ERROR: ray binary not found: ${RAY_BIN}" >&2
  exit 1
fi

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export RAY_PLASMA_DIRECTORY RAY_OBJECT_STORE_MEMORY GPUS_PER_NODE CPUS_PER_TASK
export NCCL_IB_DISABLE NCCL_NET NCCL_IB_HCA NCCL_SOCKET_FAMILY NCCL_DEBUG NCCL_DEBUG_SUBSYS TORCH_DISTRIBUTED_DEBUG RAY_DEDUP_LOGS

SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT_multi_node.sh"
LAUNCHER="${TRACE_RUN_DIR}/ray_node_start.sh"
write_ray_node_launcher "${LAUNCHER}"

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
print('python:', sys.executable)
print('ray:', ray.__version__, ray.__file__)
print('vllm:', getattr(vllm, '__version__', 'unknown'), vllm.__file__)
print('torch:', torch.__version__)
PY

# Start Ray head.
SERVER_STEP_PID=""
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

# Start workers.
if [ -n "${WORKER_NODES}" ]; then
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
fi

wait_for_ray_cluster "${NUM_NODES}" "${TOTAL_GPUS}" "${RAY_CLUSTER_READY_TIMEOUT_S}" "${RAY_READY_POLL_S}" "cluster"

# Start vLLM server.
echo "=== Starting vLLM api_server TP=${TP} PP=${PP} ==="
export RAY_TMPDIR="$(ray_tmp_link_for_node "${HEAD_NODE}")"
export RAY_SPILL_DIR="${RAY_TMPDIR}/spill"
export TMPDIR="${RAY_TMPDIR}/py_tmp"
mkdir -p "${TMPDIR}" "${RAY_SPILL_DIR}" 2>/dev/null || true

declare -a VLLM_TRACE_FLAGS=()
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  VLLM_TRACE_FLAGS+=(--ray-workers-use-nsight --enable-layerwise-nvtx-tracing --enable-logging-iteration-details)
fi

declare -a VLLM_KV_CACHE_FLAGS=()
if [ "${VLLM_KV_CACHE_METRICS}" = "1" ]; then
  VLLM_KV_CACHE_FLAGS+=(--kv-cache-metrics --kv-cache-metrics-sample "${VLLM_KV_CACHE_METRICS_SAMPLE}")
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
  "${VLLM_KV_CACHE_FLAGS[@]}" \
  --disable-custom-all-reduce &
SERVER_STEP_PID=$!

echo "Started vLLM server pid=${SERVER_STEP_PID}; waiting for health."
health_elapsed=0
while ! curl -fsS "http://${HEAD_NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "ERROR: vLLM server exited before /health." >&2
    wait "${SERVER_STEP_PID}" || true
    dump_ray_step_tails || true
    exit 1
  fi
  if [ "${health_elapsed}" -ge "${SERVER_HEALTH_TIMEOUT_S}" ]; then
    echo "ERROR: vLLM server did not become healthy within ${SERVER_HEALTH_TIMEOUT_S}s." >&2
    dump_ray_step_tails || true
    exit 1
  fi
  if [ $((health_elapsed % 60)) -eq 0 ]; then
    echo "Still waiting for http://${HEAD_NODE_IP}:${PORT}/health (${health_elapsed}/${SERVER_HEALTH_TIMEOUT_S}s)..."
  fi
  sleep 5
  health_elapsed=$((health_elapsed + 5))
done

echo "Server is healthy. Running benchmark script: ${SERVE_SCRIPT}"
HOST="${HEAD_NODE_IP}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" \
  SP="${SP}" SD="${SD}" NUM_PROMPTS="${NUM_PROMPTS}" REQUEST_RATE="${REQUEST_RATE}" \
  HEAD_NODE_IP="${HEAD_NODE_IP}" GPUS_PER_NODE="${GPUS_PER_NODE}" CPUS_PER_TASK="${CPUS_PER_TASK}" \
  RAY_PORT="${RAY_PORT}" bash "${SERVE_SCRIPT}" "${SLURM_JOB_ID}" "${HEAD_NODE}"

echo "Benchmark finished. Stopping server and collecting logs."
if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
  kill -INT "${SERVER_STEP_PID}" 2>/dev/null || true
  wait_for_pid_or_kill "${SERVER_STEP_PID}" "vLLM API server" "${SERVER_SHUTDOWN_TIMEOUT_S}"
  SERVER_STEP_PID=""
fi

for node in ${HEAD_NODE} ${WORKER_NODES}; do
  collect_ray_logs_from_node "${node}"
done

echo "NCCL logs:"
find "${TRACE_RUN_DIR}/nccl_logs" -type f -printf '%p %s bytes\n' 2>/dev/null | sort || true

echo "Ray step logs:"
find "${TRACE_RUN_DIR}" -maxdepth 2 -type f \( -name 'slurm_ray_*.out' -o -name 'slurm_ray_*.err' \) -printf '%p %s bytes\n' 2>/dev/null | sort || true

echo "Done. TRACE_RUN_DIR=${TRACE_RUN_DIR}"
