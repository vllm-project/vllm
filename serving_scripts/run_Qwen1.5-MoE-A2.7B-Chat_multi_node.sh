#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=vllm-host-qwen15-moe
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority

set -euo pipefail

SCRIPT_VERSION="arc-ray-qwen15-moe-a2.7b-chat-nsys-v2"

DEBUG_SLURM_SCRIPT="${DEBUG_SLURM_SCRIPT:-0}"
slurm_debug() {
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[slurm-debug] $*" >&2
  fi
}

# SP = prompt / prefill token bucket
# SD = decode / output tokens per request
SP="${SP:-128}"
SD="${SD:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"

export HEAD_NODE
HEAD_NODE=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)

export WORKER_NODES
WORKER_NODES=$(scontrol show hostnames "${SLURM_NODELIST}" | tail -n+2)

echo "=== vLLM multi-node host job ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "Date: $(date -Is 2>/dev/null || date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODES=${WORKER_NODES}"
slurm_debug "SLURM_NTASKS=${SLURM_NTASKS:-} SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"
slurm_debug "Full nodelist: $(scontrol show hostnames "${SLURM_NODELIST}" 2>/dev/null | tr '\n' ' ')"

resolve_host_ip() {
  local nodename="$1"
  local ip=""
  local method=""

  pick_ipv4() {
    awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'
  }

  ip=$(dig +short "${nodename}" 2>/dev/null | pick_ipv4 || true)
  if [ -n "${ip}" ]; then
    method="dig_ipv4"
  fi

  if [ -z "${ip}" ]; then
    ip=$(getent hosts "${nodename}" 2>/dev/null | awk '{print $1}' | pick_ipv4 || true)
    [ -n "${ip}" ] && method="getent_ipv4"
  fi

  if [ -z "${ip}" ]; then
    ip=$(
      srun --nodelist="${nodename}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        bash -c "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true
    )
    [ -n "${ip}" ] && method="srun_hostname_I_ipv4"
  fi

  slurm_debug "resolve_host_ip(${nodename}) -> ${ip:-<empty>} [${method:-failed}]"
  printf '%s' "${ip}"
}

interface_for_ip() {
  local target_ip="$1"
  ip -o -4 addr show 2>/dev/null | awk -v target="${target_ip}" '
    {
      split($4, addr, "/")
      if (addr[1] == target) {
        print $2
        exit
      }
    }'
}

interface_has_ip() {
  local iface="$1"
  local target_ip="$2"
  ip -o -4 addr show dev "${iface}" 2>/dev/null | awk -v target="${target_ip}" '
    {
      split($4, addr, "/")
      if (addr[1] == target) {
        found = 1
        exit
      }
    }
    END { exit(found ? 0 : 1) }'
}

configure_socket_ifnames() {
  local target_ip="$1"
  local set_nccl="${2:-1}"
  local iface="${GLOO_SOCKET_IFNAME:-}"

  if [ -n "${iface}" ] && ! interface_has_ip "${iface}" "${target_ip}"; then
    echo "Ignoring GLOO_SOCKET_IFNAME=${iface}; it does not own ${target_ip} on $(hostname)." >&2
    iface=""
  fi

  if [ -z "${iface}" ]; then
    iface="$(interface_for_ip "${target_ip}")"
  fi

  if [ -z "${iface}" ]; then
    echo "Error: could not find a network interface for ${target_ip} on $(hostname)." >&2
    ip -o -4 addr show >&2 || true
    exit 1
  fi

  export GLOO_SOCKET_IFNAME="${iface}"

  if [ "${set_nccl}" = "1" ]; then
    local nccl_iface="${NCCL_SOCKET_IFNAME:-}"
    if [ -n "${nccl_iface}" ] && ! interface_has_ip "${nccl_iface}" "${target_ip}"; then
      echo "Ignoring NCCL_SOCKET_IFNAME=${nccl_iface}; it does not own ${target_ip} on $(hostname)." >&2
      nccl_iface=""
    fi
    export NCCL_SOCKET_IFNAME="${nccl_iface:-${iface}}"
  fi

  echo "Socket interface for ${target_ip} on $(hostname): GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
}

export HEAD_NODE_IP
HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"

if [ -z "${HEAD_NODE_IP}" ]; then
  echo "Error: could not resolve an IPv4 address for head node ${HEAD_NODE}." >&2
  exit 1
fi

echo "HEAD_NODE_IP=${HEAD_NODE_IP}"

export VLLM_HOST_IP="${HEAD_NODE_IP}"
echo "VLLM_HOST_IP=${VLLM_HOST_IP}"
configure_socket_ifnames "${HEAD_NODE_IP}" 0

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno,ens,enp,eth,ib}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"

# Useful when Nsight overhead makes Ray compiled graph waits too aggressive.
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"

export RAY_PORT="${RAY_PORT:-$((6300 + (${SLURM_JOB_ID:-0} % 1000)))}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
echo "RAY_PORT=${RAY_PORT} RAY_ADDRESS=${RAY_ADDRESS}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

# === Trace output directory ===
TRACE_BASE="${TRACE_BASE:-/data/engs-glass/catz0932/inference-traces/vllm/results}"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"

mkdir -p "${TRACE_RUN_DIR}/nsight"
mkdir -p "${TRACE_RUN_DIR}/nccl_logs"
mkdir -p "${TRACE_RUN_DIR}/ray_worker_nsight"
mkdir -p "${TRACE_RUN_DIR}/ray_step_logs"
mkdir -p "${TRACE_RUN_DIR}/server"
mkdir -p "${TRACE_RUN_DIR}/slurm_logs"

# Used to ignore stale /tmp/ray sessions and Nsight files left by prior jobs on the same nodes.
JOB_START_EPOCH="$(date +%s)"
JOB_SESSION_TS="$(date -d "@${JOB_START_EPOCH}" '+%Y-%m-%d_%H-%M-%S' 2>/dev/null || date -r "${JOB_START_EPOCH}" '+%Y-%m-%d_%H-%M-%S')"

# Mirror all subsequent job stdout/stderr into the trace directory as well.
# The #SBATCH output/error files still exist, but the self-contained copy lives here.
if [ "${MIRROR_STDOUT_TO_TRACE:-1}" = "1" ]; then
  exec > >(tee -a "${TRACE_RUN_DIR}/slurm_logs/job.out") 2> >(tee -a "${TRACE_RUN_DIR}/slurm_logs/job.err" >&2)
fi

# === Nsight Systems ===
export NSYS_ENABLE="${NSYS_ENABLE:-1}"
export NSYS_DIR="${TRACE_RUN_DIR}/nsight"
export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
export NSYS_DELAY="${NSYS_DELAY:-0}"

# Keep both knobs separate:
# - NSYS_PROFILE_SERVER wraps the API server process on the head node.
# - NSYS_PROFILE_WORKERS enables vLLM's --ray-workers-use-nsight on Ray actors.
#
# Default server profiling OFF: nested nsys on the head node breaks rank-0 worker
# traces and server report finalization can consume several minutes at job end
# (which caused job 7680791 to hit TIME LIMIT before worker copies finished).
export NSYS_PROFILE_SERVER="${NSYS_PROFILE_SERVER:-0}"
export NSYS_PROFILE_WORKERS="${NSYS_PROFILE_WORKERS:-1}"
export NSYS_PROFILE_RAY="${NSYS_PROFILE_RAY:-0}"

# === NCCL logs ===
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
echo "NSYS_DIR=${NSYS_DIR}"
echo "NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"
echo "NSYS_TRACE=${NSYS_TRACE}"
echo "NSYS_DELAY=${NSYS_DELAY}"
echo "NSYS_PROFILE_SERVER=${NSYS_PROFILE_SERVER}"
echo "NSYS_PROFILE_WORKERS=${NSYS_PROFILE_WORKERS}"
echo "NSYS_PROFILE_RAY=${NSYS_PROFILE_RAY}"
echo "RAY_CGRAPH_get_timeout=${RAY_CGRAPH_get_timeout}"
echo "nsys path: $(command -v nsys || echo '<not found>')"
nsys --version || true

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

VENV_DIR="${REPO_ROOT}/.venv"
echo "REPO_ROOT=${REPO_ROOT}"
echo "VENV_DIR=${VENV_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

PYTHON_PATH="$(command -v python)"
EXPECTED_PYTHON="${VENV_DIR}/bin/python"
echo "Using python: ${PYTHON_PATH}"

if [ "${PYTHON_PATH}" != "${EXPECTED_PYTHON}" ]; then
  echo "Error: python did not resolve to venv interpreter." >&2
  echo "Expected: ${EXPECTED_PYTHON}" >&2
  echo "Got:      ${PYTHON_PATH}" >&2
  exit 1
fi

echo "After venv: python=$(command -v python) ray=$(command -v ray 2>/dev/null || echo '<not on PATH>')"
slurm_debug "PATH=${PATH}"

RAY_BIN="${VENV_DIR}/bin/ray"
if [ ! -x "${RAY_BIN}" ]; then
  echo "Error: ray binary not found at ${RAY_BIN}. Install ray into this venv." >&2
  echo "Hint: source \"${VENV_DIR}/bin/activate\" && python -m pip install ray" >&2
  exit 1
fi

echo "Using RAY_BIN=${RAY_BIN}"

# Optional: patch vLLM Ray-worker Nsight tracing to include NVTX if this version hardcodes cuda,cudnn,cublas.
# This is safe if the string is absent; it just prints "patched files: []".
if [ "${PATCH_VLLM_RAY_NSYS:-1}" = "1" ]; then
  echo "Checking whether vLLM Ray-worker Nsight trace list needs NVTX patch..."
  python - <<'PY' || true
import pathlib
import vllm

root = pathlib.Path(vllm.__file__).parent
target = "cuda,nvtx,osrt,cudnn,cublas"
patched = []

for path in root.rglob("*.py"):
    try:
        text = path.read_text()
    except Exception:
        continue

    if "worker_process_%p" not in text:
        continue

    new_text = text.replace('"t": "cuda,cudnn,cublas"', f'"t": "{target}"')
    new_text = new_text.replace("'t': 'cuda,cudnn,cublas'", f"'t': '{target}'")

    if new_text != text:
        path.write_text(new_text)
        patched.append(str(path))

print("Ray-worker Nsight target trace:", target)
print("patched files:", patched)
PY
fi

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
HOST="${HOST:-${HEAD_NODE_IP}}"
PORT="${PORT:-8000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}"
TP="${TP:-1}"
PP="${PP:-${NUM_NODES}}"
EP="${EP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT_multi_node.sh"

# Nsight-wrapped Ray workers slow model load substantially (job 7685596 needed ~16m).
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
else
  HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
fi
RAY_READY_TIMEOUT="${RAY_READY_TIMEOUT:-600}"
SERVER_SHUTDOWN_TIMEOUT="${SERVER_SHUTDOWN_TIMEOUT:-600}"
WORKER_REPORT_FLUSH_SLEEP="${WORKER_REPORT_FLUSH_SLEEP:-90}"
NODE_COPY_INTERVAL="${NODE_COPY_INTERVAL:-20}"

echo "=== runtime knobs ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} TP=${TP} PP=${PP} EP=${EP}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES} CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "SP=${SP} SD=${SD} NUM_PROMPTS=${NUM_PROMPTS}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "JOB_START_EPOCH=${JOB_START_EPOCH} JOB_SESSION_TS=${JOB_SESSION_TS}"
echo "HEALTH_TIMEOUT=${HEALTH_TIMEOUT}"
echo "RAY_READY_TIMEOUT=${RAY_READY_TIMEOUT}"
echo "SERVER_SHUTDOWN_TIMEOUT=${SERVER_SHUTDOWN_TIMEOUT}"
echo "WORKER_REPORT_FLUSH_SLEEP=${WORKER_REPORT_FLUSH_SLEEP}"
echo "NODE_COPY_INTERVAL=${NODE_COPY_INTERVAL}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_NET=${NCCL_NET} NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "NCCL_DEBUG=${NCCL_DEBUG} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
echo "SERVE_SCRIPT=${SERVE_SCRIPT}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set"
fi

SERVER_STEP_PID=""
HEAD_RAY_PID=""
WORKER_RAY_PIDS=""

nsys_rep_belongs_to_current_job() {
  local path="$1"
  local base session_ts
  base="$(basename "${path}")"
  session_ts="$(
    printf '%s' "${base}" | sed -n \
      's/^session_\([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]\)_.*/\1/p'
  )"
  if [ -z "${session_ts}" ]; then
    return 1
  fi
  # Ray session dir timestamps are lexically sortable in this format.
  [ "${session_ts}" \> "${JOB_SESSION_TS}" ] || [ "${session_ts}" = "${JOB_SESSION_TS}" ]
}

count_worker_nsys_reports() {
  local path count=0
  while IFS= read -r -d '' path; do
    if nsys_rep_belongs_to_current_job "${path}"; then
      count=$((count + 1))
    fi
  done < <(
    find "${TRACE_RUN_DIR}/ray_worker_nsight" \
      -type f -name "*.nsys-rep" -print0 2>/dev/null || true
  )
  printf '%s' "${count}"
}

print_copied_worker_reports() {
  local report_count
  report_count="$(count_worker_nsys_reports)"
  echo "=== Copied Ray worker Nsight reports under ${TRACE_RUN_DIR}/ray_worker_nsight (${report_count} *.nsys-rep) ==="
  find "${TRACE_RUN_DIR}/ray_worker_nsight" \
    -type f \
    \( -name "*.nsys-rep" -o -name "*.qdstrm" -o -name "*.sqlite" \) \
    -printf "%p %s bytes\n" 2>/dev/null | sort || true
}

warn_if_worker_reports_missing() {
  local expected="${1:-${NUM_NODES}}"
  local actual stale
  actual="$(count_worker_nsys_reports)"
  stale="$(
    find "${TRACE_RUN_DIR}/ray_worker_nsight" \
      -type f -name "*.nsys-rep" 2>/dev/null | while IFS= read -r path; do
      if ! nsys_rep_belongs_to_current_job "${path}"; then
        printf '%s\n' "${path}"
      fi
    done | wc -l | tr -d ' '
  )"
  if [ "${stale}" != "0" ]; then
    echo "WARNING: ignoring ${stale} stale *.nsys-rep file(s) from prior Ray sessions on these nodes." >&2
    find "${TRACE_RUN_DIR}/ray_worker_nsight" -type f -name "*.nsys-rep" 2>/dev/null | while IFS= read -r path; do
      if ! nsys_rep_belongs_to_current_job "${path}"; then
        echo "WARNING: stale trace (not from this job): ${path}" >&2
      fi
    done
  fi
  if [ "${actual}" -lt "${expected}" ]; then
    echo "WARNING: expected at least ${expected} current-job Ray-worker *.nsys-rep file(s), found ${actual}." >&2
    echo "WARNING: worker traces live on each node under /tmp/ray/session_*/logs/nsight/ until copied." >&2
    echo "WARNING: check ${TRACE_RUN_DIR}/ray_step_logs/ray_{head,worker}_*.out for copy-loop output." >&2
  fi
}

copy_worker_reports_from_shared_dest() {
  # Reports are copied continuously by the per-node copier loops inside the Ray srun steps.
  # This function only prints what already arrived; it does not start new srun steps.
  print_copied_worker_reports
}

stop_server_cleanly() {
  set +e

  if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Stopping vLLM/Nsight server process cleanly..."
    kill -INT "${SERVER_STEP_PID}" 2>/dev/null || true

    for _ in $(seq 1 "${SERVER_SHUTDOWN_TIMEOUT}"); do
      if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done

    if kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
      echo "Server still alive after ${SERVER_SHUTDOWN_TIMEOUT}s; sending SIGTERM"
      kill -TERM "${SERVER_STEP_PID}" 2>/dev/null || true
      sleep 20
    fi

    if kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
      echo "Server still alive after SIGTERM; sending SIGKILL"
      kill -KILL "${SERVER_STEP_PID}" 2>/dev/null || true
    fi

    wait "${SERVER_STEP_PID}" 2>/dev/null || true
    SERVER_STEP_PID=""
  fi

  set -e
}

stop_ray_steps() {
  set +e

  echo "Stopping Ray background srun steps. Per-node EXIT traps should copy final Nsight files."

  if [ -n "${HEAD_RAY_PID}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
    kill -TERM "${HEAD_RAY_PID}" 2>/dev/null || true
  fi

  for pid in ${WORKER_RAY_PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  sleep 20

  if [ -n "${HEAD_RAY_PID}" ]; then
    wait "${HEAD_RAY_PID}" 2>/dev/null || true
    HEAD_RAY_PID=""
  fi

  for pid in ${WORKER_RAY_PIDS}; do
    wait "${pid}" 2>/dev/null || true
  done
  WORKER_RAY_PIDS=""

  set -e
}

cleanup() {
  set +e
  echo "=== cleanup trap ==="
  stop_server_cleanly
  echo "Waiting briefly for worker reports before stopping Ray..."
  sleep 20
  copy_worker_reports_from_shared_dest
  stop_ray_steps
  copy_worker_reports_from_shared_dest

  echo "Trace files:"
  find "${TRACE_RUN_DIR}" -maxdepth 6 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true
}
trap cleanup EXIT TERM INT

make_ray_node_command() {
  local node_name="$1"
  local node_ip="$2"
  local ray_role="$3"

  cat <<EOF
set -Eeuo pipefail

source "${VENV_DIR}/bin/activate"

$(declare -f interface_for_ip)
$(declare -f interface_has_ip)
$(declare -f configure_socket_ifnames)

unset GLOO_SOCKET_IFNAME
export VLLM_HOST_IP="${node_ip}"
export NCCL_DEBUG="${NCCL_DEBUG}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}"
export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"
export NCCL_NET="${NCCL_NET}"
export NCCL_IB_HCA="${NCCL_IB_HCA}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout}"

configure_socket_ifnames "${node_ip}" 0

NODE_NAME="${node_name}"
NODE_NSIGHT_DEST="${TRACE_RUN_DIR}/ray_worker_nsight/${node_name}"
NODE_COPY_INTERVAL="${NODE_COPY_INTERVAL}"
JOB_START_EPOCH="${JOB_START_EPOCH}"
JOB_SESSION_TS="${JOB_SESSION_TS}"
COPIER_PID=""

mkdir -p "\${NODE_NSIGHT_DEST}"

clean_local_ray_state() {
  echo "\${NODE_NAME}: stopping local Ray and removing stale /tmp/ray sessions from prior jobs"
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
  find /tmp/ray -maxdepth 1 -type d -name 'session_*' 2>/dev/null | while IFS= read -r stale_session; do
    echo "\${NODE_NAME}: rm -rf \${stale_session}"
    rm -rf "\${stale_session}" 2>/dev/null || true
  done
  rm -f /tmp/ray/session_latest 2>/dev/null || true
}

ray_session_is_current() {
  local session_dir="\$1"
  local session_ts session_epoch=""
  session_ts="\$(basename "\${session_dir}" | sed -n 's/^session_\\([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]\\)_.*/\\1/p')"
  if [ -n "\${session_ts}" ]; then
    if [ "\${session_ts}" \> "\${JOB_SESSION_TS}" ] || [ "\${session_ts}" = "\${JOB_SESSION_TS}" ]; then
      return 0
    fi
    return 1
  fi
  session_epoch="\$(stat -c '%Y' "\${session_dir}" 2>/dev/null || echo 0)"
  [ "\${session_epoch}" -ge "\${JOB_START_EPOCH}" ]
}

copy_ray_nsight_once() {
  set +e
  mkdir -p "\${NODE_NSIGHT_DEST}"

  echo "[\$(date -Is 2>/dev/null || date)] \${NODE_NAME}: scanning /tmp/ray for Ray-worker Nsight files (job sessions >= \${JOB_SESSION_TS})"

  found=0
  while IFS= read -r -d '' f; do
    session_dir="\$(echo "\${f}" | sed -n 's#\\(/tmp/ray/session_[^/]*\\)/logs/nsight/.*#\\1#p')"
    if [ -z "\${session_dir}" ] || ! ray_session_is_current "\${session_dir}"; then
      echo "\${NODE_NAME}: skip stale Nsight file \${f}"
      continue
    fi

    found=1
    size=\$(stat -c '%s' "\${f}" 2>/dev/null || echo 0)
    base=\$(basename "\${f}")
    session=\$(basename "\${session_dir}")

    out="\${NODE_NSIGHT_DEST}/\${session}_\${base}"

    echo "\${NODE_NAME}: copy candidate \${f} \${size} bytes -> \${out}"

    # Atomic-ish copy to avoid leaving a corrupt destination if interrupted.
    cp -f "\${f}" "\${out}.tmp" 2>/dev/null && mv -f "\${out}.tmp" "\${out}" 2>/dev/null || true
  done < <(
    find /tmp/ray \
      -path '*/logs/nsight/*' \
      -type f \
      \\( -name '*.nsys-rep' -o -name '*.qdstrm' -o -name '*.sqlite' \\) \
      -print0 2>/dev/null || true
  )

  if [ "\${found}" = "0" ]; then
    echo "\${NODE_NAME}: no current-job Ray-worker Nsight files under /tmp/ray yet"
  fi

  echo "\${NODE_NAME}: currently copied files:"
  find "\${NODE_NSIGHT_DEST}" -type f -printf '  %p %s bytes\\n' 2>/dev/null | sort || true
  set -e
}

copy_ray_nsight_loop() {
  while true; do
    copy_ray_nsight_once || true
    sleep "\${NODE_COPY_INTERVAL}"
  done
}

ray_step_cleanup() {
  set +e
  trap - EXIT INT TERM

  echo "\${NODE_NAME}: Ray step cleanup starting"
  copy_ray_nsight_once || true

  if [ -n "\${COPIER_PID}" ]; then
    kill "\${COPIER_PID}" 2>/dev/null || true
    wait "\${COPIER_PID}" 2>/dev/null || true
  fi

  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true

  # One more pass after ray stop, because worker nsys reports may finalize during shutdown.
  sleep 10
  copy_ray_nsight_once || true

  echo "\${NODE_NAME}: Ray step cleanup complete"
  exit 0
}

trap ray_step_cleanup EXIT INT TERM

echo "Ray ${ray_role} host: \$(hostname) node=\${NODE_NAME} VLLM_HOST_IP=\${VLLM_HOST_IP}"
echo "Ray ${ray_role} will continuously copy /tmp/ray/session_*/logs/nsight to \${NODE_NSIGHT_DEST}"
echo "Ray ${ray_role} TMPDIR=\${TMPDIR:-<unset>} RAY_TMPDIR=\${RAY_TMPDIR:-<unset>}"

clean_local_ray_state

copy_ray_nsight_loop &
COPIER_PID="\$!"

if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_RAY}" = "1" ]; then
  echo "Profiling Ray ${ray_role} daemon itself with Nsight Systems"
fi

EOF
}

echo "=== Ray head (background srun) ==="
echo "Starting head node ${HEAD_NODE}..."

RAY_HEAD_CMD="$(make_ray_node_command "${HEAD_NODE}" "${HEAD_NODE_IP}" "head")
if [ \"${NSYS_ENABLE}\" = \"1\" ] && [ \"${NSYS_PROFILE_RAY}\" = \"1\" ]; then
  nsys profile \\
    --force-overwrite=true \\
    --trace=\"${NSYS_TRACE}\" \\
    --sample=none \\
    --delay=\"${NSYS_DELAY}\" \\
    --output=\"${NSYS_DIR}/ray_head_${HEAD_NODE}\" \\
    \"${RAY_BIN}\" start --block \\
      --head \\
      --node-ip-address=${HEAD_NODE_IP} \\
      --port=${RAY_PORT} \\
      --num-gpus=${GPUS_PER_NODE} \\
      --num-cpus=${CPUS_PER_TASK} \\
      --disable-usage-stats
else
  \"${RAY_BIN}\" start --block \\
    --head \\
    --node-ip-address=${HEAD_NODE_IP} \\
    --port=${RAY_PORT} \\
    --num-gpus=${GPUS_PER_NODE} \\
    --num-cpus=${CPUS_PER_TASK} \\
    --disable-usage-stats
fi"

srun \
  --nodelist "${HEAD_NODE}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --output="${TRACE_RUN_DIR}/ray_step_logs/ray_head_${HEAD_NODE}.out" \
  --error="${TRACE_RUN_DIR}/ray_step_logs/ray_head_${HEAD_NODE}.err" \
  bash -lc "${RAY_HEAD_CMD}" &

HEAD_RAY_PID=$!
echo "HEAD_RAY_PID=${HEAD_RAY_PID}"

sleep 20

if [ -n "${WORKER_NODES}" ]; then
  echo "=== Ray workers ==="

  for WORKER in ${WORKER_NODES}; do
    WORKER_IP="$(resolve_host_ip "${WORKER}")"

    if [ -z "${WORKER_IP}" ]; then
      echo "Error: could not resolve IP for worker ${WORKER}." >&2
      exit 1
    fi

    echo "Starting Ray worker node: ${WORKER} with IP ${WORKER_IP}"

    RAY_WORKER_CMD="$(make_ray_node_command "${WORKER}" "${WORKER_IP}" "worker")
if [ \"${NSYS_ENABLE}\" = \"1\" ] && [ \"${NSYS_PROFILE_RAY}\" = \"1\" ]; then
  nsys profile \\
    --force-overwrite=true \\
    --trace=\"${NSYS_TRACE}\" \\
    --sample=none \\
    --delay=\"${NSYS_DELAY}\" \\
    --output=\"${NSYS_DIR}/ray_worker_${WORKER}\" \\
    \"${RAY_BIN}\" start --block \\
      --address=${HEAD_NODE_IP}:${RAY_PORT} \\
      --node-ip-address=${WORKER_IP} \\
      --num-gpus=${GPUS_PER_NODE} \\
      --num-cpus=${CPUS_PER_TASK} \\
      --disable-usage-stats
else
  \"${RAY_BIN}\" start --block \\
    --address=${HEAD_NODE_IP}:${RAY_PORT} \\
    --node-ip-address=${WORKER_IP} \\
    --num-gpus=${GPUS_PER_NODE} \\
    --num-cpus=${CPUS_PER_TASK} \\
    --disable-usage-stats
fi"

    srun \
      --nodelist "${WORKER}" \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --gpus-per-task="${GPUS_PER_NODE}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --output="${TRACE_RUN_DIR}/ray_step_logs/ray_worker_${WORKER}.out" \
      --error="${TRACE_RUN_DIR}/ray_step_logs/ray_worker_${WORKER}.err" \
      bash -lc "${RAY_WORKER_CMD}" &

    WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
    echo "Worker Ray step pid: $! (WORKER_RAY_PIDS=${WORKER_RAY_PIDS})"
  done

  sleep 20
fi

echo "=== waiting for Ray cluster ==="
echo "Waiting for Ray to show ${NUM_NODES} GPUs at ${RAY_ADDRESS}..."

RAY_READY_START="$(date +%s)"
RAY_READY_ATTEMPT=0

while true; do
  RAY_READY_ATTEMPT=$((RAY_READY_ATTEMPT + 1))

  if python - <<PY
import ray
ray.init(address="${RAY_ADDRESS}", ignore_reinit_error=True)
nodes = ray.nodes()
resources = ray.cluster_resources()
available = ray.available_resources()
print("attempt ${RAY_READY_ATTEMPT}")
print("Ray nodes:")
for node in nodes:
    print(
        f"  {node.get('NodeManagerAddress')} "
        f"alive={node.get('Alive')} "
        f"resources={node.get('Resources')}"
    )
print("cluster_resources:", resources)
print("available_resources:", available)
ray.shutdown()
assert resources.get("GPU", 0) >= ${NUM_NODES}
PY
  then
    echo "Ray cluster is ready with ${NUM_NODES} GPUs."
    break
  fi

  now="$(date +%s)"
  if [ $((now - RAY_READY_START)) -gt "${RAY_READY_TIMEOUT}" ]; then
    echo "ERROR: Ray cluster did not reach ${NUM_NODES} GPUs within ${RAY_READY_TIMEOUT}s." >&2

    echo "--- head ${HEAD_NODE} out tail ---"
    tail -n 120 "${TRACE_RUN_DIR}/ray_step_logs/ray_head_${HEAD_NODE}.out" 2>/dev/null || true
    echo "--- head ${HEAD_NODE} err tail ---"
    tail -n 120 "${TRACE_RUN_DIR}/ray_step_logs/ray_head_${HEAD_NODE}.err" 2>/dev/null || true

    for WORKER in ${WORKER_NODES}; do
      echo "--- worker ${WORKER} out tail ---"
      tail -n 120 "${TRACE_RUN_DIR}/ray_step_logs/ray_worker_${WORKER}.out" 2>/dev/null || true
      echo "--- worker ${WORKER} err tail ---"
      tail -n 120 "${TRACE_RUN_DIR}/ray_step_logs/ray_worker_${WORKER}.err" 2>/dev/null || true
    done

    copy_worker_reports_from_shared_dest
    exit 1
  fi

  echo "Ray cluster not ready yet; sleeping 5s."

  for WORKER in ${WORKER_NODES}; do
    echo "--- worker ${WORKER} out tail ---"
    tail -n 30 "${TRACE_RUN_DIR}/ray_step_logs/ray_worker_${WORKER}.out" 2>/dev/null || true
    echo "--- worker ${WORKER} err tail ---"
    tail -n 30 "${TRACE_RUN_DIR}/ray_step_logs/ray_worker_${WORKER}.err" 2>/dev/null || true
  done

  sleep 5
done

echo "=== vLLM api_server (background process) ==="
echo "Starting vLLM server on head node..."

VLLM_TRACE_FLAGS=()
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  VLLM_TRACE_FLAGS+=(
    --ray-workers-use-nsight
    --enable-layerwise-nvtx-tracing
    --enable-logging-iteration-details
  )
fi

echo "VLLM_TRACE_FLAGS=${VLLM_TRACE_FLAGS[*]:-<none>}"

if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_SERVER}" = "1" ]; then
  echo "Profiling API server with Nsight Systems"
  echo "API-server Nsight output: ${NSYS_DIR}/vllm_api_server_${HEAD_NODE}.nsys-rep"
  echo "Ray-worker Nsight output should be copied continuously to: ${TRACE_RUN_DIR}/ray_worker_nsight/<node>/"

  nsys profile \
    --force-overwrite=true \
    --trace="${NSYS_TRACE}" \
    --sample=none \
    --cuda-event-trace=false \
    --delay="${NSYS_DELAY}" \
    --output="${NSYS_DIR}/vllm_api_server_${HEAD_NODE}" \
    python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_ID}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --distributed-executor-backend ray \
      --tensor-parallel-size "${TP}" \
      --pipeline-parallel-size "${PP}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --enforce-eager \
      "${VLLM_TRACE_FLAGS[@]}" \
      --disable-custom-all-reduce \
      > "${TRACE_RUN_DIR}/server/vllm_server.log" 2>&1 &
else
  echo "Starting API server without server-level Nsight profile"
  echo "Ray-worker Nsight output should be copied continuously to: ${TRACE_RUN_DIR}/ray_worker_nsight/<node>/"

  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --distributed-executor-backend ray \
    --tensor-parallel-size "${TP}" \
    --pipeline-parallel-size "${PP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enforce-eager \
    "${VLLM_TRACE_FLAGS[@]}" \
    --disable-custom-all-reduce \
    > "${TRACE_RUN_DIR}/server/vllm_server.log" 2>&1 &
fi

SERVER_STEP_PID=$!
echo "Started vLLM server process wrapper pid=${SERVER_STEP_PID}. Waiting for /health ..."

_health_wait_n=0
_health_start=$(date +%s)

until curl -fsS "http://${HEAD_NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server process exited before becoming ready." >&2
    wait "${SERVER_STEP_PID}" || true
    exit 1
  fi

  now=$(date +%s)
  if [ $((now - _health_start)) -gt "${HEALTH_TIMEOUT}" ]; then
    echo "Timed out waiting for /health after ${HEALTH_TIMEOUT}s" >&2
    echo "Last lines of ${TRACE_RUN_DIR}/server/vllm_server.log:" >&2
    tail -n 80 "${TRACE_RUN_DIR}/server/vllm_server.log" 2>/dev/null >&2 || true
    copy_worker_reports_from_shared_dest
    exit 1
  fi

  _health_wait_n=$((_health_wait_n + 1))

  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ] || [ "$((_health_wait_n % 12))" -eq 0 ]; then
    echo "Still waiting for http://${HEAD_NODE_IP}:${PORT}/health (attempt ${_health_wait_n}) ..."
    python - <<PY || true
import ray
ray.init(address="${RAY_ADDRESS}", ignore_reinit_error=True)
print("cluster_resources:", ray.cluster_resources())
print("available_resources:", ray.available_resources())
ray.shutdown()
PY
    copy_worker_reports_from_shared_dest
  fi

  sleep 5
done

unset _health_wait_n
unset _health_start

echo "Server is healthy. Running ${SERVE_SCRIPT} ..."
echo "SP=${SP} SD=${SD} NUM_PROMPTS=${NUM_PROMPTS}"
echo "Expect Ray-worker Nsight under: ${TRACE_RUN_DIR}/ray_worker_nsight/<node>/session_*_worker_process_*.nsys-rep"
print_copied_worker_reports

HOST="${HEAD_NODE_IP}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" \
  SP="${SP}" SD="${SD}" NUM_PROMPTS="${NUM_PROMPTS}" \
  HEAD_NODE_IP="${HEAD_NODE_IP}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" CPUS_PER_TASK="${CPUS_PER_TASK}" \
  RAY_PORT="${RAY_PORT}" bash "${SERVE_SCRIPT}" "${SLURM_JOB_ID}" "${HEAD_NODE}"

echo "Workload finished."
echo "Current worker report copies before server shutdown:"
copy_worker_reports_from_shared_dest

stop_server_cleanly

echo "Waiting ${WORKER_REPORT_FLUSH_SLEEP}s for Ray-worker Nsight reports to finalize and be copied by node-local copy loops..."
sleep "${WORKER_REPORT_FLUSH_SLEEP}"

echo "Worker report copies after server shutdown:"
copy_worker_reports_from_shared_dest
warn_if_worker_reports_missing "${NUM_NODES}"

stop_ray_steps

echo "Worker report copies after Ray shutdown:"
copy_worker_reports_from_shared_dest
warn_if_worker_reports_missing "${NUM_NODES}"

echo "Server log:"
find "${TRACE_RUN_DIR}/server" -maxdepth 1 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "Server Nsight reports:"
find "${NSYS_DIR}" -maxdepth 1 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "NCCL logs:"
find "${TRACE_RUN_DIR}/nccl_logs" -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "Trace files:"
find "${TRACE_RUN_DIR}" -maxdepth 6 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

trap - EXIT TERM INT

echo "Done. Results in ${TRACE_RUN_DIR}"