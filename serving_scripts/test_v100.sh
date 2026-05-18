    #!/usr/bin/env bash
#SBATCH --nodelist=htc-g[048-049]
#SBATCH --job-name=vllm-v100-nsys-debug
#SBATCH --nodes=2
#SBATCH --partition=interactive
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority

set -Eeuo pipefail

SCRIPT_VERSION="v100-ray-worker-nsys-debug"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
RAY_PORT="${RAY_PORT:-6389}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
TP="${TP:-1}"
PP="${PP:-2}"

NSYS_PROFILE_SERVER="${NSYS_PROFILE_SERVER:-1}"
NSYS_PROFILE_WORKERS="${NSYS_PROFILE_WORKERS:-1}"
NSYS_PROFILE_RAY="${NSYS_PROFILE_RAY:-0}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
NODE_COPY_INTERVAL="${NODE_COPY_INTERVAL:-10}"
WORKER_REPORT_FLUSH_SLEEP="${WORKER_REPORT_FLUSH_SLEEP:-45}"

HEAD_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
WORKER_NODES="$(scontrol show hostnames "${SLURM_NODELIST}" | tail -n+2)"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}"

TRACE_BASE="${TRACE_BASE:-/data/engs-glass/catz0932/inference-traces/vllm/results}"
TRACE_RUN_DIR="${TRACE_BASE}/v100_debug_${SLURM_JOB_ID}"
NSYS_DIR="${TRACE_RUN_DIR}/nsight"

mkdir -p \
  "${TRACE_RUN_DIR}/ray_worker_nsight" \
  "${TRACE_RUN_DIR}/ray_step_logs" \
  "${TRACE_RUN_DIR}/server" \
  "${NSYS_DIR}"

echo "=== V100 Ray/vLLM Nsight debug ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODES=${WORKER_NODES}"
echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
echo "MODEL_ID=${MODEL_ID}"
echo "TP=${TP} PP=${PP}"
echo "NSYS_PROFILE_SERVER=${NSYS_PROFILE_SERVER}"
echo "NSYS_PROFILE_WORKERS=${NSYS_PROFILE_WORKERS}"
echo "NSYS_PROFILE_RAY=${NSYS_PROFILE_RAY}"

resolve_host_ip() {
  local nodename="$1"
  local ip=""

  ip=$(getent hosts "${nodename}" 2>/dev/null | awk '{print $1}' | awk '/^[0-9]+\./ {print; exit}' || true)

  if [ -z "${ip}" ]; then
    ip=$(
      srun --overlap --nodelist="${nodename}" --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=1G \
        bash -lc "hostname -I | tr ' ' '\n' | awk '/^[0-9]+\./ {print; exit}'" 2>/dev/null || true
    )
  fi

  printf "%s" "${ip}"
}

HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
if [ -z "${HEAD_NODE_IP}" ]; then
  echo "ERROR: could not resolve HEAD_NODE_IP for ${HEAD_NODE}" >&2
  exit 1
fi

HOST="${HEAD_NODE_IP}"
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
echo "HOST=${HOST}"
echo "RAY_ADDRESS=${RAY_ADDRESS}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="/data/engs-glass/catz0932/inference-traces/vllm"
fi

VENV_DIR="${REPO_ROOT}/.venv"
source "${VENV_DIR}/bin/activate"

RAY_BIN="${VENV_DIR}/bin/ray"
if [ ! -x "${RAY_BIN}" ]; then
  echo "ERROR: ray binary not found at ${RAY_BIN}" >&2
  exit 1
fi

echo "python=$(command -v python)"
echo "ray=${RAY_BIN}"
echo "nsys=$(command -v nsys || true)"
nsys --version || true

export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"
mkdir -p "${TRACE_RUN_DIR}/nccl_logs"

# Patch vLLM's Ray-worker Nsight trace list to include NVTX, if your checkout hardcodes cuda,cudnn,cublas.
if [ "${PATCH_VLLM_RAY_NSYS:-1}" = "1" ]; then
  echo "Patching vLLM Ray-worker Nsight trace list if needed..."
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

    new = text.replace('"t": "cuda,cudnn,cublas"', f'"t": "{target}"')
    new = new.replace("'t': 'cuda,cudnn,cublas'", f"'t': '{target}'")

    if new != text:
        path.write_text(new)
        patched.append(str(path))

print("patched files:", patched)
PY
fi

SERVER_PID=""
HEAD_RAY_PID=""
WORKER_RAY_PIDS=""

print_reports() {
  echo "=== copied Ray-worker Nsight files ==="
  find "${TRACE_RUN_DIR}/ray_worker_nsight" \
    -type f \
    \( -name "*.nsys-rep" -o -name "*.qdstrm" -o -name "*.sqlite" \) \
    -printf "%p %s bytes\n" 2>/dev/null | sort || true

  echo "=== server Nsight files ==="
  find "${NSYS_DIR}" \
    -type f \
    -printf "%p %s bytes\n" 2>/dev/null | sort || true
}

stop_server() {
  set +e
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping vLLM server..."
    kill -INT "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 180); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
      sleep 10
    fi
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  set -e
}

stop_ray_steps() {
  set +e
  echo "Stopping Ray srun steps..."
  [ -n "${HEAD_RAY_PID}" ] && kill -TERM "${HEAD_RAY_PID}" 2>/dev/null || true
  for pid in ${WORKER_RAY_PIDS}; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 15
  [ -n "${HEAD_RAY_PID}" ] && wait "${HEAD_RAY_PID}" 2>/dev/null || true
  for pid in ${WORKER_RAY_PIDS}; do
    wait "${pid}" 2>/dev/null || true
  done
  HEAD_RAY_PID=""
  WORKER_RAY_PIDS=""
  set -e
}

cleanup() {
  set +e
  echo "=== cleanup ==="
  stop_server
  echo "Waiting for worker report copy before stopping Ray..."
  sleep "${WORKER_REPORT_FLUSH_SLEEP}"
  print_reports
  stop_ray_steps
  sleep 10
  print_reports
  echo "=== all trace files ==="
  find "${TRACE_RUN_DIR}" -maxdepth 6 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true
}
trap cleanup EXIT TERM INT

make_ray_node_command() {
  local node_name="$1"
  local node_ip="$2"
  local role="$3"

  cat <<EOF
set -Eeuo pipefail

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0
source "${VENV_DIR}/bin/activate"

export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s}"

export NODE_NAME="${node_name}"
export NODE_COPY_INTERVAL="${NODE_COPY_INTERVAL}"
export NODE_NSIGHT_DEST="${TRACE_RUN_DIR}/ray_worker_nsight/${node_name}"
mkdir -p "\${NODE_NSIGHT_DEST}"

copy_ray_nsight_once() {
  set +e
  mkdir -p "\${NODE_NSIGHT_DEST}"
  echo "[\$(date -Is 2>/dev/null || date)] \${NODE_NAME}: scan /tmp/ray/session_*/logs/nsight"

  while IFS= read -r -d '' f; do
    base=\$(basename "\${f}")
    session=\$(echo "\${f}" | sed -n 's#.*\\(session_[^/]*\\)/logs/nsight/.*#\\1#p')
    [ -n "\${session}" ] || session="session_unknown"
    out="\${NODE_NSIGHT_DEST}/\${session}_\${base}"
    size=\$(stat -c '%s' "\${f}" 2>/dev/null || echo 0)
    echo "\${NODE_NAME}: copy \${f} \${size} bytes -> \${out}"
    cp -f "\${f}" "\${out}.tmp" 2>/dev/null && mv -f "\${out}.tmp" "\${out}" 2>/dev/null || true
  done < <(
    find /tmp/ray \
      -path '*/logs/nsight/*' \
      -type f \
      \\( -name '*.nsys-rep' -o -name '*.qdstrm' -o -name '*.sqlite' \\) \
      -print0 2>/dev/null || true
  )

  find "\${NODE_NSIGHT_DEST}" -type f -printf "\${NODE_NAME}: copied %p %s bytes\\n" 2>/dev/null | sort || true
  set -e
}

copy_ray_nsight_loop() {
  while true; do
    copy_ray_nsight_once || true
    sleep "\${NODE_COPY_INTERVAL}"
  done
}

COPIER_PID=""

ray_step_cleanup() {
  set +e
  trap - EXIT TERM INT
  echo "\${NODE_NAME}: ray_step_cleanup"
  copy_ray_nsight_once || true
  [ -n "\${COPIER_PID}" ] && kill "\${COPIER_PID}" 2>/dev/null || true
  [ -n "\${COPIER_PID}" ] && wait "\${COPIER_PID}" 2>/dev/null || true
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
  sleep 10
  copy_ray_nsight_once || true
  echo "\${NODE_NAME}: cleanup complete"
  exit 0
}

trap ray_step_cleanup EXIT TERM INT

echo "Ray ${role}: host=\$(hostname) node=${node_name} ip=${node_ip}"
echo "Ray ${role}: copying worker reports to \${NODE_NSIGHT_DEST}"

copy_ray_nsight_loop &
COPIER_PID="\$!"

EOF
}

echo "=== starting Ray head ==="
RAY_HEAD_CMD="$(make_ray_node_command "${HEAD_NODE}" "${HEAD_NODE_IP}" "head")
if [ \"${NSYS_PROFILE_RAY}\" = \"1\" ]; then
  nsys profile \\
    --force-overwrite=true \\
    --trace=\"${NSYS_TRACE}\" \\
    --sample=none \\
    --output=\"${NSYS_DIR}/ray_head_${HEAD_NODE}\" \\
    \"${RAY_BIN}\" start --block \\
      --head \\
      --node-ip-address=${HEAD_NODE_IP} \\
      --port=${RAY_PORT} \\
      --num-gpus=1 \\
      --num-cpus=${SLURM_CPUS_PER_TASK} \\
      --object-store-memory=8000000000 \\
      --disable-usage-stats
else
  \"${RAY_BIN}\" start --block \\
    --head \\
    --node-ip-address=${HEAD_NODE_IP} \\
    --port=${RAY_PORT} \\
    --num-gpus=1 \\
    --num-cpus=${SLURM_CPUS_PER_TASK} \\
    --object-store-memory=8000000000 \\
    --disable-usage-stats
fi"

srun \
  --nodelist "${HEAD_NODE}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --output="${TRACE_RUN_DIR}/ray_step_logs/ray_head_${HEAD_NODE}.out" \
  --error="${TRACE_RUN_DIR}/ray_step_logs/ray_head_${HEAD_NODE}.err" \
  bash -lc "${RAY_HEAD_CMD}" &

HEAD_RAY_PID=$!
echo "HEAD_RAY_PID=${HEAD_RAY_PID}"
sleep 20

echo "=== starting Ray workers ==="
for WORKER in ${WORKER_NODES}; do
  WORKER_IP="$(resolve_host_ip "${WORKER}")"
  echo "WORKER=${WORKER} WORKER_IP=${WORKER_IP}"

  RAY_WORKER_CMD="$(make_ray_node_command "${WORKER}" "${WORKER_IP}" "worker")
\"${RAY_BIN}\" start --block \\
  --address=${HEAD_NODE_IP}:${RAY_PORT} \\
  --node-ip-address=${WORKER_IP} \\
  --num-gpus=1 \\
  --num-cpus=${SLURM_CPUS_PER_TASK} \\
  --object-store-memory=8000000000 \\
  --disable-usage-stats"

  srun \
    --nodelist "${WORKER}" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gpus-per-task=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --output="${TRACE_RUN_DIR}/ray_step_logs/ray_worker_${WORKER}.out" \
    --error="${TRACE_RUN_DIR}/ray_step_logs/ray_worker_${WORKER}.err" \
    bash -lc "${RAY_WORKER_CMD}" &

  WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
done

sleep 25

echo "=== Ray status ==="
"${RAY_BIN}" status --address="${RAY_ADDRESS}" || true

python - <<PY
import ray
ray.init(address="${RAY_ADDRESS}", ignore_reinit_error=True)
print("cluster_resources:", ray.cluster_resources())
print("available_resources:", ray.available_resources())
assert ray.cluster_resources().get("GPU", 0) >= ${NUM_NODES}
ray.shutdown()
PY

echo "=== starting vLLM ==="
VLLM_TRACE_FLAGS=()
if [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  VLLM_TRACE_FLAGS+=(
    --ray-workers-use-nsight
    --enable-layerwise-nvtx-tracing
    --enable-logging-iteration-details
  )
fi

echo "VLLM_TRACE_FLAGS=${VLLM_TRACE_FLAGS[*]:-<none>}"

if [ "${NSYS_PROFILE_SERVER}" = "1" ]; then
  nsys profile \
    --force-overwrite=true \
    --trace="${NSYS_TRACE}" \
    --sample=none \
    --output="${NSYS_DIR}/vllm_api_server_${HEAD_NODE}" \
    python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_ID}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --distributed-executor-backend ray \
      --tensor-parallel-size "${TP}" \
      --pipeline-parallel-size "${PP}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --dtype float16 \
      --gpu-memory-utilization 0.70 \
      --enforce-eager \
      "${VLLM_TRACE_FLAGS[@]}" \
      --disable-custom-all-reduce \
      > "${TRACE_RUN_DIR}/server/vllm_server.log" 2>&1 &
else
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --distributed-executor-backend ray \
    --tensor-parallel-size "${TP}" \
    --pipeline-parallel-size "${PP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype float16 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager \
    "${VLLM_TRACE_FLAGS[@]}" \
    --disable-custom-all-reduce \
    > "${TRACE_RUN_DIR}/server/vllm_server.log" 2>&1 &
fi

SERVER_PID=$!
echo "SERVER_PID=${SERVER_PID}"

echo "Waiting for health..."
_health_start=$(date +%s)
while ! curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server died before health"
    tail -n 200 "${TRACE_RUN_DIR}/server/vllm_server.log" || true
    exit 1
  fi

  now=$(date +%s)
  if [ $((now - _health_start)) -gt "${HEALTH_TIMEOUT}" ]; then
    echo "Timed out waiting for health"
    tail -n 200 "${TRACE_RUN_DIR}/server/vllm_server.log" || true
    exit 1
  fi

  sleep 5
done

echo "Server healthy."

echo "Sending one debug request..."
curl -fsS "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_ID}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one short sentence.\"}],
    \"max_tokens\": 32,
    \"temperature\": 0
  }" | tee "${TRACE_RUN_DIR}/response.json"

echo "Request done."
print_reports

stop_server

echo "Waiting ${WORKER_REPORT_FLUSH_SLEEP}s for worker Nsight finalization/copy..."
sleep "${WORKER_REPORT_FLUSH_SLEEP}"
print_reports

stop_ray_steps

echo "Final reports:"
print_reports

echo "Trace files:"
find "${TRACE_RUN_DIR}" -maxdepth 6 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

trap - EXIT TERM INT
echo "Done: ${TRACE_RUN_DIR}"