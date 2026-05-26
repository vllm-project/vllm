#!/usr/bin/env bash
#SBATCH --job-name=vllm-bench-sharegpt
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

set -euo pipefail

SCRIPT_VERSION="arc-bench-sharegpt-generic-autogen-2026-05-10-v4"

resolve_host_ipv4() {
  local nodename="$1"
  local ip=""

  pick_ipv4() {
    awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'
  }

  ip=$(dig +short "${nodename}" 2>/dev/null | pick_ipv4 || true)

  if [ -z "${ip}" ]; then
    ip=$(getent hosts "${nodename}" 2>/dev/null | awk '{print $1}' | pick_ipv4 || true)
  fi

  if [ -z "${ip}" ]; then
    ip=$(
      srun --nodelist="${nodename}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        bash -c "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true
    )
  fi

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

configure_gloo_ifname() {
  local target_ip="$1"
  local iface="${GLOO_SOCKET_IFNAME:-}"

  if [ -n "${iface}" ] && ! interface_has_ip "${iface}" "${target_ip}"; then
    echo "Ignoring GLOO_SOCKET_IFNAME=${iface}; it does not own ${target_ip} on $(hostname)." >&2
    iface=""
  fi

  if [ -z "${iface}" ]; then
    iface="$(interface_for_ip "${target_ip}")"
  fi

  if [ -n "${iface}" ]; then
    export GLOO_SOCKET_IFNAME="${iface}"
  fi
}

make_model_slug() {
  printf '%s' "$1" | sed -E 's#[/:]+#_#g; s#[^A-Za-z0-9._-]+#_#g'
}

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

VENV_DIR="${REPO_ROOT}/.venv"

echo "=== vLLM ShareGPT benchmark ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
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

python -m pip install -U pip

VLLM_BIN="${VENV_DIR}/bin/vllm"
if [ ! -x "${VLLM_BIN}" ]; then
  echo "Error: vllm binary not found at ${VLLM_BIN}. Run the host setup/install first." >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8}"
MODEL_SLUG="${MODEL_SLUG:-$(make_model_slug "${MODEL_ID}")}"

# Workload shape:
# SP = prompt / prefill token bucket
# SD = decode / output tokens per request
SP="${SP:-128}"
SD="${SD:-128}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets/sharegpt_buckets/${MODEL_SLUG}/sd${SD}}"
DEFAULT_DATASET_PATH="${DATASET_DIR}/sharegpt_sp${SP}.jsonl"
DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"

AUTO_GENERATE_DATASET="${AUTO_GENERATE_DATASET:-1}"
DATASET_NAME="${DATASET_NAME:-Aeala/ShareGPT_Vicuna_unfiltered}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_MAX_PER_BUCKET="${DATASET_MAX_PER_BUCKET:-1000}"
DATASET_MAX_SOURCE_ROWS="${DATASET_MAX_SOURCE_ROWS:-200000}"

NUM_PROMPTS="${NUM_PROMPTS:-100}"
REQUEST_RATE="${REQUEST_RATE:-1}"
BURSTINESS="${BURSTINESS:-1.0}"
SEED="${SEED:-100}"

HOST="${HOST:-}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-/v1/completions}"

CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
RAY_PORT="${RAY_PORT:-6378}"

SAVE_RESULT="${SAVE_RESULT:-0}"
SAVE_DETAILED="${SAVE_DETAILED:-0}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/traces/bench_results}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/${MODEL_SLUG}_sp${SP}_sd${SD}_np${NUM_PROMPTS}}"
RESULT_FILENAME="${RESULT_FILENAME:-}"

RAY_JOBID="${1:-${SLURM_JOB_ID:-}}"
HEAD_NODE="${2:-${HEAD_NODE:-}}"

if [ -z "${HEAD_NODE}" ] && [ -n "${SLURM_NODELIST:-}" ]; then
  HEAD_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
fi

if [ -z "${RAY_JOBID}" ] || [ -z "${HEAD_NODE}" ]; then
  echo "Usage: $0 <ray_jobid> <head_node>" >&2
  echo "Or run inside an sbatch allocation with SLURM_JOB_ID/SLURM_NODELIST set." >&2
  exit 1
fi

if [ -n "${HEAD_NODE_IP:-}" ]; then
  case "${HEAD_NODE_IP}" in
    *.*.*.*) ;;
    *)
      echo "Error: HEAD_NODE_IP must be IPv4, got ${HEAD_NODE_IP}." >&2
      exit 1
      ;;
  esac
else
  HEAD_NODE_IP="$(resolve_host_ipv4 "${HEAD_NODE}")"
fi

if [ -z "${HEAD_NODE_IP}" ]; then
  echo "Failed to resolve an IPv4 HEAD_NODE_IP for ${HEAD_NODE}" >&2
  exit 1
fi

HOST="${HOST:-${HEAD_NODE_IP}}"

configure_gloo_ifname "${HEAD_NODE_IP}"
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo "=== benchmark config ==="
echo "RAY_JOBID=${RAY_JOBID}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
echo "RAY_ADDRESS=${RAY_ADDRESS}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<unset>}"
echo "MODEL_ID=${MODEL_ID}"
echo "MODEL_SLUG=${MODEL_SLUG}"
echo "SP=${SP} SD=${SD}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "DATASET_SPLIT=${DATASET_SPLIT}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "AUTO_GENERATE_DATASET=${AUTO_GENERATE_DATASET}"
echo "DATASET_MAX_PER_BUCKET=${DATASET_MAX_PER_BUCKET}"
echo "DATASET_MAX_SOURCE_ROWS=${DATASET_MAX_SOURCE_ROWS}"
echo "NUM_PROMPTS=${NUM_PROMPTS}"
echo "REQUEST_RATE=${REQUEST_RATE}"
echo "BURSTINESS=${BURSTINESS}"
echo "SEED=${SEED}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "ENDPOINT=${ENDPOINT}"
echo "SAVE_RESULT=${SAVE_RESULT}"
echo "SAVE_DETAILED=${SAVE_DETAILED}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "RESULT_FILENAME=${RESULT_FILENAME:-<default>}"

if [ ! -f "${DATASET_PATH}" ]; then
  echo "Dataset file not found: ${DATASET_PATH}"

  if [ "${AUTO_GENERATE_DATASET}" != "1" ]; then
    echo "AUTO_GENERATE_DATASET=${AUTO_GENERATE_DATASET}; refusing to generate automatically." >&2
    exit 1
  fi

  echo "Generating ShareGPT bucket:"
  echo "  MODEL_ID=${MODEL_ID}"
  echo "  SP=${SP}"
  echo "  SD=${SD}"
  echo "  DATASET_DIR=${DATASET_DIR}"

  mkdir -p "${DATASET_DIR}"
  cd "${REPO_ROOT}"

  python datasets/build_sharegpt_length_buckets.py \
    --model "${MODEL_ID}" \
    --dataset-name "${DATASET_NAME}" \
    --split "${DATASET_SPLIT}" \
    --targets "${SP}" \
    --out-dir "${DATASET_DIR}" \
    --max-source-rows "${DATASET_MAX_SOURCE_ROWS}" \
    --max-per-bucket "${DATASET_MAX_PER_BUCKET}" \
    --custom-output-len "${SD}" \
    --write-jsonl

  GENERATED_PATH="${DATASET_DIR}/sharegpt_sp${SP}.jsonl"

  if [ ! -f "${GENERATED_PATH}" ]; then
    echo "Dataset generation finished, but expected file is missing: ${GENERATED_PATH}" >&2
    exit 1
  fi

  if [ "${DATASET_PATH}" != "${GENERATED_PATH}" ]; then
    echo "Custom DATASET_PATH differs from generated path."
    echo "Copying:"
    echo "  from ${GENERATED_PATH}"
    echo "  to   ${DATASET_PATH}"
    mkdir -p "$(dirname "${DATASET_PATH}")"
    cp "${GENERATED_PATH}" "${DATASET_PATH}"
  fi

  if [ ! -f "${DATASET_PATH}" ]; then
    echo "Dataset still missing after generation/copy: ${DATASET_PATH}" >&2
    exit 1
  fi
fi

mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"

BENCH_CMD=(
  "${VLLM_BIN}" bench serve
  --backend vllm
  --host "${HOST}"
  --port "${PORT}"
  --endpoint "${ENDPOINT}"
  --model "${MODEL_ID}"
  --dataset-name custom
  --dataset-path "${DATASET_PATH}"
  --num-prompts "${NUM_PROMPTS}"
  --request-rate "${REQUEST_RATE}"
  --burstiness "${BURSTINESS}"
  --seed "${SEED}"
  --custom-output-len "${SD}"
  --ignore-eos
)

if [ "${SAVE_RESULT}" = "1" ]; then
  BENCH_CMD+=(--save-result --result-dir "${RESULT_DIR}")
fi

if [ "${SAVE_DETAILED}" = "1" ]; then
  BENCH_CMD+=(--save-detailed)
fi

if [ -n "${RESULT_FILENAME}" ]; then
  BENCH_CMD+=(--result-filename "${RESULT_FILENAME}")
fi

echo "=== running benchmark ==="
printf ' %q' "${BENCH_CMD[@]}"
echo

"${BENCH_CMD[@]}"
