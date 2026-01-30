#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

GPU="6"
HOST="0.0.0.0"
PORT="8006"
CONDA_ENV="vllm-src"
MODEL_ID="microsoft/VibeVoice-ASR"
MODEL_DIR="/data2/mayufeng/models/VibeVoice-ASR"
SERVED_MODEL_NAME="vibevoice"
DTYPE="bfloat16"
MAX_MODEL_LEN="65536"
MAX_NUM_SEQS="64"
MAX_NUM_BATCHED_TOKENS="32768"
GPU_MEMORY_UTILIZATION="0.8"
ATTENTION_BACKEND="TRITON_ATTN"
ALLOWED_LOCAL_MEDIA_PATH="/home/mayufeng/projects"
CHAT_TEMPLATE="${SCRIPT_DIR}/chat_template.jinja"

ENFORCE_EAGER="1"
ENABLE_PREFIX_CACHING="0"
ENABLE_CHUNKED_PREFILL="1"
TRUST_REMOTE_CODE="1"
FFMPEG_MAX_CONCURRENCY="${VIBEVOICE_FFMPEG_MAX_CONCURRENCY:-64}"
PYTORCH_ALLOC_CONF_DEFAULT="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options] [-- <extra vLLM api_server args>]

Options:
  --gpu INT                     (default: ${GPU})
  --host HOST                   (default: ${HOST})
  --port INT                    (default: ${PORT})
  --conda-env NAME              (default: ${CONDA_ENV})
  --model-id HF_REPO            (default: ${MODEL_ID})
  --model-dir PATH              (default: ${MODEL_DIR})
  --served-model-name NAME      (default: ${SERVED_MODEL_NAME})
  --dtype DTYPE                 (default: ${DTYPE})
  --max-model-len INT           (default: ${MAX_MODEL_LEN})
  --max-num-seqs INT            (default: ${MAX_NUM_SEQS})
  --max-num-batched-tokens INT  (default: ${MAX_NUM_BATCHED_TOKENS})
  --gpu-memory-utilization F    (default: ${GPU_MEMORY_UTILIZATION})
  --attention-backend NAME      (default: ${ATTENTION_BACKEND})
  --chat-template PATH          (default: ${CHAT_TEMPLATE})
  --allowed-local-media-path P  (default: ${ALLOWED_LOCAL_MEDIA_PATH})
  --(no-)enforce-eager          (default: enforce eager)
  --(no-)enable-prefix-caching  (default: disabled)
  --(no-)enable-chunked-prefill (default: enabled)
  --(no-)trust-remote-code      (default: trust remote code)
  --ffmpeg-max-concurrency INT  (default: ${FFMPEG_MAX_CONCURRENCY})
  --pytorch-alloc-conf STR      (default: ${PYTORCH_ALLOC_CONF_DEFAULT})
  -h, --help                    Show help

Example:
  bash examples/online_serving/vibevoice_asr/start_vibevoice_asr_server.sh \\
    --gpu 6 --port 8006 --model-dir /data2/mayufeng/models/VibeVoice-ASR \\
    --allowed-local-media-path /home/mayufeng/projects
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --model-id) MODEL_ID="$2"; shift 2;;
    --model-dir) MODEL_DIR="$2"; shift 2;;
    --served-model-name) SERVED_MODEL_NAME="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2;;
    --chat-template) CHAT_TEMPLATE="$2"; shift 2;;
    --allowed-local-media-path) ALLOWED_LOCAL_MEDIA_PATH="$2"; shift 2;;
    --enforce-eager) ENFORCE_EAGER="1"; shift 1;;
    --no-enforce-eager) ENFORCE_EAGER="0"; shift 1;;
    --enable-prefix-caching) ENABLE_PREFIX_CACHING="1"; shift 1;;
    --no-enable-prefix-caching) ENABLE_PREFIX_CACHING="0"; shift 1;;
    --enable-chunked-prefill) ENABLE_CHUNKED_PREFILL="1"; shift 1;;
    --no-enable-chunked-prefill) ENABLE_CHUNKED_PREFILL="0"; shift 1;;
    --trust-remote-code) TRUST_REMOTE_CODE="1"; shift 1;;
    --no-trust-remote-code) TRUST_REMOTE_CODE="0"; shift 1;;
    --ffmpeg-max-concurrency) FFMPEG_MAX_CONCURRENCY="$2"; shift 2;;
    --pytorch-alloc-conf) PYTORCH_ALLOC_CONF_DEFAULT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    --) shift; EXTRA_ARGS+=("$@"); break;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VIBEVOICE_FFMPEG_MAX_CONCURRENCY="${FFMPEG_MAX_CONCURRENCY}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF_DEFAULT}"

echo "[start_server] Preparing model in: ${MODEL_DIR}"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${SCRIPT_DIR}/prepare_model.py" \
    --model-id "${MODEL_ID}" \
    --model-dir "${MODEL_DIR}"

echo "[start_server] Starting vLLM OpenAI server: http://${HOST}:${PORT}"
echo "[start_server] Served model name: ${SERVED_MODEL_NAME}"
echo "[start_server] Chat template: ${CHAT_TEMPLATE}"
echo "[start_server] Allowed local media path: ${ALLOWED_LOCAL_MEDIA_PATH}"
echo
echo "Smoke test (in another shell):"
echo "  conda run -n ${CONDA_ENV} python -m vllm_plugin.tests.test_api \\"
echo "    ${ALLOWED_LOCAL_MEDIA_PATH%/}/sample.wav --url http://localhost:${PORT}"
echo

VLLM_ARGS=(
  serve
  "${MODEL_DIR}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --limit-mm-per-prompt '{"audio": 1}'
  --attention-backend "${ATTENTION_BACKEND}"
  --chat-template-content-format openai
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
  --tensor-parallel-size 1
)

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  VLLM_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--no-enforce-eager)
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
else
  VLLM_ARGS+=(--no-enable-prefix-caching)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  VLLM_ARGS+=(--enable-chunked-prefill)
else
  VLLM_ARGS+=(--no-enable-chunked-prefill)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
else
  VLLM_ARGS+=(--no-trust-remote-code)
fi

exec conda run --no-capture-output -n "${CONDA_ENV}" \
  vllm "${VLLM_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
