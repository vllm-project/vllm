#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-mistralai/Ministral-3-3B-Instruct-2512}"
IMAGE_URL="${IMAGE_URL:-}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
ENCODER_TOKEN_BUDGET="${ENCODER_TOKEN_BUDGET:-4096}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-FLASH_ATTN}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/pixtral_encoder_cg_profile_${RUN_ID}}"
PROFILE_DIR="${OUTPUT_DIR}/trace"
SERVER_LOG="${OUTPUT_DIR}/server.log"
RESPONSE_FILE="${OUTPUT_DIR}/response.json"
REQUEST_FILE="${OUTPUT_DIR}/request.json"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Profile one Pixtral encoder CUDA Graph replay through the OpenAI server.

Usage:
  benchmarks/profile_pixtral_encoder_cudagraph.sh [extra vllm serve arguments]

Common overrides are environment variables:
  MODEL, IMAGE_URL, IMAGE_SIZE, ENCODER_TOKEN_BUDGET, HOST, PORT,
  MAX_MODEL_LEN, OUTPUT_DIR, MM_ENCODER_ATTN_BACKEND, PYTHON_BIN, and VLLM_BIN.

By default, the script generates a 448x448 image that fits the 4096-token
encoder graph for both 14- and 16-pixel patch sizes. Set IMAGE_URL to profile a
specific image, and increase ENCODER_TOKEN_BUDGET if that image needs it.

Example for two GPUs:
  benchmarks/profile_pixtral_encoder_cudagraph.sh --tensor-parallel-size 2

The script leaves the trace, server log, request, and response in OUTPUT_DIR.
Open the *.pt.trace.json.gz file at https://ui.perfetto.dev/ and search for
"encoder_cudagraph: replay". The nested cudaGraphLaunch is the encoder replay.
EOF
    exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ -n "${VLLM_BIN}" && ! -x "${VLLM_BIN}" ]]; then
    echo "vLLM executable not found: ${VLLM_BIN}" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required" >&2
    exit 1
fi
if ((IMAGE_SIZE <= 0 || ENCODER_TOKEN_BUDGET <= 0)); then
    echo "IMAGE_SIZE and ENCODER_TOKEN_BUDGET must be positive." >&2
    exit 1
fi

mkdir -p "${PROFILE_DIR}"

if [[ -n "${VLLM_BIN}" ]]; then
    VLLM_CMD=("${VLLM_BIN}")
else
    VLLM_CMD=("${PYTHON_BIN}" -m vllm.entrypoints.cli.main)
fi

MODEL="${MODEL}" IMAGE_URL="${IMAGE_URL}" IMAGE_SIZE="${IMAGE_SIZE}" \
    "${PYTHON_BIN}" - "${REQUEST_FILE}" <<'PY'
import base64
from io import BytesIO
import json
import os
import sys

from PIL import Image, ImageDraw


image_url = os.environ["IMAGE_URL"]
if not image_url:
    image_size = int(os.environ["IMAGE_SIZE"])
    image = Image.new("RGB", (image_size, image_size), color=(32, 96, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (image_size // 4, image_size // 4, image_size // 2, image_size // 2),
        fill=(224, 128, 48),
    )
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()

request = {
    "model": os.environ["MODEL"],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ],
    "max_tokens": 1,
    "temperature": 0,
}
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(request, output)
PY

COMPILATION_CONFIG="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_mm_encoder\":true,\"encoder_cudagraph_token_budgets\":[${ENCODER_TOKEN_BUDGET}],\"encoder_cudagraph_max_vision_items_per_batch\":1}"
PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting ${MODEL}; logs: ${SERVER_LOG}"
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 "${VLLM_CMD[@]}" serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 1 \
    --limit-mm-per-prompt.image 1 \
    --mm-encoder-attn-backend "${MM_ENCODER_ATTN_BACKEND}" \
    --compilation-config "${COMPILATION_CONFIG}" \
    --profiler-config "${PROFILER_CONFIG}" \
    "$@" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 900))
until curl --silent --fail "http://${HOST}:${PORT}/health" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "The vLLM server exited before becoming healthy." >&2
        tail -n 100 "${SERVER_LOG}" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Timed out waiting for the vLLM server." >&2
        tail -n 100 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 2
done

if ! grep -q "Encoder CUDA graph capture complete" "${SERVER_LOG}"; then
    echo "The server is healthy, but encoder graph capture was not logged." >&2
    tail -n 100 "${SERVER_LOG}" >&2
    exit 1
fi

grep "EncoderCudaGraphManager initialized\|Encoder CUDA graph capture complete" \
    "${SERVER_LOG}"

echo "Profiling one image request..."
curl --silent --show-error --fail --max-time 60 \
    -X POST "http://${HOST}:${PORT}/start_profile" >/dev/null
curl --silent --show-error --fail --max-time 600 \
    -H 'Content-Type: application/json' \
    --data-binary "@${REQUEST_FILE}" \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${RESPONSE_FILE}"
curl --silent --show-error --fail --max-time 600 \
    -X POST "http://${HOST}:${PORT}/stop_profile" >/dev/null

"${PYTHON_BIN}" - "${PROFILE_DIR}" <<'PY'
import gzip
from pathlib import Path
import sys

profile_dir = Path(sys.argv[1])
traces = sorted(
    path
    for path in profile_dir.rglob("*")
    if path.name.endswith((".pt.trace.json", ".pt.trace.json.gz"))
)
if not traces:
    raise SystemExit(f"No PyTorch profiler trace found in {profile_dir}")

launches = 0
replay_scopes = 0
for trace in traces:
    opener = gzip.open if trace.suffix == ".gz" else open
    with opener(trace, "rb") as trace_file:
        contents = trace_file.read().lower()
    launches += contents.count(b"cudagraphlaunch")
    replay_scopes += contents.count(b"encoder_cudagraph: replay")

print(f"Trace files: {len(traces)}")
print(f"encoder_cudagraph: replay scopes: {replay_scopes}")
print(f"cudaGraphLaunch events: {launches}")
if replay_scopes == 0 or launches == 0:
    raise SystemExit(
        "Encoder CUDA Graph replay was not found in the trace; the image may "
        "exceed ENCODER_TOKEN_BUDGET"
    )
for trace in traces:
    print(f"  {trace}")
PY

echo
echo "Success: encoder CUDA Graph replay is present in the trace."
echo "Open the trace at https://ui.perfetto.dev/ and search for:"
echo "  encoder_cudagraph: replay"
echo "The cudaGraphLaunch nested inside that scope is the multimodal encoder."
echo "Artifacts: ${OUTPUT_DIR}"
