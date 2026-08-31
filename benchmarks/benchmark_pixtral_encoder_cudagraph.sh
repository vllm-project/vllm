#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-mistralai/Ministral-3-3B-Instruct-2512}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
PATCH_SIZE="${PATCH_SIZE:-14}"
NUM_REQUESTS="${NUM_REQUESTS:-50}"
NUM_WARMUPS="${NUM_WARMUPS:-10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-FLASH_ATTN}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-}"
RUN_ORDER="${RUN_ORDER:-off,on}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/pixtral_encoder_cg_benchmark_${RUN_ID}}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Compare Pixtral request latency with encoder CUDA Graphs off and on.

Usage:
  benchmarks/benchmark_pixtral_encoder_cudagraph.sh \
      [extra vllm serve arguments]

Common overrides are environment variables:
  MODEL, IMAGE_SIZE, PATCH_SIZE, NUM_REQUESTS, NUM_WARMUPS, MAX_MODEL_LEN,
  OUTPUT_DIR, MM_ENCODER_ATTN_BACKEND, HOST, PORT, RUN_ORDER, PYTHON_BIN,
  and VLLM_BIN.

Examples:
  NUM_REQUESTS=100 benchmarks/benchmark_pixtral_encoder_cudagraph.sh
  RUN_ORDER=on,off benchmarks/benchmark_pixtral_encoder_cudagraph.sh \
      --tensor-parallel-size 2

Run both orders when collecting final PR numbers. The script uses unique
synthetic images, a persistent localhost connection, one output token, and one
request at a time. Decoder CUDA Graph configuration is identical in both runs.
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
if ((PATCH_SIZE <= 0 || IMAGE_SIZE <= 0 || IMAGE_SIZE % PATCH_SIZE != 0)); then
    echo "IMAGE_SIZE must be a positive multiple of PATCH_SIZE." >&2
    exit 1
fi
if ((NUM_REQUESTS <= 0 || NUM_WARMUPS < 0)); then
    echo "NUM_REQUESTS must be positive and NUM_WARMUPS must be non-negative." >&2
    exit 1
fi
if [[ "${RUN_ORDER}" != "off,on" && "${RUN_ORDER}" != "on,off" ]]; then
    echo "RUN_ORDER must be either off,on or on,off." >&2
    exit 1
fi

PATCHES_PER_SIDE=$((IMAGE_SIZE / PATCH_SIZE))
TOKEN_BUDGET=$((PATCHES_PER_SIDE * PATCHES_PER_SIDE))
CG_OFF_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_mm_encoder":false}'
CG_ON_CONFIG="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_mm_encoder\":true,\"encoder_cudagraph_token_budgets\":[${TOKEN_BUDGET}],\"encoder_cudagraph_max_vision_items_per_batch\":1}"

mkdir -p "${OUTPUT_DIR}"

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
GPU_INFO="unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO="$(
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
            2>/dev/null | paste -sd ';' - || true
    )"
    GPU_INFO="${GPU_INFO:-unavailable}"
fi

if [[ -n "${VLLM_BIN}" ]]; then
    VLLM_CMD=("${VLLM_BIN}")
else
    VLLM_CMD=("${PYTHON_BIN}" -m vllm.entrypoints.cli.main)
fi

SERVER_PID=""
stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
    local server_log=$1
    local deadline=$((SECONDS + 900))
    until curl --silent --fail "http://${HOST}:${PORT}/health" >/dev/null; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "The vLLM server exited before becoming healthy." >&2
            tail -n 100 "${server_log}" >&2
            exit 1
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for the vLLM server." >&2
            tail -n 100 "${server_log}" >&2
            exit 1
        fi
        sleep 2
    done
}

run_requests() {
    local result_file=$1
    HOST="${HOST}" PORT="${PORT}" MODEL="${MODEL}" IMAGE_SIZE="${IMAGE_SIZE}" \
        NUM_REQUESTS="${NUM_REQUESTS}" NUM_WARMUPS="${NUM_WARMUPS}" \
        "${PYTHON_BIN}" - "${result_file}" <<'PY'
import base64
import http.client
from io import BytesIO
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from PIL import Image, ImageDraw


def percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def make_request(index: int, image_size: int, model: str) -> bytes:
    color = ((index * 17) % 256, (index * 29) % 256, (index * 43) % 256)
    image = Image.new("RGB", (image_size, image_size), color=color)
    draw = ImageDraw.Draw(image)
    block = max(16, image_size // 16)
    x = (index * 37) % (image_size - block + 1)
    y = (index * 53) % (image_size - block + 1)
    draw.rectangle((x, y, x + block - 1, y + block - 1), fill=color[::-1])
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()
    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
    }
    return json.dumps(request).encode()


host = os.environ["HOST"]
port = int(os.environ["PORT"])
model = os.environ["MODEL"]
image_size = int(os.environ["IMAGE_SIZE"])
num_requests = int(os.environ["NUM_REQUESTS"])
num_warmups = int(os.environ["NUM_WARMUPS"])
bodies = [
    make_request(index, image_size, model)
    for index in range(num_warmups + num_requests)
]

connection = http.client.HTTPConnection(host, port, timeout=600)
latencies = []
for index, body in enumerate(bodies):
    start = time.perf_counter()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    response_body = response.read()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if response.status != 200:
        raise RuntimeError(
            f"Request {index} failed with HTTP {response.status}: "
            f"{response_body.decode(errors='replace')}"
        )
    if index >= num_warmups:
        latencies.append(elapsed_ms)
connection.close()

latencies.sort()
result = {
    "num_requests": num_requests,
    "num_warmups": num_warmups,
    "image_size": image_size,
    "latencies_ms": latencies,
    "mean_ms": statistics.fmean(latencies),
    "median_ms": statistics.median(latencies),
    "p90_ms": percentile(latencies, 0.90),
    "p99_ms": percentile(latencies, 0.99),
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(
    f"mean={result['mean_ms']:.3f} ms, median={result['median_ms']:.3f} ms, "
    f"p90={result['p90_ms']:.3f} ms, p99={result['p99_ms']:.3f} ms"
)
PY
}

run_case() {
    local case_name=$1
    local compilation_config=$2
    shift 2
    local case_dir="${OUTPUT_DIR}/${case_name}"
    local server_log="${case_dir}/server.log"
    local result_file="${case_dir}/result.json"
    mkdir -p "${case_dir}"

    echo
    echo "Starting ${case_name}: ${MODEL}, ${IMAGE_SIZE}x${IMAGE_SIZE}"
    "${VLLM_CMD[@]}" serve "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --tokenizer-mode mistral \
        --config-format mistral \
        --load-format mistral \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs 1 \
        --limit-mm-per-prompt.image 1 \
        --mm-processor-cache-gb 0 \
        --mm-encoder-attn-backend "${MM_ENCODER_ATTN_BACKEND}" \
        --compilation-config "${compilation_config}" \
        "$@" >"${server_log}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${server_log}"

    if [[ "${case_name}" == "on" ]]; then
        if ! grep -q "Encoder CUDA graph capture complete" "${server_log}"; then
            echo "The server did not report encoder graph capture." >&2
            tail -n 100 "${server_log}" >&2
            exit 1
        fi
        grep "EncoderCudaGraphManager initialized\|Encoder CUDA graph capture complete" \
            "${server_log}"
    fi

    echo "Running ${NUM_WARMUPS} warmups and ${NUM_REQUESTS} measured requests..."
    run_requests "${result_file}"
    stop_server
}

IFS=',' read -r -a CASES <<<"${RUN_ORDER}"
for case_name in "${CASES[@]}"; do
    case "${case_name}" in
        off)
            run_case off "${CG_OFF_CONFIG}" "$@"
            ;;
        on)
            run_case on "${CG_ON_CONFIG}" "$@"
            ;;
    esac
done

MODEL="${MODEL}" IMAGE_SIZE="${IMAGE_SIZE}" PATCH_SIZE="${PATCH_SIZE}" \
    TOKEN_BUDGET="${TOKEN_BUDGET}" \
    RUN_ORDER="${RUN_ORDER}" GIT_COMMIT="${GIT_COMMIT}" GPU_INFO="${GPU_INFO}" \
    MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND}" \
    EXTRA_SERVE_ARGS="$*" "${PYTHON_BIN}" - \
    "${OUTPUT_DIR}/off/result.json" "${OUTPUT_DIR}/on/result.json" \
    "${OUTPUT_DIR}/comparison.md" <<'PY'
import json
import os
from pathlib import Path
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as result_file:
        return json.load(result_file)


def improvement(baseline: float, graph: float) -> str:
    return f"{(baseline - graph) / baseline * 100:.2f}%"


off = load(sys.argv[1])
on = load(sys.argv[2])
rows = [
    ("Mean", off["mean_ms"], on["mean_ms"]),
    ("Median", off["median_ms"], on["median_ms"]),
    ("P90", off["p90_ms"], on["p90_ms"]),
    ("P99", off["p99_ms"], on["p99_ms"]),
]

lines = [
    "# Pixtral encoder CUDA Graph benchmark",
    "",
    f"- Model: `{os.environ['MODEL']}`",
    f"- Commit: `{os.environ['GIT_COMMIT']}`",
    f"- GPU and driver: `{os.environ['GPU_INFO']}`",
    f"- Image: `{os.environ['IMAGE_SIZE']}x{os.environ['IMAGE_SIZE']}`",
    f"- Patch size: `{os.environ['PATCH_SIZE']}`",
    f"- Encoder attention backend: `{os.environ['MM_ENCODER_ATTN_BACKEND']}`",
    f"- Encoder graph token budget: `{os.environ['TOKEN_BUDGET']}`",
    f"- Run order: `{os.environ['RUN_ORDER']}`",
    f"- Extra server arguments: `{os.environ['EXTRA_SERVE_ARGS'] or 'none'}`",
    f"- Measured requests per arm: `{off['num_requests']}`",
    "- Decoder CUDA Graph mode: `FULL_DECODE_ONLY` in both runs",
    "- Output tokens per request: `1`",
    "",
    "| Request latency | CG off (ms) | CG on (ms) | Improvement |",
    "| --- | ---: | ---: | ---: |",
]
for name, baseline, graph in rows:
    lines.append(
        f"| {name} | {baseline:.3f} | {graph:.3f} | "
        f"{improvement(baseline, graph)} |"
    )

report = "\n".join(lines) + "\n"
Path(sys.argv[3]).write_text(report, encoding="utf-8")
print()
print(report)
print(f"Raw results and logs: {Path(sys.argv[3]).parent}")
PY
