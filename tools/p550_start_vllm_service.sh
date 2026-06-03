#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

cd "${VLLM_P550_HOME:-$HOME/vllm}"
if [ ! -d .venv-p550 ]; then
    echo "Missing .venv-p550 under $(pwd). Build the P550 vLLM environment first." >&2
    exit 1
fi
. .venv-p550/bin/activate

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cpu}"
export VLLM_RVV_VLEN="${VLLM_RVV_VLEN:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-nobind}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
if [ -n "${VLLM_CPU_KVCACHE_SPACE:-}" ]; then
    export VLLM_CPU_KVCACHE_SPACE
fi

host="${HOST:-0.0.0.0}"
port="${PORT:-8000}"
model="${VLLM_P550_MODEL:-$PWD/.p550_models/tiny-llama}"
served_model="${VLLM_P550_SERVED_MODEL_NAME:-p550-tiny}"
kv_cache_bytes="${VLLM_P550_KV_CACHE_BYTES:-536870912}"

if [ ! -d "$model" ]; then
    echo "Local tiny model not found at $model; generating it now..." >&2
    python tools/p550_make_tiny_llama.py --output-dir "$model"
fi

service_py="$(mktemp /tmp/p550_vllm_http.XXXXXX.py)"
trap 'rm -f "$service_py"' EXIT

cat >"$service_py" <<'PY'
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

HOST = os.environ.get("P550_SERVICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("P550_SERVICE_PORT", "8000"))
MODEL = os.environ["P550_SERVICE_MODEL"]
SERVED_MODEL = os.environ.get("P550_SERVICE_MODEL_NAME", "p550-tiny")
KV_CACHE_BYTES = int(os.environ.get("P550_SERVICE_KV_CACHE_BYTES", "536870912"))

generate_lock = threading.Lock()
llm: LLM | None = None
tokenizer: Any | None = None


def json_response(
    handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(data)


def render_prompt(messages: list[dict[str, Any]]) -> str:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        normalized.append({"role": role, "content": str(content)})

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True
        )

    parts = [item["content"] for item in normalized]
    return "\n".join(parts) or "Hello from P550"


class Handler(BaseHTTPRequestHandler):
    server_version = "p550-vllm-minimal/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(
            f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} {fmt % args}",
            flush=True,
        )

    def do_GET(self) -> None:
        if self.path == "/health":
            json_response(self, 200, {"status": "ok", "model": SERVED_MODEL})
            return
        if self.path == "/v1/models":
            json_response(
                self,
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": SERVED_MODEL, "object": "model", "owned_by": "p550"}
                    ],
                },
            )
            return
        json_response(self, 404, {"error": "not found"})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            json_response(self, 404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages") or []
            if not isinstance(messages, list) or not messages:
                json_response(self, 400, {"error": "messages must be non-empty"})
                return
            max_tokens = int(body.get("max_tokens", 8))
            temperature = float(body.get("temperature", 0.0))
            prompt = render_prompt(messages)

            print(
                f"chat request: prompt={prompt!r} max_tokens={max_tokens}",
                flush=True,
            )
            assert llm is not None
            with generate_lock:
                outputs = llm.generate(
                    [prompt],
                    SamplingParams(max_tokens=max_tokens, temperature=temperature),
                    use_tqdm=False,
                )
            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            now = int(time.time())
            json_response(
                self,
                200,
                {
                    "id": f"chatcmpl-p550-{now}",
                    "object": "chat.completion",
                    "created": now,
                    "model": body.get("model", SERVED_MODEL),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "length",
                        }
                    ],
                },
            )
        except Exception as exc:
            json_response(self, 500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    global llm, tokenizer
    print(f"Loading vLLM model: {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        dtype="float",
        max_model_len=64,
        max_num_seqs=1,
        enforce_eager=True,
        disable_log_stats=True,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
    )
    print(f"Serving minimal vLLM chat API on http://{HOST}:{PORT}", flush=True)
    print("Endpoints: /health, /v1/models, /v1/chat/completions", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
PY

export P550_SERVICE_HOST="$host"
export P550_SERVICE_PORT="$port"
export P550_SERVICE_MODEL="$model"
export P550_SERVICE_MODEL_NAME="$served_model"
export P550_SERVICE_KV_CACHE_BYTES="$kv_cache_bytes"

echo "Starting minimal vLLM HTTP service. Stop with Ctrl-C."
echo "Health: http://127.0.0.1:${port}/health"
echo "Chat:   tools/p550_chat_test.sh 'Hello from P550 service'"
exec python "$service_py"
