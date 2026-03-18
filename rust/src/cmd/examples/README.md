# `vllm-rs` CLI Quick Start

Start Qwen3 with one managed `vllm-rs serve` command from the repo root:

```bash
HF_HUB_OFFLINE=1 \
VLLM_CPU_KVCACHE_SPACE=2 \
VLLM_HOST_IP=127.0.0.1 \
VLLM_LOOPBACK_IP=127.0.0.1 \
cargo run --bin vllm-rs -- serve \
  --model Qwen/Qwen3-0.6B \
  --python ../vllm/.venv/bin/python \
  -- \
  --max-model-len 512 \
  --dtype float16
```

This launches:

- a managed headless Python `vllm` engine
- the Rust OpenAI-compatible frontend on `127.0.0.1:8000`

You can then send OpenAI-style requests to the Rust frontend:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```

If you already started headless `vllm` yourself, use `frontend` instead:

```bash
cargo run --bin vllm-rs -- frontend \
  --handshake-address tcp://127.0.0.1:62100 \
  --model Qwen/Qwen3-0.6B
```
