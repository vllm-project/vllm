# vllm-frontend-rs

`vllm-frontend-rs` is an early-stage Rust alternative frontend for the [vLLM](https://github.com/vllm-project/vllm) inference engine. The current goal is to rebuild the northbound serving layer in Rust while still talking to a headless Python vLLM engine via ZMQ over the existing engine boundary.

At the moment, the repository contains a minimal end-to-end path up to an OpenAI-compatible HTTP server plus a Rust CLI that can either connect to an existing headless Python engine or manage one for you.

## Architecture

The project is organized as a Cargo workspace with several crates, layered bottom-up:

```
┌─────────────────────────────────┐
│  vllm-cmd / vllm-rs             │  CLI entrypoint: external-engine
│                                 │  frontend mode + managed-engine
│                                 │  serve mode
├─────────────────────────────────┤
│  vllm-server                    │  OpenAI-compatible HTTP API (axum)
├─────────────────────────────────┤
│  vllm-chat                      │  Chat completions: template rendering,
│                                 │  structured assistant events,
│                                 │  reasoning & tool parsing
├─────────────────────────────────┤
│  vllm-text                      │  Tokenizer & incremental detokenizer
├─────────────────────────────────┤
│  vllm-llm                       │  Thin token-in/token-out facade over
│                                 │  the engine client
├─────────────────────────────────┤
│  vllm-engine-core-client        │  ZMQ transport + MessagePack protocol
│                                 │  for the headless vLLM engine
└─────────────────────────────────┘
```

## Quick Start

Install the CLI from the repo root first:

```bash
# from the local checkout
cargo install --path src/cmd --bin vllm-rs

# or directly from the git repo
cargo install --git https://github.com/inferact/vllm-frontend-rs --bin vllm-rs
```

### Managed Engine

Use `serve` when you want `vllm-rs` to start a headless Python `vllm` engine for you and then launch the Rust OpenAI-compatible frontend. The long-term goal of this path is to become a drop-in replacement for Python `vllm serve`.

### Minimal Startup

```bash
vllm-rs serve Qwen/Qwen3-0.6B --python xxx/vllm/.venv/bin/python
```

This starts the Rust OpenAI-compatible server on `127.0.0.1:8000` by default.

### Additional Frontend and Engine Args

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --python xxx/vllm/.venv/bin/python \
  --data-parallel-size 2 \
  --max-model-len 4096 \
  --tensor-parallel-size 2 \
  --more-engine-args ...
```

`vllm-rs serve` automatically keeps Rust frontend-owned args on the Rust side and forwards
everything else to the managed Python engine. If you want to force raw passthrough without any
Rust-side interpretation, you can still use an explicit `--`:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --python xxx/vllm/.venv/bin/python \
  --max-model-len 4096 \
  -- \
  --tensor-parallel-size 2 \
  --more-engine-args ...
```

Note that not all original `vllm serve` frontend args are supported yet. Unsupported frontend args
are recognized by `vllm-rs` and reported explicitly instead of being forwarded.

### External Engine

Use `frontend` when you already have a headless Python `vllm` engine running:

```bash
python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1
```

Then start the Rust frontend against that engine:

```bash
vllm-rs frontend \
  --handshake-address tcp://127.0.0.1:62100 \
  Qwen/Qwen3-0.6B
```

### Example Request

After either startup path, you can use any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```
