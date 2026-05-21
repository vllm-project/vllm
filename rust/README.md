# vllm-frontend-rs

`vllm-frontend-rs` is an early-stage Rust drop-in alternative frontend for the [vLLM](https://github.com/vllm-project/vllm) inference engine. The current goal is to rebuild the northbound serving layer in Rust while still talking to the core Python vLLM engine process(es) via ZMQ over the existing engine boundary.

## Architecture

The project is organized as a Cargo workspace with several crates, layered bottom-up:

```
┌─────────────────────────────────┐
│  vllm-cmd / vllm-rs             │  CLI entrypoint:
│                                 │  Python vLLM frontend subprocess
│                                 │  Rust managed-engine serve mode
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

### Python Integration

`vllm-rs` integrates into Python `vllm` as a Rust frontend subprocess. In that setup,
Python owns process startup and launches the Rust API server as a Python-supervised worker, while
passing the inherited listening socket and transport addresses into `vllm-rs`.

For example:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B
```

As a tightly-coupled sub-component of vLLM, it is expected that the code in this repo will be relocated
to live in a `rust` subdirectory of the vLLM repo. For staging purposes however, we're currently including
it as a submodule.

### External Engine

`vllm-rs serve` can be run standalone with `--data-parallel-size-local 0` when the Python engines
are started elsewhere and this node should run only the Rust frontend. The frontend still uses
the global `--data-parallel-size` to determine how many engines it expects to join the shared handshake.

```bash
vllm serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1
```

Then start the Rust frontend-only server:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 0
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
