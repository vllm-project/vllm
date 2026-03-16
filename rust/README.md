# vllm-frontend-rs

`vllm-frontend-rs` is an early-stage Rust alternative frontend for the [vLLM](https://github.com/vllm-project/vllm) inference engine. The current goal is to rebuild the northbound serving layer in Rust while still talking to a headless Python vLLM engine via ZMQ over the existing engine boundary.

At the moment, the repository contains a minimal end-to-end path up to an OpenAI-compatible HTTP server.

## Architecture

The project is organized as a Cargo workspace with 4 crates, layered bottom-up:

```
┌─────────────────────────────────┐
│  vllm-openai-server             │  OpenAI-compatible HTTP API (axum)
├─────────────────────────────────┤
│  vllm-chat                      │  Chat interface: message rendering,
│                                 │  tokenization, streaming chat events
├─────────────────────────────────┤
│  vllm-llm                       │  Thin token-in/token-out facade over the
│                                 │  engine client
├─────────────────────────────────┤
│  vllm-engine-core-client        │  ZMQ transport + MessagePack protocol
│                                 │  for the headless vLLM engine
└─────────────────────────────────┘
```

## Quick Start

```bash
python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1
```

Start the Rust OpenAI-compatible server:

```bash
cargo run -p vllm-openai-server -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --model Qwen/Qwen3-0.6B
```

The server listens on `127.0.0.1:8000` by default. You can then use any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```
