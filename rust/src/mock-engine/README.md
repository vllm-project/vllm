# vLLM Mock Engine

`vllm-mock-engine` is a small engine-side process for frontend stress testing. It
joins a frontend-owned startup handshake, reports a large ready response, treats
prefill as instant, and emits random decode tokens until each request reaches
its `max_tokens`.

The frontend must own the handshake socket. Start the frontend first, then start
the mock engine with the same handshake address.

## Start the mock engine

```bash
cargo run -p vllm-mock-engine -- \
  --handshake-address tcp://127.0.0.1:29550 \
  --engine-count 1 \
  --output-token-chunk-size 1 \
  --vocab-size 32000 \
  --seed 0 \
  --log-requests
```

Useful knobs:

- `--engine-count` must match the frontend's expected data-parallel engine
  count.
- `--output-token-chunk-size` controls how many token IDs appear in one
  `EngineCoreOutput`; values greater than 1 are useful for MTP/spec-decode
  shaped frontend tests.
- `--vocab-size` should stay within the tokenizer vocabulary of the model used
  by the frontend.

Stop it with Ctrl-C.

## Rust Frontend

Terminal 1:

```bash
cargo run --bin vllm-rs -- serve Qwen/Qwen3-0.6B \
  --data-parallel-size 1 \
  --data-parallel-size-local 0 \
  --handshake-port 29550
```

Terminal 2:

```bash
cargo run -p vllm-mock-engine -- \
  --handshake-address tcp://127.0.0.1:29550
```

For multiple mock engines, set both sides to the same count:

```bash
cargo run --bin vllm-rs -- serve Qwen/Qwen3-0.6B \
  --data-parallel-size 4 \
  --data-parallel-size-local 0 \
  --handshake-port 29550

cargo run -p vllm-mock-engine -- \
  --handshake-address tcp://127.0.0.1:29550 \
  --engine-count 4
```

## Python Frontend

Use `vllm serve` with `--data-parallel-size-local 0` so the Python process runs
as a frontend/API server and waits for external engines on
`--data-parallel-rpc-port`.

Terminal 1:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 29550 \
  --data-parallel-size 1 \
  --data-parallel-size-local 0
```

Terminal 2:

```bash
cargo run -p vllm-mock-engine -- \
  --handshake-address tcp://127.0.0.1:29550
```

## Smoke Request

After either frontend is ready:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16,
    "stream": true
  }'
```

Always pass `max_tokens`; the mock engine stops by length.
