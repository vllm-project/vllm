# Server Smoke Test

Start a fresh headless `vllm` engine:

```bash
source ../vllm/.venv/bin/activate
HF_HUB_OFFLINE=1 \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_CPU_KVCACHE_SPACE=2 \
VLLM_HOST_IP=127.0.0.1 \
VLLM_LOOPBACK_IP=127.0.0.1 \
python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16
```

Run the Rust server smoke test:

```bash
cargo run -p vllm-server --example external_engine_openai_qwen -- \
  --handshake-address tcp://127.0.0.1:62100
```

The example starts the Rust OpenAI-compatible server on an ephemeral local port,
connects to it via the `async-openai` Rust client, lists models, and then checks
that a streamed chat completion yields the assistant role chunk, final-answer
content chunks, and a terminal finish chunk. This example intentionally uses
`async-openai`'s standard typed `create_stream` API instead of BYOT, so it does
not inspect the nonstandard `reasoning_content` field even though the Rust
server may emit it for reasoning-capable models such as Qwen3. For reasoning
behavior itself, use the `vllm-chat` smoke test or the `vllm-server`
route tests.

IMPORTANT: Restart `vllm` each time you run the smoke test. The current headless
engine cannot safely handle frontend reconnects after the client shuts down.
