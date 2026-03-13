# Chat Smoke Test

Start a fresh headless `vllm` engine:

```bash
source ../vllm/.venv/bin/activate
HF_HUB_OFFLINE=1 \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_CPU_KVCACHE_SPACE=2 \
VLLM_HOST_IP=127.0.0.1 \
VLLM_LOOPBACK_IP=127.0.0.1 \
python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen1.5-0.5B-Chat \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16
```

Run the Rust chat smoke test through the `vllm-chat` interface:

```bash
cargo run -p vllm-chat --example external_engine_chat_qwen -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1 \
  --prompt 'What is the capital of France? Answer with one word.'
```

The example now defaults to `Qwen/Qwen1.5-0.5B-Chat`, which is smaller and still
uses a string-style chat template that works with the current text-only chat
request model. The example now relies entirely on the crates.io
`llm-tokenizer` package, imported in Rust as `smg_tokenizer`, to download or
reuse the tokenizer and to auto-discover the chat template from model metadata
or adjacent template files.

IMPORTANT: Restart `vllm` each time you run the smoke test. The current headless
engine cannot safely handle frontend reconnects after the client shuts down.
