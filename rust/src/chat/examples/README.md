# Chat Smoke Test

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

Run the Rust chat smoke test through the `vllm-chat` interface:

```bash
cargo run -p vllm-chat --example external_engine_chat_qwen -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1 \
  --prompt 'What is the capital of France? Answer with one word.'
```

The example now defaults to `Qwen/Qwen3-0.6B`. The current `vllm-chat`
request model stays text-only, but it now supports either plain string content
or OpenAI-style text blocks. Tool use, reasoning fields, and multimodal parts
are still out of scope. The example also sets
`chat_options.template_kwargs["enable_thinking"] = false` so Qwen3 runs in
non-thinking mode by default. It uses the Rust `tokenizers` library for the
tokenizer itself, plus standard Hugging Face config files to load the chat
template and EOS metadata.

IMPORTANT: Restart `vllm` each time you run the smoke test. The current headless
engine cannot safely handle frontend reconnects after the client shuts down.
