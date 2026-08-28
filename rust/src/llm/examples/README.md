# LLM Smoke Test

Start headless `vllm`:

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

Run the Rust smoke test through the `vllm-llm` generate interface:

```bash
cargo run -p vllm-llm --example external_engine_smoke -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

IMPORTANT: You must restart `vllm` each time you run the smoke test, as the vLLM engine cannot manage frontend closures and subsequent reconnects. In other words, do not reuse existing `vllm` instances, if any.

## Embedding smoke test

Start a fresh headless pooling engine:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vllm.entrypoints.cli.main serve \
  Qwen/Qwen3-Embedding-0.6B \
  --headless \
  --runner pooling \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16 \
  --enforce-eager \
  --kernel-config.enable_flashinfer_autotune=False \
  --gpu-memory-utilization 0.1
```

Run one token-level embedding request through the Rust `Llm::encode()` interface:

```bash
cargo run -p vllm-llm --example external_engine_embedding -- \
  --handshake-address tcp://127.0.0.1:62100
```
