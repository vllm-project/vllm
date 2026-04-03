# Gemma4 Fast Prefill

## Summary

Add `--kv-sharing-fast-prefill` support for Gemma 4 models, porting the YOCO (You Only Cache Once) fast prefill optimization from Gemma 3n. When enabled, the cross-decoder layers (KV-shared) skip prefill tokens and only process decode tokens, significantly reducing prefill latency and improving throughput under concurrent load.

**Changes:**

- Split Gemma4 decoder layers into `Gemma4SelfDecoderLayers` (non-KV-shared, layers 0..K-1) and `Gemma4CrossDecoderLayers` (KV-shared, layers K..N-1), each as a separate `@support_torch_compile` unit
- Add `fast_prefill_forward()` to `Gemma4Model` that runs the self-decoder on the full batch but the cross-decoder only on decode tokens (via `logits_indices_padded`)
- Include the batch_descriptor fix from #38834 to ensure correct CUDA graph dispatch for the reduced cross-decoder batch
- Extract `_run_decoder_layers()` helper to eliminate layer-loop duplication between decoders
- Normal (non-fast-prefill) forward path is unchanged and retains PP support

Based on #38826 (Gemma 4 architecture support).

## Test Plan

### GSM8K accuracy (Gemma4-E4B, 5-shot)

```bash
# FP=OFF (baseline)
lm_eval --model vllm --tasks gsm8k --num_fewshot 5 \
  --model_args pretrained=google/gemma-4-E4B-it,gpu_memory_utilization=0.9,max_model_len=4096,tensor_parallel_size=1,trust_remote_code=True,attention_backend=TRITON_ATTN,kv_sharing_fast_prefill=False \
  --batch_size auto --apply_chat_template --fewshot_as_multiturn

# FP=ON (this PR)
lm_eval --model vllm --tasks gsm8k --num_fewshot 5 \
  --model_args pretrained=google/gemma-4-E4B-it,gpu_memory_utilization=0.9,max_model_len=4096,tensor_parallel_size=1,trust_remote_code=True,attention_backend=TRITON_ATTN,kv_sharing_fast_prefill=True \
  --batch_size auto --apply_chat_template --fewshot_as_multiturn
```

### Serving benchmark

```bash
# Start server (without fast prefill)
vllm serve google/gemma-4-E4B-it \
  --port 8434 \
  --disable-log-stats \
  --no-enable-prefix-caching \
  --max-num-seqs 128 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --attention-backend TRITON_ATTN \
  --trust-remote-code

# Start server (with fast prefill)
vllm serve google/gemma-4-E4B-it \
  --port 8434 \
  --disable-log-stats \
  --no-enable-prefix-caching \
  --max-num-seqs 128 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --attention-backend TRITON_ATTN \
  --trust-remote-code \
  --kv-sharing-fast-prefill

# Run benchmark (after server is ready)
# concurrency=8
vllm bench serve \
  --backend vllm \
  --ignore-eos \
  --port 8434 \
  --model google/gemma-4-E4B-it \
  --dataset-name random \
  --max-concurrency 8 \
  --request-rate inf \
  --num-prompts 256 \
  --random-input-len 8192 \
  --random-output-len 150

# concurrency=32
vllm bench serve \
  --backend vllm \
  --ignore-eos \
  --port 8434 \
  --model google/gemma-4-E4B-it \
  --dataset-name random \
  --max-concurrency 32 \
  --request-rate inf \
  --num-prompts 256 \
  --random-input-len 8192 \
  --random-output-len 150
```

## Test Results

### GSM8K accuracy (Gemma4-E4B, 5-shot)

No accuracy regression:

|                   | strict-match | flexible-extract |
|-------------------|--------------|------------------|
| FP=OFF (baseline) | 0.1054       | 0.1751           |
| FP=ON (this PR)   | 0.1031       | 0.1850           |

### Serving performance (Gemma4-E4B, 1xB200, ISL=8192, OSL=150, n=256)

#### concurrency=8

| Metric          | NORMAL      | FAST_PREFILL | Delta       |
|-----------------|-------------|--------------|-------------|
| Throughput      | 4.22 req/s  | 5.06 req/s   | **+19.9%**  |
| Mean TTFT       | 570 ms      | 363 ms       | **-36.3%**  |
| Mean TPOT       | 8.90 ms     | 8.16 ms      | **-8.3%**   |

#### concurrency=32

| Metric          | NORMAL      | FAST_PREFILL | Delta       |
|-----------------|-------------|--------------|-------------|
| Throughput      | 6.53 req/s  | 9.07 req/s   | **+38.9%**  |
| Mean TTFT       | 942 ms      | 622 ms       | **-34.0%**  |
| Mean TPOT       | 26.43 ms    | 19.37 ms     | **-26.7%**  |
