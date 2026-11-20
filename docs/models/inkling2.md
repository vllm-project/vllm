# Inkling 2

`Inkling2ForConditionalGeneration` serves Small (`inkling2`), and
`Inkling2LoopForConditionalGeneration` serves Mini (`inkling2_loop`, codename
Granitehollow). Mini reuses the core layer weights across passes, with independent
attention and convolution caches
for each visit. Its message controllers mix the preceding pass's residual
blocks into the next pass. The checkpoint's `loop_config` selects the core
bounds, pass count, and controller rank.

The checkpoint directory must include `config.json`, tokenizer files,
`hf_quant_config.json`, and the model safetensors. Expert quantization is selected
from the checkpoint's per-tensor metadata. Keep the original NVFP4/MXFP8 weights;
no conversion is required. The bundled DSpark draft additionally requires
`mtp.safetensors` and `mtp_config`.

Mini's `config.json` must declare `"model_type": "inkling2_loop"` and
`"architectures": ["Inkling2LoopForConditionalGeneration"]`. Earlier checkpoint
metadata may name `Inkling2ForConditionalGeneration`; update that field for Mini
when using this runtime. The weight files do not change.

## NVIDIA SM120 and SM121

On consumer Blackwell, including GB10, NVFP4 experts default to native block-scaled
Triton GEMMs. MXFP8 experts and relative attention use the portable Triton paths.
SM100-only tcgen05, TRT-LLM expert GEMMs, and NVLink multicast kernels are not
selected. Use a vLLM build targeting the device's SM version with the
repository's supported dependencies. Validation used Triton 3.8, also pinned
by the repository's CUDA test requirements. The runtime dispatch is shared
across SM120 and SM121; the reported hardware validation below used SM121.

Select `--moe-backend humming` to use Humming's native NVFP4 GEMMs on
SM120/SM121. This path keeps the model's FP4 activation quantization, E4M3
block scales, routing, and softcapped activation. Weights are repacked once
after loading; the original payload storage is released. MXFP8 experts keep
their portable implementation. This option requires the repository's Humming
dependency and does not change the attention projections or LM head.

The shared SM120/SM121 attention path handles both prefill and decode. Its
backend inherits FlashAttention's metadata builder, but executes the model's
custom relative-bias Triton kernels rather than FlashInfer or stock
FlashAttention kernels.

BF16 projections with 1–16 rows use a tiled tensor-core GEMV on SM120/SM121.
The same kernel computes FP32 logits for the unquantized target LM head,
including speculative verification. This replaces single-row head padding
while preserving vocabulary gathering and logit softcaps. Larger batches use
the existing GEMM path. The model's ordered K-group accumulations are unchanged.
Low-M kernel selection can change floating-point rounding in the projections.

Mini uses the loop executor and stream-batched message controllers, including
direct pair normalization, provided by the model implementation. On SM12x,
hidden-stream convolution and residual mixing use the portable kernels;
Hopper and SM10x retain their native fused kernels.

For throughput-oriented serving, combine `--moe-backend humming`,
`--kv-cache-dtype fp8_e4m3`, and `--max-num-batched-tokens 8192` with decode
CUDA graphs and the 16 GiB cache budget shown below. The same settings support
the bundled DSpark draft with BF16 draft KV.

For an initial single-GPU text run on a 64 GB GB10:

```bash
vllm serve /path/to/checkpoint \
    --served-model-name inkling-2-mini \
    --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
    --host 127.0.0.1 --port 30001 \
    --max-model-len 4096 --max-num-seqs 4 \
    --max-num-batched-tokens 256 --gpu-memory-utilization 0.75 \
    --limit-mm-per-prompt '{"image": 0}' --enforce-eager
```

The Inkling tokenizer and renderer are selected automatically. Send messages
through `/v1/chat/completions` using `inkling-2-mini` as the model name.

To enable images, omit the zero image limit. To select the bundled draft, add
`--speculative-config '{"method":"dspark","num_speculative_tokens":7}'`.
Draft feature indices refer to execution visits (48 for the three-pass Mini),
not the number of physical layers (24).

### Decode CUDA graphs and longer context

Mini also supports full decode CUDA graphs on GB10. The following configuration
captures batches of one through four requests, keeps prefill eager, and reserves
16 GiB for the KV cache to leave headroom for CPU work in unified memory:

```bash
vllm serve /path/to/checkpoint \
    --served-model-name inkling-2-mini \
    --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
    --host 127.0.0.1 --port 30001 \
    --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.70 --kv-cache-memory-bytes 17179869184 \
    --moe-backend humming --kv-cache-dtype fp8_e4m3 \
    --limit-mm-per-prompt '{"image": 1}' --no-enable-prefix-caching \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1,2,3,4],"prefill_cudagraph_capture_token_sizes":[]}'
```

Omit `--enforce-eager` when enabling graphs. With seven speculative tokens,
the four target graphs cover 8, 16, 24, and 32 tokens. `mode: 0` disables
`torch.compile` independently of CUDA graph capture.

Tensor parallelism, LoRA, and workload-specific performance require separate
validation. The context limit alone does not guarantee that every combination
of request lengths and concurrency fits in the cache.

### KV precision on SM120 and SM121

The default `--kv-cache-dtype auto` uses BF16 for this checkpoint. Explicit
`--kv-cache-dtype float16` and `--kv-cache-dtype fp8_e4m3` (also spelled `fp8`)
are available. FP16 KV requires speculative decoding disabled. FP8 target KV
also supports the bundled DSpark draft. The model still computes in
BF16; these options change attention-cache storage. The convolution-state
cache remains BF16.

FP8 uses E4M3 with unit scales for the model's clipped keys and normalized
values. The portable attention kernel converts cached values to the query
dtype before matrix multiplication. No checkpoint conversion or external KV
scale file is needed. Validate accuracy for the intended workload before
changing the serving default.

On SM120/SM121, DSpark defaults its own KV cache to BF16 independently of the
target cache. Its unnormalized values do not use the target's unit-scale FP8
storage convention. Explicit draft `kv_cache_dtype` accepts `auto` or
`bfloat16`; other draft cache types are rejected. For FP8 target KV with BF16
draft KV, use these settings in the graph-enabled command above:

```bash
--kv-cache-dtype fp8_e4m3 \
--speculative-config '{"method":"dspark","num_speculative_tokens":7}'
```

The draft cache, draft model, and larger verification graphs consume memory
in addition to the target cache. Recheck available system memory after graph
capture and under load when enabling speculation.

With a 131,072-token context limit and a 16 GiB cache budget, long concurrent
requests may queue. The scheduler's four-request limit does not imply that four
full-context requests fit in memory together. Check the reported cache capacity
for the selected precision, chunk size and speculative-decoding settings.

Increasing `--max-num-batched-tokens` also increases the hybrid cache planner's
reserve for attention and convolution state. FP8 does not halve the entire
cache allocation. Check the startup cache capacity and available system memory
through warmup and inference when choosing the chunk and cache budgets. On
GB10, GPU allocations share physical memory with the host.
