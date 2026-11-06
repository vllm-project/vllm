# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## Offline Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Qwen3.5 GDN ReplaySSM

GDN ReplaySSM stores recent recurrent inputs and reconstructs state from checkpoints.
This optional FlashInfer backend is disabled by default. Enable it for Qwen3.5 with:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 SGLANG_GDN_WY_STRIDED_QKV=1 \
VLLM_GDN_DECODE_KERNEL=flashinfer_replayssm \
vllm serve Qwen/Qwen3.5-0.8B \
    --dtype bfloat16 --mamba-cache-dtype bfloat16 \
    --mamba-ssm-cache-dtype bfloat16 --use-replayssm \
    --replayssm-buffer-len 16 --no-enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

This backend supports standard decoding and one to seven speculative tokens.
Omit `--speculative-config` for standard decoding. The default path uses BF16 state
and 128-element key and value heads. Pipeline parallelism, microbatching, and KV
connectors are unsupported. Ragged scheduled rows are supported. Adaptive
verification is unsupported because it changes row lengths on the GPU after the
CPU metadata builder determines the output indices.
Set `enable_adaptive_verification` to `false` in the speculative configuration.
Prefix caching requires `--mamba-cache-mode align`.

BF16 standard decoding requires `gated_delta_rule_stp_ucache_flush`. Speculative
decoding requires `gated_delta_rule_mtp_ucache_flush`, available in FlashInfer
0.7.2.post1. Prefix caching also requires `gdn_prefix_materialize`, available in
FlashInfer 0.7.3. The STP API is not included in either release, so BF16 standard
decoding requires a compatible provider build. This integration does not install
missing kernels or change FlashInfer.

FlashInfer 0.7.3 specializes the BF16 speculative decoder for batch and pool sizes.
Avoiding those recompilations requires a patched provider that declares
`SUPPORTS_DYNAMIC_STRIDED_SHAPES = True`. With that provider, startup also warms the
reachable input layouts and launch tactics. The extra warmup stays disabled for
providers without this capability. The prefix materializer has separate
pool-size specialization, which this decoder fix does not remove.

Set `SGLANG_GDN_WY_STRIDED_QKV=1` before process startup. FlashInfer reads this
setting when its kernel module is imported; setting it later is insufficient.

For standard or speculative decoding on SM100-class GPUs, set
`--mamba-ssm-cache-dtype float16` to select FlashInfer 0.7.3's direct UMMA API.
Keep model activations and convolution state in BF16. This path supports at most
four value heads per key head. It uses a 32-slot ring for both decoding modes.
The state and rings retain the existing paged allocation. Mamba cache stochastic
rounding is unsupported.

For FP16 prefix caching, use `--mamba-cache-mode align` and enable prefix caching.
Set `GDN_UCACHE_STATE_DTYPE=fp16` before process startup in each worker environment.
FlashInfer selects the materializer's state dtype during import. Changing the
environment after import does not change that dtype. BF16 prefix caching requires
the default BF16 materializer, so use separate processes for different state dtypes.

## Notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
- When the MTP head shares the target's unquantized lm_head (most MTP models),
  `"draft_lm_head_quantization": "fp8"` or `"nvfp4"` in `--speculative-config`
  gives the drafter a quantized copy of that head (Model Runner V2). The target
  still verifies with its own head, so outputs are unchanged; only the
  acceptance length can move. It helps most on bandwidth-bound GPUs with a
  large vocabulary. The copy costs about one byte (FP8) or 0.56 bytes (NVFP4)
  per lm_head weight, e.g. 0.6 or 0.35 GiB for a 248k x 2560 head; it is
  counted in the model weights, so with `--kv-cache-memory-bytes` leave room
  for it.
