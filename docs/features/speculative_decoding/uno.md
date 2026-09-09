# Uno

[Uno](https://github.com/ifm-ai/uno) generates speculative candidates with the target model and a trained diffusion LoRA adapter. Model Runner V2 shares the target weights and KV cache with the drafter; no second base model is loaded.

Each draft pass has one base-model seed row followed by `K-1` noisy rows routed through the Uno adapter. Those `K` rows propose `K` candidates in parallel. Draft attention exposes the committed prefix and earlier rows within the new block, as in the [original Uno implementation](https://github.com/ifm-ai/uno/blob/main/nano_vllm_uno/layers/attention.py). Target verification uses base weights and vLLM's native rejection sampler. Draft KV is temporary: rejected candidates and noise rows do not become verified prefix-cache entries.

## Qwen3-8B example

Start with `Qwen/Qwen3-8B` in BF16 and the original `s-sahoo/uno-qwen3-8B` adapter. Download the adapter's `adapter` subdirectory before starting the server:

```python
from huggingface_hub import snapshot_download

snapshot = snapshot_download(
    "s-sahoo/uno-qwen3-8B",
    revision="8819e09ac901e7290d8d89d62c98b9f756c602fe",
    allow_patterns=["adapter/*"],
)
print(f"{snapshot}/adapter")
```

Use that directory as `/path/to/adapter` below:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3-8B \
    --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --dtype bfloat16 \
    --enable-lora --max-lora-rank 128 --max-loras 2 \
    --async-scheduling \
    --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}' \
    --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"uno","uno_lora_path":"/path/to/adapter","uno_mask_token_id":151669,"num_speculative_tokens":8}'
```

Normal sampling settings, including greedy decoding, temperature, top-k and top-p, apply to the target distribution. For the Qwen3 non-thinking profile, pass `chat_template_kwargs={"enable_thinking": false}` in chat requests.

## Configuration and scope

- `uno_lora_path` is required and points to a PEFT adapter. Use an adapter trained for the exact base model.
- `num_speculative_tokens` is a fixed positive `K`. `K=1` exercises the seed-only path. `parallel_drafting` is enabled automatically.
- `uno_mask_token_id` is the exclusive upper bound of the uniform noise range `[1, uno_mask_token_id)`, not a fixed token inserted into every noisy row. Match the training configuration; the default is the target vocabulary size.
- `uno_noise_seed` controls deterministic input noise for a fixed request seed, batch row order and draft step. The released hash includes the flat batch-row term, so noise is not invariant to batch reordering. Request sampling seeds retain their normal meaning.
- `max_loras` must be at least `2`; Uno reserves one slot for its shared adapter while native graph setup may use another slot.
- `max_num_batched_tokens` must be at least `max_num_seqs * K` to fit the draft queries and native LoRA metadata buffers.

The initial scope is one NVIDIA GPU, text-only decoder models with one homogeneous full-attention KV group, and FlashAttention. Model Runner V2 and async scheduling are required. Request-specific LoRA adapters, tensor/pipeline/data/context parallelism, sliding or hybrid attention, dynamic speculation depth, adaptive verification, KV transfer/offloading and dual batch overlap are unsupported.

Prefix caching and chunked prefill use the native scheduler. Draft sampling retains raw logits by persistent request slot for native probabilistic rejection sampling. Uno uses native full CUDA graphs for drafting when the target graph mode and attention backend support uniform decode capture; `--enforce-eager` disables graphs.

The native `VLLM_LORA_ENABLE_DUAL_STREAM=1` option can overlap base and LoRA linear work. Measure it on the intended hardware and workload. Uno's adapter routing caches shape-dependent native Punica metadata and restores the existing buffers before each pass; a cache miss for a new or evicted shape or adapter-slot layout uses native metadata preparation, which can synchronize with the host. The bounded cache avoids that synchronization for resident shapes.

At the context boundary, unused draft rows have clamped positions and no KV writes. Native scheduling and verification limit the usable candidates to the remaining context.
