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

The initial scope is text-only decoder models with one homogeneous full-attention KV group, on NVIDIA GPUs with the `FLASH_ATTN` or `TRITON_ATTN` attention backend, and on AMD ROCm GPUs with `TRITON_ATTN`: replace the `--attention-config` above with `--attention-config '{"backend":"TRITON_ATTN"}'`. On ROCm, Uno has been tested on the Radeon AI PRO R9700 (gfx1201); `TRITON_ATTN` on NVIDIA has been tested on one RTX 3090 (sm_86). Every other attention backend, FlashInfer included, is refused at startup; on ROCm the backend must be set explicitly. Model Runner V2 and async scheduling are required. Tensor parallelism is supported and the draft uses the target's `tensor_parallel_size`. User LoRA adapters are not supported yet: `--enable-lora` serves only Uno's own adapter. A request carrying a LoRA adapter is refused with a validation error before it is queued, and runtime LoRA loading is refused at startup. Pipeline/data/context parallelism, sliding or hybrid attention, dynamic speculation depth, adaptive verification, KV transfer/offloading, `--kv-sharing-fast-prefill` and dual batch overlap are unsupported.

Prefix caching and chunked prefill use the native scheduler. Draft sampling retains raw logits by persistent request slot for native probabilistic rejection sampling. Uno uses native full CUDA graphs for drafting when the target graph mode and attention backend support uniform decode capture; a draft decode graph exists only when some `cudagraph_capture_sizes` entry is at most `max_num_seqs * K`, so small products (for example `K=1` with few sequences) draft eagerly. `--enforce-eager` disables graphs.

The native `VLLM_LORA_ENABLE_DUAL_STREAM=1` option can overlap base and LoRA linear work. Measure it on the intended hardware and workload. Uno routes its draft rows through its adapter with the same LoRA mapping call the runner uses for target steps, so each draft pass and the following target step rebuild the native LoRA metadata, which waits on the host once per pass.

At the context boundary, unused draft rows have clamped positions and no KV writes. Native scheduling and verification limit the usable candidates to the remaining context.
