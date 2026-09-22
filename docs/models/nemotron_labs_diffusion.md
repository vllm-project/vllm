# Nemotron Labs Diffusion

`NemotronLabsDiffusionModel` checkpoints support text generation with masked
block diffusion. Prompt prefill and completed-block KV refresh use causal
attention; denoising uses bidirectional attention within the current block.

```bash
vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
    --attention-backend TRITON_ATTN \
    --max-num-seqs 8 \
    --diffusion-config '{"temperature": 0.0, "confidence_threshold": 0.9}'
```

The canvas length defaults to the checkpoint's `block_size`. The default
`confidence_threshold` policy reveals all masked positions whose selected-token
probability is at least 0.9, and at least the most confident position each step.
Revealed positions remain fixed. The final permitted denoising step resolves any
remaining masks before the block is committed.

`DiffusionConfig.max_denoising_steps` limits iterations **per block** and defaults
to the canvas length. `SamplingParams.max_tokens` controls the response length.
Set request temperature to `0` for greedy decoding or `1` to use the engine's
`DiffusionConfig.temperature`. Confidence is measured before temperature scaling.
The `low_confidence` policy instead reveals a scheduled number of the most
confident positions; `leftmost` reveals that number from left to right.

The model uses vLLM's diffusion support in Model Runner V2. Triton attention is the default;
FlashAttention requires FA4. FlashInfer does not support the mixed
causal/bidirectional attention. This implementation covers
text-only masked diffusion; linear speculation and vision inputs are not included.

## Autoregressive inference

The same checkpoint also supports ordinary causal, token-by-token generation:

```bash
vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
    --hf-overrides '{"ar_mode": true}'
```

For Python, pass `hf_overrides={"ar_mode": True}` to `LLM`. The architecture
alias `hf_overrides={"architectures": ["NemotronLabsDiffusionForCausalLM"]}`
also selects AR mode, for compatibility with existing callers.

AR mode uses the same backbone and `diffusion_head` weights, with causal
attention and vLLM's standard scheduler, KV cache, and sampler. Sampling
parameters such as temperature, top-p, and top-k are set per request. Do not
pass `diffusion_config` in AR mode; denoising policies and thresholds do not
apply. The default, without either override, remains block diffusion.
