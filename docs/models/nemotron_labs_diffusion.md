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
Use ordinary request temperatures: `0` is greedy, `0.7` samples at temperature
0.7, and `1` samples at temperature 1. Different temperatures can share a batch.
For example, pass `SamplingParams(temperature=0.7, top_p=0.9)` to `LLM.generate`.
Temperature scaling precedes top-k/top-p filtering and applies to returned
sampling logprobs. Confidence is measured using unscaled logits over the retained
candidates.

`DiffusionConfig.temperature`, when provided, sets the default for requests that
omit temperature; it no longer overrides explicit request temperatures.
`--override-generation-config '{"temperature": 0.7}'` sets the same standard
default and takes precedence over `DiffusionConfig.temperature`. Without either
setting, the checkpoint's ordinary generation defaults apply (temperature 1 if
unspecified). The previous `1` selector for an engine temperature is removed.
The `low_confidence` policy instead reveals a scheduled number of the most
confident positions; `leftmost` reveals that number from left to right.

The model uses vLLM's diffusion support in Model Runner V2. Triton attention is the default;
FlashAttention requires FA4. FlashInfer does not support the mixed
causal/bidirectional attention. This implementation covers
text-only generation; vision inputs are not included.

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

## Linear speculation

Use the diffusion model to draft a block, then verify it with the same model
under causal attention:

```bash
vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
    --max-num-seqs 8 \
    --diffusion-config '{"algorithm": "linear_spec", "canvas_length": 32}'
```

For Python, pass `diffusion_config={"algorithm": "linear_spec"}` to `LLM` and
`SamplingParams(temperature=0)` to `generate`. Server requests default to
zero temperature in this mode. Nonzero temperatures and sampling penalties
are rejected.

The algorithm follows SGLang's greedy `LinearSpec`: causal prefill predicts an
AR seed, a bidirectional pass drafts the remaining positions, and a causal
pass verifies them. It accepts the seed plus the longest consecutive prefix
where `draft[i] == ar[i-1]`. The first unaccepted AR prediction seeds the next
block. Rejected KV positions are rolled back and overwritten. Each block takes
two forward passes; this does not guarantee a speedup over ordinary AR.

Logprobs, when requested, come from the causal predictions that produced the
accepted tokens (including the carried seed). Mask token 100 is excluded from
both draft and verification sampling. Greedy outputs follow the AR verification
rule; floating-point differences between attention shapes can still change
close argmax decisions. Denoising thresholds and iteration limits do not apply
to linear speculation. Use the original diffusion architecture, without
`ar_mode` or the causal architecture alias.
