# Adaptive Verification

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. At batch size 256 it is a much more delicate one. Draft tokens now compete with real tokens for the same compute, and every rejected token is compute wasted; with enough of them, throughput drops.

That matters because per-position acceptance decays fast. While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost. The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies.

Adaptive verification decides per step how much of the draft to verify instead. Every (request, position) draft slot is scored by its *survival probability*, the running product of that request's per-position acceptance estimates, and the highest-scoring slots are admitted until a global budget is spent. Slots compete across requests: position 5 of a confident request can outrank position 1 of a doubtful one, so one request keeps its full block while another could be trimmed after a token or two.

The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.

The practical effect is that one configuration holds up across the whole load range, which removes most of the need to tune `num_speculative_tokens` per deployment.

## Support

Adaptive verification is supported for model-based drafters (EAGLE, MTP, DFlash, DSpark and standalone draft models). Verification lengths are selected using the following sources:

- **Confidence head**: DSpark checkpoints that ship a trained confidence head use its predictions directly.
- **Online acceptance estimator**: every other drafter predicts acceptance from its draft logits with a small logistic model, recalibrated during serving against the target's verification results.
- **Length head**: AdaFlash DFlash checkpoints predict a verification length directly. They use the same variable-length verification backend, with lengths supplied by the trained head instead of the confidence-based budget policy.

## Usage

It is off by default. Enable it in the speculative config with `enable_adaptive_verification`. For example, with DSpark:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": true
  }'
```

Or with DFlash on MiMo-V2.5-Pro, whose drafter ships in the checkpoint's `dflash/` subdirectory:

```bash
hf download XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash --local-dir MiMo-V2.5-Pro-FP4-DFlash

vllm serve MiMo-V2.5-Pro-FP4-DFlash \
  --tensor-parallel-size 4 --trust-remote-code --max-model-len auto \
  --speculative-config '{
    "method": "dflash",
    "model": "MiMo-V2.5-Pro-FP4-DFlash/dflash",
    "num_speculative_tokens": 8,
    "enable_adaptive_verification": true
  }'
```

Set `enable_adaptive_verification: false` to verify the full block for every request.

### AdaFlash

The released AdaFlash Qwen3-8B drafter includes an adaptive length head:

```bash
vllm serve Qwen/Qwen3-8B \
  --attention-backend TRITON_ATTN \
  --speculative-config '{
    "method": "dflash",
    "model": "AdaFlash/Qwen3-8B-AdaFlash",
    "num_speculative_tokens": 15,
    "enable_adaptive_verification": true
  }'
```

The checkpoint declares `dflash_config.use_thresh_head_two_model=true` and
`dflash_config.thresh_head_direct_len=true`. vLLM loads the head with the draft
backbone and evaluates it on the complete block of normalized hidden states,
including the anchor. Its predicted ratio is multiplied by the block size,
rounded and clamped to `[1, block_size]`. The anchor accounts for one verification
position, so a block size of 16 requires 15 speculative tokens, and the head can
select zero through 15 drafts for each request.

This path reuses the adaptive verification batch compaction, attention metadata,
KV cache and rejection sampler. It does not run the confidence-based cost-model
policy or the online acceptance estimator. The CPU waits for the current head
lengths before selecting the next verification shape; stale lengths cannot bound
that batch safely. Scheduler capacities and the rejection sampler's logits budget
can further shorten the selected prefixes.

With `enable_adaptive_verification=false`, the head weights are still loaded, and
the drafter uses fixed-length DFlash verification. Other DFlash and DSpark
checkpoints retain their existing acceptance-based policy.

Speculative decoding statistics retain their existing scheduled-draft counting
semantics; the reported draft count is not the head's selected verification length.

On Ampere GPUs, FlashAttention 2 does not support the variable-length decode
CUDA graphs required by adaptive verification. Select a supported backend such
as `TRITON_ATTN`, as shown above.

## Requirements and limitations

- The attention backend must tolerate device-decided query lengths, since the CPU lengths only bound them from above. Backends that plan off the CPU lengths are excluded by the attention selector, and rejected at startup for models that hard-wire their backend.
- Full cudagraphs are required by this backend, so `--enforce-eager` is rejected at startup. Acceptance-based policies also profile step costs from the captured graphs; the direct-length AdaFlash policy does not need this cost profile.
- Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries) or pipeline parallelism (cost curves and confidences exist only on the last rank).
- Without a confidence head, `use_local_argmax_reduction` is rejected. The estimator needs the full draft logits, which local argmax reduction never materializes.

## Tuning the cost profile

Step costs are profiled against a synthetic KV context, 8192 tokens by default. Deployments serving much longer contexts may want to raise it so the profiled step reads a more realistic amount of cache (this matters a bit less for sparse attention models like DeepSeek-v4 since the cheap indexer is the main cost that scales with context length).

```bash
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=131072
```
