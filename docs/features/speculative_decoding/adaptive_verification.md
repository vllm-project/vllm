# Adaptive Verification

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. At batch size 256 it is a much more delicate one. Draft tokens now compete with real tokens for the same compute, and every rejected token is compute wasted; with enough of them, throughput drops.

That matters because per-position acceptance decays fast. While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost. The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies.

Adaptive verification decides per step how much of the draft to verify instead. Every (request, position) draft slot is scored by its *survival probability*, the running product of that request's per-position acceptance estimates, and the highest-scoring slots are admitted until a global budget is spent. Slots compete across requests: position 5 of a confident request can outrank position 1 of a doubtful one, so one request keeps its full block while another could be trimmed after a token or two.

The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.

The practical effect is that one configuration holds up across the whole load range, which removes most of the need to tune `num_speculative_tokens` per deployment.

## Support

Adaptive verification is supported for model-based drafters (EAGLE, MTP, DFlash, DSpark, etc.) and for [suffix decoding](suffix.md#adaptive-verification). The per-position acceptance estimates come from one of three sources:

- **Confidence head**: DSpark checkpoints that ship a trained confidence head use its predictions directly.
- **Online acceptance estimator**: every other model-based drafter predicts acceptance from its draft logits with a small logistic model, recalibrated during serving against the target's verification results.
- **Suffix acceptance statistics**: suffix decoding has no draft logits, so it estimates acceptance from the observed acceptance rate of past drafts with the same source (own history or corpus), match length and position.

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

## Requirements and limitations

- The attention backend must tolerate device-decided query lengths, since the CPU lengths only bound them from above. Backends that plan off the CPU lengths are excluded by the attention selector, and rejected at startup for models that hard-wire their backend.
- Full cudagraphs are required: step costs are profiled from captured graphs, so `--enforce-eager` is rejected at startup.
- Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries) or pipeline parallelism (cost curves and confidences exist only on the last rank).
- Without a confidence head, `use_local_argmax_reduction` is rejected. The estimator needs the full draft logits, which local argmax reduction never materializes.

## Tuning the cost profile

Step costs are profiled against a synthetic KV context, 8192 tokens by default. Deployments serving much longer contexts may want to raise it so the profiled step reads a more realistic amount of cache (this matters a bit less for sparse attention models like DeepSeek-v4 since the cheap indexer is the main cost that scales with context length).

```bash
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=131072
```

## Saving and reusing the cost profile

The cost profile is measured once at startup, so a boot on a busy GPU or a node that has not reached its steady clocks prices every later step from a skewed profile, and two boots of the same configuration can differ. To inspect a profile, or to pin one that was measured on a quiet node, write the profiled curves to a file on one boot and read them back on later boots:

```bash
# Boot 1: profile as usual and write the curves.
export VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_DUMP=/var/lib/vllm/av_cost_curves.json

# Later boots: use the stored curves instead of the freshly profiled ones.
export VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_LOAD=/var/lib/vllm/av_cost_curves.json
```

The file holds the profiled curves (not the derived tables) and a fingerprint of what shapes them: the model and engine configuration, the capture sizes, the profiling context length, the tensor-parallel size, the GPU and the torch and CUDA versions. A boot only uses a file whose fingerprint matches its own. A missing, corrupt or mismatching file is never used silently: rank 0 logs a warning naming the fields that differ and keeps the profiled curves. Setting both variables fails at startup, since dumping and pinning at once is ambiguous. With neither variable set, nothing changes.

The profile runs again if the KV cache is resized after startup, so the file is overwritten on each profile and holds the last one.
