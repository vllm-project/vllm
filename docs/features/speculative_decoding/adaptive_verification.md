# Adaptive Verification

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. At batch size 256 it is a much more delicate one. Draft tokens now compete with real tokens for the same compute, and every rejected token is compute wasted; with enough of them, throughput drops.

That matters because per-position acceptance decays fast. While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost. The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies.

Adaptive verification decides per step how much of the draft to verify instead.
DSpark scores every (request, position) draft slot by its *survival
probability*, the running product of that request's per-position confidences,
and admits the highest-scoring slots until a global budget is spent. DFlash
uses its observed accepted-prefix survival to choose one K for the next batch.

The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.

The practical effect is that one configuration holds up across the whole load range, which removes most of the need to tune `num_speculative_tokens` per deployment.

## Support

Adaptive verification supports:

- DSpark checkpoints with a **confidence head**. It trims the current
  verification batch per request.
- DFlash checkpoints. It chooses a uniform K from batch size, accepted-prefix
  history, and the profiled draft/verify costs. K=0 skips the DFlash forward and
  runs ordinary target decoding for the next step.

## Usage

It is off by default. Enable it in the speculative config:

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

Set `enable_adaptive_verification: false` to verify the full block for every request.

For DFlash, use the same flag with `"method": "dflash"`. The configured
`num_speculative_tokens` remains the maximum K. DFlash drafts all mask positions
in one parallel forward, so lowering a nonzero K reduces target verification
work but not draft-model work; K=0 is the only choice that skips the drafter.
The runtime considers compact graph buckets (K=0, 1, 3, 7, ... and the
configured maximum) rather than capturing a graph for every integer K.

## Requirements and limitations

- DSpark requires an attention backend that tolerates device-decided query
  lengths. Backends that plan off CPU lengths are rejected at startup.
- Full cudagraphs are required: step costs are profiled from captured graphs,
  so `--enforce-eager` is rejected at startup.
- Adaptive verification is not supported with LoRA or pipeline parallelism.
  DSpark additionally does not support output logprobs. DFlash automatic K
  does not compact the current verification logits and supports output
  logprobs.

## Tuning the cost profile

Step costs are profiled against a synthetic KV context, 8192 tokens by default. Deployments serving much longer contexts may want to raise it so the profiled step reads a more realistic amount of cache (this matters a bit less for sparse attention models like DeepSeek-v4 since the cheap indexer is the main cost that scales with context length).

```bash
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=131072
```
