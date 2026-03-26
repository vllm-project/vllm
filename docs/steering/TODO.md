# Steering Implementation — Open Issues

Tracked issues from code review of the steering implementation on `naive-steering` branch.

## Pending Benchmark

### FlashAttention steering fix — choose between Option A and Option C

Both options are implemented on separate branches. Benchmark under load (pure decode and mixed prefill+decode) to decide which to merge.

- **PR #4 — Option A (`fix/steering-flash-attn-option-a`):** Adds `num_decode_tokens` field to `FlashAttentionMetadata`, populated via `split_decodes_and_prefills()` in the builder. No sync cost. Mirrors FlashInfer. Downside: larger diff from upstream.
- **PR #5 — Option C (`fix/steering-flash-attn-option-c`):** Adds fallback derivation in `get_num_decode_tokens()` from `max_query_len`/`query_start_loc`. More upstreamable. Downside: GPU→CPU sync on `query_start_loc` for mixed batches.

If Option C shows no measurable regression, prefer it for upstreamability. If the sync matters, use Option A. Merge one, abandon the other.

## Fixed

| Issue | PR | Branch |
|-------|----|--------|
| Dtype mismatch — steering vector upcast to float32 | [#3](https://github.com/RhizoNymph/vllm/pull/3) | `fix/steering-dtype` |
| String-based error routing, GET lock, uncached layer walk | [#6](https://github.com/RhizoNymph/vllm/pull/6) | `fix/steering-api-cleanup` |

## Low

### No GPU synchronization barrier for buffer updates

Buffer updates via `.copy_()` in `set_steering_vectors` can race with an in-flight forward pass. Safe for non-CUDA-graph paths (CUDA stream serialization), but could be undefined behavior during CUDA graph replay where the graph captures the buffer pointer.

CUDA stream ordering on the default stream means `.copy_()` is ordered with respect to subsequent kernel launches. The existing LoRA update pattern has the same characteristic and works in practice. Deferred until real-world issues are observed.

**Files:**
- `vllm/v1/worker/worker_base.py` — `.copy_()` call in `set_steering_vectors`
