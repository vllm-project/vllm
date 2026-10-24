# FA3 Split-KV Wrong Logits Bug Analysis (#62626)

## Root Cause

FA3 combine kernel (`flash_fwd_combine_kernel.h`) chooses its tile scheduler and grid shape on the HOST at **capture time** using the **dummy batch** shape. CUDA graphs freeze this grid.

### The Problem Chain:

1. **Capture uses dummy batch:**
   - FULL graphs captured with `num_reqs = min(num_tokens, max_num_seqs)` dummy requests
   - Dummy has `max_seqlen_q = ceil(num_tokens / num_reqs)` per request
   - Example: 168-token graph with `max_num_seqs=16` → dummy has `max_seqlen_q=11`

2. **Scheduler chosen at capture time:**
   - FA3's `StaticVarlenTileScheduler::choose_scheduling_algo` runs on host
   - Decision: STANDARD vs LINEARIZE_M_AND_BATCH based on `max_seqlen_q`
   - STANDARD grid: `(ceil(max_seqlen_q * h / 8), k_blocks, b)` → `(11, 1, 16)` in example
   - Grid binds CTAs to requests (11 CTAs per request)

3. **Replay with different batch:**
   - Real batch may be 1 request with 161 tokens
   - Grid still has only 11 CTAs (frozen from capture)
   - Kernel reads real `cu_seqlens`: 161 × 8 = 1,288 rows
   - Only rows 0–87 are combined (11 × 8)
   - Other rows keep stale buffer values → WRONG LOGITS

## Prediction Table (All Cases Confirmed)

| Scenario | Captured Shape | Scheduler | Result |
|----------|---|---|---|
| 168-token, max_num_seqs 8–40 | 8×21...40×5 | STANDARD | ❌ Wrong |
| 168-token, max_num_seqs 48–64 | 48×4...64×3 | LINEARIZE | ✅ Correct |
| 2-token graph (trigger A) | 2×1 | STANDARD (1 row/req) | ❌ 2nd token never combined |
| Split cap=1 | — | No combine | ✅ Correct |

## Solution

**vLLM-side fix:** Use `max_seqlen_q = num_tokens` (upper bound) instead of `ceil(num_tokens / num_reqs)` for FULL graphs.

This ensures:
- Scheduler decision at capture time matches replay reality
- Grid shape accommodates the maximum possible single-request size
- Each CTA can process all its tokens regardless of request distribution

## Files to Change

1. `vllm/v1/worker/gpu/input_batch.py` — Make dummy batch with correct `max_seqlen_q`
2. `vllm/v1/worker/gpu/model_states/default.py` — Use upper-bound `max_seqlen_q`
3. `vllm/v1/worker/gpu/cudagraph_utils.py` — Thread `max_seqlen_q` through batch descriptor
4. Tests — Verify fix with reproduce script, both triggers A and B
