# RIY: Profile-driven MoE expert pruning

RIY applies a per-layer expert profile when a model is loaded. It composes the
profile with vLLM's expert placement and avoids allocating pruned expert
weights.

## Load-time pruning

Pass a profile with `--riy-expert-profile`:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --riy-expert-profile /data/profiles/workload.json
```

The equivalent `RIY_EXPERT_PROFILE` environment variable is supported for
compatibility. The CLI option takes precedence.

### Version 1: pre-top-k masking

Existing version-1 profiles retain their original behavior:

```json
{
  "version": 1,
  "model": "Qwen3-30B-A3B",
  "pruned_experts": [[0, 3], [0, 11], [4, 7]]
}
```

RIY masks pruned router logits with `-inf`, then selects a complete top-k from
retained experts. A version-1 profile must therefore keep at least `top_k`
experts in every layer.

### Version 2: explicit routing mode

Version-2 profiles require `routing_mode`:

```json
{
  "version": 2,
  "routing_mode": "post_topk_drop",
  "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "model_revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
  "pruned_experts": [[0, 3], [0, 11], [4, 7]]
}
```

Supported modes:

- `pre_topk_mask`: mask logits before top-k and refill all routing slots.
- `post_topk_drop`: compute the original top-k, then zero pruned slots without
  replacement or renormalization.

Both modes compact the global-to-local expert map and reduce
`local_num_experts`. Unknown profile versions, unknown modes, duplicate entries,
and out-of-range coordinates fail before expert allocation.

`post_topk_drop` initially requires `--moe-backend triton`. It preserves global
logical IDs until the expert-map-aware Triton path maps pruned experts to `-1`.
Surviving routing weights remain unchanged. A layer may retain fewer experts
than `top_k`, but must retain at least one physical expert.

Profile pruning currently rejects EPLB, round-robin expert placement, and a
profile that removes every expert assigned to an EP rank.

## Optional monitoring

Set `VLLM_RIY_MONITOR=1` to collect per-layer expert frequency and routing-weight
statistics. Monitoring starts a process-local control server on
`127.0.0.1:8019`:

```bash
VLLM_RIY_MONITOR=1 vllm serve Qwen/Qwen3-30B-A3B
curl http://127.0.0.1:8019/riy/stats
```

For `post_topk_drop`, the statistics response also includes original, surviving,
and dropped top-k slots; average effective experts per token; the expert compute
reduction ratio; and per-layer effective-expert-count histograms.

The server can be bound to another trusted interface with `VLLM_RIY_HOST`. It
has no authentication or transport encryption and must not be exposed to an
untrusted network. Monitoring is not yet aggregated across TP or EP workers and
should remain disabled during formal performance runs.

## Implementation

- `vllm/model_executor/layers/fused_moe/riy.py` validates versioned profiles,
  builds mode-explicit layer plans, and owns optional statistics.
- `vllm/model_executor/layers/fused_moe/expert_map_manager.py` composes profile
  filtering with expert placement.
- `vllm/model_executor/layers/fused_moe/layer.py` wires allocation and the
  selected routing phase into each MoE layer.
- `vllm/model_executor/layers/fused_moe/router/base_router.py` applies either the
  pre-top-k logit mask or the post-top-k slot drop without renormalization.
