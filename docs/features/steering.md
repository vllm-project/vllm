# Activation Steering

Activation steering injects additive vectors into the residual stream of
decoder layers during generation.  This allows shifting model behaviour
(tone, topic, safety) without fine-tuning.

Steering is applied **only during decode steps** — prefill tokens are
never modified.

## Scope

**In scope:**

- Global (server-wide) steering via HTTP API
- Per-request steering via `SamplingParams.steering_vectors`
- Four hook points: `pre_attn`, `post_attn`, `post_mlp_pre_ln`, `post_mlp_post_ln`
- Additive combination: global + per-request vectors are summed
- Scheduler admission control for per-request configs
- CUDA graph and torch.compile compatibility (zero graph partitions)
- Gemma 3 model family

**Not in scope (future work):**

- Named / pre-registered steering configs (LoRA-style)
- Models other than Gemma 3 (requires wiring buffers into each model)
- Speculative decoding interaction (decode detection heuristic assumes 1 token)
- Per-request steering during prefill

## Enabling Steering

```bash
# Global steering only (always available, minimal overhead — 2-row table)
vllm serve google/gemma-3-4b-it

# Per-request steering (allocates larger tables, enables scheduler admission)
vllm serve google/gemma-3-4b-it --enable-steering --max-steering-configs 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-steering` | `False` | Allocate per-request steering table rows |
| `--max-steering-configs` | `4` | Max distinct per-request configs in one batch |

All four hook point buffers are always allocated on every decoder layer.
The memory cost is trivial (~3.6 MB for 26 layers at `max_steering_configs=4`).

Without `--enable-steering`, each layer gets a 2-row table (zeros + global).
Per-request `steering_vectors` in `SamplingParams` are silently ignored by
the scheduler since there is no `SteeringConfig` to enable admission control.

## Hook Points

Steering vectors can be applied at four positions within each decoder layer,
all operating on the residual stream:

| Hook Point | Position |
|-----------|----------|
| `pre_attn` | After `input_layernorm`, before `self_attn` |
| `post_attn` | After `post_attention_layernorm`, before `pre_feedforward_layernorm` |
| `post_mlp_pre_ln` | After `mlp`, before `post_feedforward_layernorm` |
| `post_mlp_post_ln` | After `post_feedforward_layernorm` |

All four are always active. The `apply_steering` custom op is **not** a
graph-splitting op — it is opaque to the torch.compile tracer (preventing
constant-folding) but does not partition the compiled graph. Zero rows
act as a no-op, so unused hook points add no computational overhead.

## Usage

### Global Steering (HTTP API)

Requires `VLLM_SERVER_DEV_MODE=1`.

```bash
# Set steering on layer 15 at the post_mlp_pre_ln hook point
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "post_mlp_pre_ln": {"15": [0.1, 0.2, ...]}
    },
    "scales": {"15": 1.5}
  }'

# Set steering at multiple hook points
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "pre_attn": {"15": [0.1, 0.2, ...]},
      "post_mlp_pre_ln": {"15": [0.3, 0.4, ...]}
    },
    "scales": {"15": 1.5}
  }'

# Check active steering
curl http://localhost:8000/v1/steering

# Clear all steering
curl -X POST http://localhost:8000/v1/steering/clear
```

Global vectors affect **all** decode tokens in **all** requests until cleared.

### Per-Request Steering (SamplingParams)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-3-4b-it",
    enable_steering=True,
    max_steering_configs=4,
)

# Request with steering at one hook point
steered = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={"post_mlp_pre_ln": {15: [0.1, 0.2, ...]}},
)

# Request with steering at multiple hook points
multi_hook = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={
        "pre_attn": {15: [0.1, 0.2, ...]},
        "post_mlp_pre_ln": {15: [0.3, 0.4, ...]},
    },
)

# Request without steering (unaffected)
normal = SamplingParams(max_tokens=100, temperature=0.7)

# All can run in the same batch
outputs = llm.generate(
    ["Be creative:", "Summarize this:", "Explain:"],
    [steered, multi_hook, normal],
)
```

### Per-Request Steering (OpenAI Client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "steering_vectors": {
            "pre_attn": {15: [0.1, 0.2, ...]},
            "post_mlp_pre_ln": {15: [0.3, 0.4, ...]},
        },
    },
)
```

## Data Flow

```
SamplingParams.steering_vectors
    │  {hook_point: {layer_idx: [floats]}}
    ▼
Request.steering_config_hash  ◄── deterministic SHA-256 hash
    │
    ▼
Scheduler ── checks capacity against max_steering_configs
    │         (same pattern as LoRA admission control)
    │         excess requests queued in skipped_waiting
    ▼
NewRequestData.steering_config_hash
    │
    ▼
Model Runner._update_states()
    ├── CachedRequestState.steering_config_hash
    ├── InputBatch.request_steering_config_hash[req_idx]
    └── SteeringManager.register_config(hash, vectors) → row assignment
    │
    ▼
Model Runner._update_steering_buffers()  (called before each forward pass)
    ├── SteeringManager.populate_steering_tables(layers)
    │     For EACH hook point's table:
    │       Row 0 = zeros (prefill sentinel)
    │       Row 1 = global vector for that hook point
    │       Rows 2+ = global + per_request (additive)
    └── Build steering_index: token → table row (shared across hook points)
          prefill tokens → 0
          decode, no per-request → 1
          decode, per-request → assigned row
    │
    ▼
Gemma3DecoderLayer.forward()
    # At each hook point position:
    residual = torch.ops.vllm.apply_steering(
        residual, self.steering_table_<hook>, self.steering_index
    )
    # = residual + steering_table_<hook>[steering_index[:N]]
```

## CUDA Graph and torch.compile Compatibility

The `apply_steering` custom op is registered but **not** as a splitting
op.  It is opaque to the torch.compile tracer (preventing constant-folding
of buffer values) but does not partition the compiled graph.  This means:

1. Steering adds **zero graph partitions** regardless of hook point count
2. In-place buffer updates (`steering_table_*`, `steering_index`) are
   visible across CUDA graph replays (same mechanism as KV cache)
3. Buffer shapes are fixed at allocation time — only content changes
4. All four hook points are unconditionally present in the compiled graph;
   zero rows make unused hook points a no-op

## Steering Table Layout

Each decoder layer has a `steering_table_<hook>` buffer per hook point
(e.g., `steering_table_post_mlp_pre_ln`):

```
Row 0:  [0, 0, 0, ..., 0]          ← prefill / no steering
Row 1:  [g₁, g₂, g₃, ..., gₕ]     ← global vector for this hook point
Row 2:  [g₁+a₁, g₂+a₂, ..., gₕ+aₕ] ← global + per-request config A
Row 3:  [g₁+b₁, g₂+b₂, ..., gₕ+bₕ] ← global + per-request config B
...
```

Each hook point has its own table with independent global/per-request vectors.

The shared `steering_index` buffer maps each token position to a row:

```
Token:  [decode₁, decode₂, prefill₁, prefill₂, prefill₃, decode₃]
Index:  [   2,       1,        0,        0,        0,       3    ]
```

## Deduplication

Requests with identical `steering_vectors` dicts produce the same
`steering_config_hash`.  The `SteeringManager` deduplicates by hash:
multiple requests sharing a config consume **one** table row with
reference counting.  When the last request using a config finishes,
the row is freed.

## Scheduler Admission Control

When `--enable-steering` is set, the scheduler tracks distinct steering
config hashes in the current batch (same pattern as `scheduled_loras`).
If `max_steering_configs` slots are occupied by distinct configs:

1. New requests with a **new** config hash are moved to `skipped_waiting`
2. They retry on the next scheduling step (FCFS priority)
3. Requests with a hash already in the batch pass through (dedup)
4. Requests without steering (`hash == 0`) are never blocked

## File Reference

| Component | File |
|-----------|------|
| Config | `vllm/config/steering.py` |
| Custom op + hook point enum | `vllm/model_executor/layers/steering.py` |
| Gemma 3 buffers | `vllm/model_executor/models/gemma3.py` |
| SteeringManager | `vllm/v1/worker/steering_manager.py` |
| InputBatch tracking | `vllm/v1/worker/gpu_input_batch.py` |
| Scheduler admission | `vllm/v1/core/sched/scheduler.py` |
| Model runner integration | `vllm/v1/worker/gpu_model_runner.py` |
| Worker global API | `vllm/v1/worker/worker_base.py` |
| HTTP endpoints | `vllm/entrypoints/serve/steering/api_router.py` |
| Protocol types | `vllm/entrypoints/serve/steering/protocol.py` |
| SamplingParams field | `vllm/sampling_params.py` |
| Request hash | `vllm/v1/request.py` |
| CLI args | `vllm/engine/arg_utils.py` |
| Prefix cache key integration | `vllm/v1/core/kv_cache_utils.py` (`_gen_steering_extra_hash_keys`) |

## Prefix Cache Key Integration

Per-request prefill steering config hashes are included in block hash extra
keys via `_gen_steering_extra_hash_keys()` in `kv_cache_utils.py`. This
ensures that blocks computed under different prefill steering produce
different cache entries.

Key design decisions:
- **Only prefill steering hash is included.** Decode steering does not affect
  KV cache content during prefill, so it is excluded from block hashes.
- **Zero impact when unused.** When `prefill_steering_config_hash` is 0 or
  absent, the helper returns an empty list, adding nothing to the extra keys.
- **Global prefill steering is NOT in per-request hashes.** Instead, global
  steering changes trigger `reset_prefix_cache()` to clear all cached blocks.
  This avoids encoding mutable global state into every block hash. The
  invariant is: within a "global steering epoch" (between cache resets), all
  cached blocks were computed with the same global state.
- **Forward-compatible.** Uses `getattr(request, 'prefill_steering_config_hash', 0)`
  so this code works even before the Request attribute is added.

## Invariants

1. **Row 0 is always zeros.** No token should ever receive steering from row 0.
2. **Prefill tokens always index row 0.** Steering is decode-only.
3. **Combined rows are recomputed every step.** `populate_steering_tables()` runs before each forward pass, so changes to global vectors are immediately reflected.
4. **Buffer shapes are fixed at init.** The table has `max_steering_configs + 2` rows regardless of how many are active. This is required for CUDA graph compatibility.
5. **The steering_index tensor is shared across all layers and all hook points.** One in-place update is visible to all decoder layers. Token-to-row mapping is independent of hook point.
6. **All four hook point buffers are always allocated.** The memory cost is trivial. Zero rows make unused hook points a no-op.
7. **The custom op is not a splitting op.** It prevents constant-folding but does not partition the compiled graph.
8. **Reference counting is exact.** Every `register_config` is balanced by a `release_config` when the request finishes. Rows are only freed at refcount 0.
9. **Validation is all-or-nothing.** `SamplingParams._validate_steering_vectors()` checks all keys and values before any request processing begins.
10. **Prefix cache correctness.** Blocks computed under different per-request prefill steering configs produce different block hashes. Global steering changes invalidate the entire prefix cache rather than being encoded per-block.
