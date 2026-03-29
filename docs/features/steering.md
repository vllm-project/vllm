# Activation Steering

Activation steering injects additive vectors into the residual stream of
decoder layers during generation.  This allows shifting model behaviour
(tone, topic, safety) without fine-tuning.

Steering supports a three-tier additive composition model with separate
prefill and decode phases:

```text
effective_prefill[hook][layer] = scale(steering_vectors) + scale(prefill_steering_vectors)
effective_decode[hook][layer]  = scale(steering_vectors) + scale(decode_steering_vectors)
```

Where each entry is either a bare `list[float]` (scale=1.0) or
`{"vector": [...], "scale": float}` with a co-located scale factor.

## Scope

**In scope:**

- Global (server-wide) steering via HTTP API
- Per-request steering via `SamplingParams.steering_vectors`
- Phase-specific steering via `prefill_steering_vectors` and `decode_steering_vectors`
- Co-located scale factors on each vector entry
- Four hook points: `pre_attn`, `post_attn`, `post_mlp_pre_ln`, `post_mlp_post_ln`
- Additive composition: base + phase-specific vectors are pre-scaled and summed
- Separate config hashes for prefill and decode phases
- Scheduler admission control for per-request configs
- CUDA graph and torch.compile compatibility (zero graph partitions)
- Gemma 3 model family

**Not in scope (future work):**

- Named / pre-registered steering configs (LoRA-style)
- Models other than Gemma 3 (requires wiring buffers into each model)

## Enabling Steering

```bash
# Global steering only (always available, minimal overhead — 2-row table)
vllm serve google/gemma-3-4b-it

# Per-request steering (allocates larger tables, enables scheduler admission)
vllm serve google/gemma-3-4b-it --enable-steering --max-steering-configs 4
```

| Flag                      | Default | Description                                  |
|---------------------------|---------|----------------------------------------------|
| `--enable-steering`       | `False` | Allocate per-request steering table rows     |
| `--max-steering-configs`  | `4`     | Max distinct per-request configs in one batch|

All four hook point buffers are always allocated on every decoder layer.
The memory cost is trivial (~3.6 MB for 26 layers at `max_steering_configs=4`).

Without `--enable-steering`, each layer gets a 3-row table (zeros + global prefill + global decode).
Per-request `steering_vectors` in `SamplingParams` are silently ignored by
the scheduler since there is no `SteeringConfig` to enable admission control.

## Hook Points

Steering vectors can be applied at four positions within each decoder layer,
all operating on the residual stream:

| Hook Point         | Position                                                             |
|--------------------|----------------------------------------------------------------------|
| `pre_attn`         | After `input_layernorm`, before `self_attn`                          |
| `post_attn`        | After `post_attention_layernorm`, before `pre_feedforward_layernorm` |
| `post_mlp_pre_ln`  | After `mlp`, before `post_feedforward_layernorm`                     |
| `post_mlp_post_ln` | After `post_feedforward_layernorm`                                   |

All four are always active. The `apply_steering` custom op is **not** a
graph-splitting op — it is opaque to the torch.compile tracer (preventing
constant-folding) but does not partition the compiled graph. Zero rows
act as a no-op, so unused hook points add no computational overhead.

## Usage

### Global Steering (HTTP API)

Requires `VLLM_SERVER_DEV_MODE=1`.

```bash
# Set base steering (applies to both prefill and decode)
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "post_mlp_pre_ln": {"15": [0.1, 0.2, ...]}
    }
  }'

# Co-located scale factor (scale embedded in the vector entry)
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "post_mlp_pre_ln": {
        "15": {"vector": [0.1, 0.2, ...], "scale": 1.5}
      }
    }
  }'

# Three-tier: base + prefill-specific + decode-specific
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "post_mlp_pre_ln": {"15": [0.1, 0.2, ...]}
    },
    "prefill_vectors": {
      "pre_attn": {"15": [0.5, 0.6, ...]}
    },
    "decode_vectors": {
      "pre_attn": {"15": [0.3, 0.4, ...]}
    }
  }'

# Replace all existing vectors atomically
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {"post_mlp_pre_ln": {"15": [0.1, 0.2, ...]}},
    "replace": true
  }'

# Check active steering
curl http://localhost:8000/v1/steering

# Clear all steering (all tiers)
curl -X POST http://localhost:8000/v1/steering/clear
```

Global vectors affect **all** requests until cleared.  Base vectors
affect both prefill and decode phases.  Phase-specific vectors
(`prefill_vectors`, `decode_vectors`) are additive on top of base.

### Per-Request Steering (SamplingParams)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-3-4b-it",
    enable_steering=True,
    max_steering_configs=4,
)

# Base steering (applies to both prefill and decode)
steered = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={"post_mlp_pre_ln": {15: [0.1, 0.2, ...]}},
)

# Co-located scale factor
scaled = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={
        "post_mlp_pre_ln": {
            15: {"vector": [0.1, 0.2, ...], "scale": 2.0}
        }
    },
)

# Phase-specific: different steering for prefill vs decode
phase_specific = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={"post_mlp_pre_ln": {15: [0.1, 0.2, ...]}},
    prefill_steering_vectors={"pre_attn": {15: [0.5, 0.6, ...]}},
    decode_steering_vectors={"pre_attn": {15: [0.3, 0.4, ...]}},
)

# Request without steering (unaffected)
normal = SamplingParams(max_tokens=100, temperature=0.7)

# All can run in the same batch
outputs = llm.generate(
    ["Be creative:", "Summarize:", "Explain:", "Hello:"],
    [steered, scaled, phase_specific, normal],
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

```text
SamplingParams
    ├── steering_vectors        (base, both phases)
    ├── prefill_steering_vectors  (prefill-specific)
    └── decode_steering_vectors   (decode-specific)
    │
    ▼  resolve_effective_vectors() via cached_property
SamplingParams.effective_prefill_steering  → pre-scaled flat vectors
SamplingParams.effective_decode_steering   → pre-scaled flat vectors
    │
    ▼  hash_steering_config()
Request.prefill_steering_config_hash  ◄── deterministic SHA-256 hash
Request.decode_steering_config_hash   ◄── deterministic SHA-256 hash
    │
    ▼
Scheduler ── checks capacity against max_steering_configs
    │         (tracks union of active hashes: prefill for prefill,
    │          decode for decode; new requests must fit BOTH hashes)
    │         excess requests queued in skipped_waiting
    ▼
NewRequestData.prefill_steering_config_hash
NewRequestData.decode_steering_config_hash
    │
    ▼
Model Runner._update_states()
    ├── CachedRequestState.{prefill,decode}_steering_config_hash
    ├── InputBatch.request_{prefill,decode}_steering_hash[req_idx]
    └── SteeringManager.register_config(hash, vectors, phase="prefill")
    │     Only prefill config registered initially; decode registered on
    │     phase transition
    │
    ▼
Model Runner._update_steering_buffers()  (called before each forward pass)
    ├── SteeringManager.populate_steering_tables(layers)
    │     For EACH hook point's table:
    │       Row 0 = zeros (no-steering sentinel)
    │       Row 1 = global_base + global_prefill (prefill effective)
    │       Row 2 = global_base + global_decode (decode effective)
    │       Rows 3+ = phase-appropriate global + per_request (additive)
    ├── Build steering_index: token → table row (shared across hook points)
    │     prefill tokens, no per-request → 1 (global prefill)
    │     prefill tokens, per-request → prefill hash row (3+)
    │     decode tokens, no per-request → 2 (global decode)
    │     decode tokens, per-request → decode hash row (3+)
    └── Detect prefill→decode transitions and swap configs
          _handle_steering_transition() releases prefill, registers decode
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

```text
Row 0:  [0, 0, 0, ..., 0]              ← no steering (zeros sentinel)
Row 1:  [gB+gP₁, gB+gP₂, ..., gB+gPₕ] ← global prefill effective (base + prefill)
Row 2:  [gB+gD₁, gB+gD₂, ..., gB+gDₕ] ← global decode effective (base + decode)
Row 3:  [(gB+gP)+a₁, ..., (gB+gP)+aₕ]  ← prefill global + per-request config A
Row 4:  [(gB+gD)+b₁, ..., (gB+gD)+bₕ]  ← decode global + per-request config B
...
```

Each hook point has its own table with independent global/per-request vectors.
The global effective vectors are composed from three tiers:

- **base**: applies to both phases (from `steering_vectors` global API)
- **prefill**: prefill-specific globals
- **decode**: decode-specific globals

Per-request rows (3+) combine the phase-appropriate global effective vector
with the per-request vector based on the config's registered phase.

The shared `steering_index` buffer maps each token position to a row:

```text
Token:  [decode₁, decode₂, prefill₁, prefill₂, prefill₃, decode₃]
Index:  [   4,       2,        1,        1,        3,       4    ]
```

Phase detection uses `num_computed_tokens < num_prompt_tokens` (not the
`n_tokens == 1` heuristic), making it correct for chunked prefill and
speculative decoding.

## Deduplication

Requests with identical effective steering vectors (after resolution and
pre-scaling) produce the same config hash.  The `SteeringManager`
deduplicates by hash: multiple requests sharing a config consume **one**
table row with reference counting.  When the last request using a config
finishes, the row is freed.  Prefill and decode hashes are tracked
independently.

## Scheduler Admission Control

When `--enable-steering` is set, the scheduler tracks distinct steering
config hashes in the current batch (same pattern as `scheduled_loras`).

**Running request hash collection:** For each running request, the
scheduler adds its currently-active hash: `prefill_steering_config_hash`
for requests still in prefill (`num_computed_tokens < num_tokens`),
`decode_steering_config_hash` for those in decode.

**New request admission:** A new request may need up to two distinct
hashes over its lifetime (one for prefill, one for decode).  The
scheduler counts how many genuinely new unique hashes the request would
add to the scheduled set.  If the union would exceed
`max_steering_configs`, the request is skipped.

1. Compute `new_hashes = {prefill_hash, decode_hash}` (excluding zeros)
2. Compute `new_unique = new_hashes - scheduled_steering_configs`
3. If `len(scheduled) + len(new_unique) > max_configs`, skip
4. When admitted, add **both** hashes to the scheduled set
5. Requests without steering (`both hashes == 0`) are never blocked
6. Requests whose hashes are already in the batch pass through (dedup)

## File Reference

| Component                      | File                                                                |
|--------------------------------|---------------------------------------------------------------------|
| Config                         | `vllm/config/steering.py`                                           |
| Type definitions + helpers     | `vllm/config/steering_types.py`                                     |
| Custom op + hook point enum    | `vllm/model_executor/layers/steering.py`                            |
| Gemma 3 buffers                | `vllm/model_executor/models/gemma3.py`                              |
| SteeringManager                | `vllm/v1/worker/steering_manager.py`                                |
| InputBatch tracking            | `vllm/v1/worker/gpu_input_batch.py`                                 |
| Scheduler admission            | `vllm/v1/core/sched/scheduler.py`                                   |
| Model runner integration       | `vllm/v1/worker/gpu_model_runner.py`                                |
| Worker global API              | `vllm/v1/worker/worker_base.py`                                     |
| HTTP endpoints                 | `vllm/entrypoints/serve/steering/api_router.py`                     |
| Protocol types                 | `vllm/entrypoints/serve/steering/protocol.py`                       |
| SamplingParams field           | `vllm/sampling_params.py`                                           |
| Request hash                   | `vllm/v1/request.py`                                                |
| CLI args                       | `vllm/engine/arg_utils.py`                                          |
| Prefix cache key integration   | `vllm/v1/core/kv_cache_utils.py` (`_gen_steering_extra_hash_keys`)  |

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

1. **Row 0 is always zeros.** No token should ever receive steering from row 0 (unless no steering is configured at all).
2. **Prefill and decode phases use independent config hashes.** Each phase resolves its own effective vectors and hashes them separately.
3. **Combined rows are recomputed every step.** `populate_steering_tables()` runs before each forward pass, so changes to global vectors are immediately reflected.
4. **Buffer shapes are fixed at init.** The table has `max_steering_configs + 3` rows (0=zeros, 1=global prefill, 2=global decode, 3+=per-request). This is required for CUDA graph compatibility.
5. **The steering_index tensor is shared across all layers and all hook points.** One in-place update is visible to all decoder layers. Token-to-row mapping is independent of hook point.
6. **All four hook point buffers are always allocated.** The memory cost is trivial. Zero rows make unused hook points a no-op.
7. **The custom op is not a splitting op.** It prevents constant-folding but does not partition the compiled graph.
8. **One-active-at-a-time registration.** Only one phase's config is registered per request at any time. Prefill config is registered on request creation; on prefill->decode transition, the prefill config is released and the decode config is registered. On request completion, only the currently-active config is released.
9. **Validation is all-or-nothing.** `SamplingParams._validate_steering_vectors()` checks all three vector fields before any request processing begins.
10. **Additive composition is pre-scaled.** `resolve_effective_vectors()` applies co-located scales before summing base and phase-specific vectors.
11. **Phase detection is token-count based.** `num_computed_tokens < num_prompt_tokens` determines prefill vs decode, not the `n_tokens == 1` heuristic.
12. **Reference counting is exact.** Every `register_config` is balanced by a `release_config` when the request finishes. Rows are only freed at refcount 0.
13. **Prefix cache correctness.** Blocks computed under different per-request prefill steering configs produce different block hashes. Global steering changes invalidate the entire prefix cache rather than being encoded per-block.
14. **Phase-specific global vectors survive lazy init.** When `set_steering_vectors()` is called with `prefill_vectors` or `decode_vectors` before the first forward pass (before `SteeringManager` is lazily created), the phase-specific vectors are captured with their phase labels into `_pending_steering_globals` on the model runner. During lazy init, these pending entries are replayed in order so that base, prefill, and decode globals are registered with the correct phase. Without this, all vectors would collapse into the last-written buffer contents and be misidentified as base-phase.
15. **`clear_steering_vectors()` clears pending globals.** When clearing steering state, both the live `SteeringManager` globals and the `_pending_steering_globals` list are cleared. This prevents stale vectors from being replayed during lazy init if a clear (or `replace=True` update, which calls clear internally) happens before the first forward pass.
