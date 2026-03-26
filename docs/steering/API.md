# Steering API

REST endpoints for managing activation steering vectors at runtime. All endpoints require `VLLM_SERVER_DEV_MODE=1`.

## Endpoints

### POST /v1/steering/set

Set steering vectors on specific decoder layers. Only listed layers are modified; other layers keep their current state.

**Request body** (`SetSteeringRequest`):

```json
{
  "vectors": {
    "14": [0.1, -0.2, 0.3, "...hidden_size floats"],
    "20": [0.5, 0.1, -0.4, "...hidden_size floats"]
  },
  "scales": {
    "14": 1.5,
    "20": 2.0
  },
  "replace": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `vectors` | `dict[int, list[float]]` | Yes | Layer index to steering vector. Each vector must have length equal to the model's `hidden_size`. |
| `scales` | `dict[int, float]` | No | Per-layer scale factors. Defaults to 1.0 for unspecified layers. Scales are pre-multiplied into vectors before sending to workers. |
| `replace` | `bool` | No | When `true`, atomically clears all existing vectors before applying the new ones. Default `false`. |

**Success response** (200):

```json
{
  "status": "ok",
  "layers_updated": [14, 20]
}
```

**Error responses** (400):

```json
{"error": "No vectors provided. Include at least one layer index and vector."}
{"error": "Layer(s) [99] not found in model. Steerable layers that matched: [0, 1, ..., 25]"}
{"error": "Layer 14: expected vector of size 3584, got 100"}
{"error": "Layer 14: steering vector contains non-finite values (NaN or Infinity)"}
```

### POST /v1/steering/clear

Zero all steering vectors across all layers.

**Request body**: None.

**Success response** (200):

```json
{"status": "ok"}
```

### GET /v1/steering

Return which layers currently have non-zero steering vectors.

**Success response** (200):

```json
{
  "active_layers": {
    "14": {"norm": 1.234567},
    "20": {"norm": 0.567890}
  }
}
```

Returns an empty `active_layers` object when no steering is active.

## Data Flow

```
Client POST /v1/steering/set
    │
    ▼
api_router.py: set_steering()
    │  Pre-multiply scales into vectors
    │  Acquire asyncio lock
    │
    ▼
Phase 1 — Validate (collective_rpc → all workers)
    │  set_steering_vectors(vectors, validate_only=True)
    │  Workers return layer indices they own
    │  Router checks: all requested layers found?
    │
    ▼
Phase 2 — Apply (collective_rpc → all workers)
    │  If replace=true: clear_steering_vectors() first
    │  set_steering_vectors(vectors, validate_only=False)
    │  Workers copy vectors into model buffers via .copy_()
    │
    ▼
Model forward pass
    │  Gemma3DecoderLayer.forward() calls torch.ops.vllm.apply_steering()
    │  Custom op reads num_decode_tokens from ForwardContext
    │  Steering vector added to first N rows of hidden_states (decode tokens only)
    │
    ▼
Steered output
```

The two-phase flow prevents partial application in pipeline-parallel setups — if any worker would reject a vector (wrong size, non-finite values), no worker applies anything.

The `asyncio.Lock` serializes all mutations so a concurrent `/set` and `/clear` cannot interleave between the validate and apply phases.

## Usage Examples

### Python (requests)

```python
import requests

base = "http://localhost:8000"

# Set steering on layer 14 with scale 1.5
hidden_size = 3584  # Gemma 3 4B
vector = [0.01] * hidden_size

resp = requests.post(f"{base}/v1/steering/set", json={
    "vectors": {14: vector},
    "scales": {14: 1.5},
})
print(resp.json())  # {"status": "ok", "layers_updated": [14]}

# Check status
resp = requests.get(f"{base}/v1/steering")
print(resp.json())  # {"active_layers": {"14": {"norm": ...}}}

# Clear all
resp = requests.post(f"{base}/v1/steering/clear")
print(resp.json())  # {"status": "ok"}
```

### Atomic replacement

```python
# Replace all existing steering with a new configuration
resp = requests.post(f"{base}/v1/steering/set", json={
    "vectors": {10: new_vec_10, 20: new_vec_20},
    "replace": True,  # clears all layers first, then applies
})
```

### Python (programmatic, no server)

When using `vllm.LLM` directly without the HTTP server, use `collective_rpc`:

```python
from vllm import LLM

llm = LLM(model="google/gemma-3-4b-it")

# Set steering
vec = [10.0] * 3584
llm.collective_rpc("set_steering_vectors", args=({14: vec}, False))

# Generate with steering active
outputs = llm.generate(["Hello"], sampling_params)

# Clear
llm.collective_rpc("clear_steering_vectors")
```

## Key Files

| File | Purpose |
|------|---------|
| `vllm/entrypoints/serve/steering/api_router.py` | Endpoint handlers, lock, two-phase flow |
| `vllm/entrypoints/serve/steering/protocol.py` | `SetSteeringRequest` Pydantic model |
| `vllm/entrypoints/serve/__init__.py` | Router registration |
| `vllm/v1/worker/worker_base.py` | `set_steering_vectors`, `clear_steering_vectors`, `get_steering_status` |
