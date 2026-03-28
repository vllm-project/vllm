# Steering API

## Global Steering (HTTP Endpoints)

REST endpoints for managing server-wide activation steering vectors at runtime. All endpoints require `VLLM_SERVER_DEV_MODE=1`.

Global steering affects all requests. For per-request steering, see the SamplingParams section below.

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

## Per-Request Steering (SamplingParams)

Per-request steering does not require the HTTP API or dev mode. It requires `--enable-steering` on the server.

### SamplingParams Field

```python
from vllm import SamplingParams

params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={15: [0.1, 0.2, ...], 20: [0.3, 0.4, ...]},
)
```

| Field | Type | Description |
|-------|------|-------------|
| `steering_vectors` | `dict[int, list[float]] \| None` | Layer index to steering vector. Keys are non-negative ints, values are lists of finite floats with length `hidden_size`. |

### OpenAI Client (extra_body)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "steering_vectors": {15: [0.1, 0.2, ...]},
    },
)
```

### Additive Combination

When both global and per-request steering are active, the effective vector per layer is:

```
effective = global_vector + per_request_vector
```

Per-request steering without global steering applies just the per-request vector. Requests without per-request steering get just the global vector (or nothing if no global steering is set).

## Data Flow

### Global Steering

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
    │  Workers copy vectors into steering_vector buffers
    │  Workers notify SteeringManager.update_global_vectors()
    │
    ▼
Model forward pass
    │  SteeringManager.populate_steering_tables() writes global to row 1
    │  _update_steering_buffers() sets steering_index for each token
    │  apply_steering(residual, table, index) adds table[index[:N]] to residual
    │
    ▼
Steered output
```

### Per-Request Steering

```
SamplingParams(steering_vectors={15: [...]})
    │
    ▼
Request.steering_config_hash = SHA-256 hash of vectors dict
    │
    ▼
Scheduler: check len(scheduled_steering_configs) < max_steering_configs
    │  If at capacity with new hash → skipped_waiting queue
    │
    ▼
NewRequestData.steering_config_hash → Model Runner
    │
    ▼
SteeringManager.register_config(hash, vectors) → assigns table row
    │
    ▼
InputBatch.request_steering_config_hash[req_idx] = hash
    │
    ▼
_update_steering_buffers():
    │  populate_steering_tables() → row N = global + per_request
    │  steering_index[token_offset] = row N for this request's decode tokens
    │
    ▼
apply_steering(residual, table, index) → per-token steering via gather
```

The two-phase flow in global steering prevents partial application in pipeline-parallel setups — if any worker would reject a vector (wrong size, non-finite values), no worker applies anything.

## Usage Examples

### Python (requests) — Global

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

### Python (programmatic, no server) — Global

```python
from vllm import LLM

llm = LLM(model="google/gemma-3-4b-it")

# Set global steering
vec = [10.0] * 3584
llm.collective_rpc("set_steering_vectors", args=({14: vec}, False))

# Generate with steering active
outputs = llm.generate(["Hello"], sampling_params)

# Clear
llm.collective_rpc("clear_steering_vectors")
```

### Python (programmatic) — Per-Request

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-3-4b-it",
    enable_steering=True,
    max_steering_configs=4,
)

# Different steering per request
steered_a = SamplingParams(
    max_tokens=100,
    steering_vectors={15: [1.0] * 3072},
)
steered_b = SamplingParams(
    max_tokens=100,
    steering_vectors={15: [-1.0] * 3072},
)
normal = SamplingParams(max_tokens=100)

outputs = llm.generate(
    ["Prompt A", "Prompt B", "Prompt C"],
    [steered_a, steered_b, normal],
)
```

## Key Files

| File | Purpose |
|------|---------|
| `vllm/entrypoints/serve/steering/api_router.py` | Endpoint handlers, lock, two-phase flow |
| `vllm/entrypoints/serve/steering/protocol.py` | `SetSteeringRequest` Pydantic model |
| `vllm/entrypoints/serve/__init__.py` | Router registration |
| `vllm/v1/worker/worker_base.py` | `set_steering_vectors`, `clear_steering_vectors`, `get_steering_status` |
| `vllm/sampling_params.py` | `steering_vectors` field on `SamplingParams` |
| `vllm/v1/request.py` | `steering_config_hash` property |
| `vllm/engine/arg_utils.py` | `--enable-steering`, `--max-steering-configs` CLI args |
