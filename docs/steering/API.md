# Steering API

## Global Steering (HTTP Endpoints)

REST endpoints for managing server-wide activation steering vectors at runtime. All endpoints require `VLLM_SERVER_DEV_MODE=1`.

Global steering affects all requests. For per-request steering, see the SamplingParams section below.

### POST /v1/steering/set

Set steering vectors on specific decoder layers. Supports three-tier steering: base (both phases), prefill-specific, and decode-specific. Only listed layers are modified; other layers keep their current state.

Setting `vectors` or `prefill_vectors` triggers prefix cache invalidation (all cached blocks are cleared and running requests are preempted).

**Request body** (`SetSteeringRequest`):

```json
{
  "vectors": {
    "post_mlp_pre_ln": {
      "14": [0.1, -0.2, 0.3, "...hidden_size floats"],
      "20": {"vector": [0.5, 0.1, -0.4, "..."], "scale": 2.0}
    }
  },
  "prefill_vectors": {
    "pre_attn": {
      "14": [0.3, 0.4, "...hidden_size floats"]
    }
  },
  "decode_vectors": {
    "post_attn": {
      "14": [0.2, -0.1, "...hidden_size floats"]
    }
  },
  "replace": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `vectors` | `SteeringVectorSpec` | No | Base vectors applied to both phases. |
| `prefill_vectors` | `SteeringVectorSpec` | No | Additive prefill-specific vectors. |
| `decode_vectors` | `SteeringVectorSpec` | No | Additive decode-specific vectors. |
| `replace` | `bool` | No | When `true`, atomically clears all existing vectors across all tiers before applying. Default `false`. |

Each `SteeringVectorSpec` is `{hook_point: {layer_idx: entry}}` where `entry` is either a bare `list[float]` (scale=1.0) or `{"vector": [...], "scale": float}`.

At least one of `vectors`, `prefill_vectors`, or `decode_vectors` must be provided.

**Success response** (200):

```json
{
  "status": "ok",
  "hook_points": ["post_mlp_pre_ln", "pre_attn", "post_attn"],
  "layers_updated": [14, 20]
}
```

**Error responses** (400):

```json
{"error": "No vectors provided. Include at least one of vectors, prefill_vectors, or decode_vectors with hook point/layer data."}
{"error": "Invalid hook point name(s): ['invalid']. Valid values: ['post_attn', 'post_mlp_post_ln', 'post_mlp_pre_ln', 'pre_attn']"}
{"error": "Layer(s) [99] not found in model. Steerable layers that matched: [0, 1, ..., 25]"}
{"error": "Layer 14: expected vector of size 3584, got 100"}
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

### SamplingParams Fields

```python
from vllm import SamplingParams

# Base steering (both phases)
params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={"post_mlp_pre_ln": {15: [0.1, 0.2, ...]}},
)

# Phase-specific
params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={"post_mlp_pre_ln": {15: [0.1, 0.2, ...]}},
    prefill_steering_vectors={"pre_attn": {15: [0.5, 0.6, ...]}},
    decode_steering_vectors={"pre_attn": {15: [0.3, 0.4, ...]}},
)

# Co-located scale
params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    steering_vectors={"post_mlp_pre_ln": {15: {"vector": [0.1, 0.2, ...], "scale": 1.5}}},
)
```

| Field | Type | Description |
|-------|------|-------------|
| `steering_vectors` | `SteeringVectorSpec \| None` | Base vectors (both phases). |
| `prefill_steering_vectors` | `SteeringVectorSpec \| None` | Prefill-specific additive vectors. |
| `decode_steering_vectors` | `SteeringVectorSpec \| None` | Decode-specific additive vectors. |

### OpenAI Client (extra_body)

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
        "prefill_steering_vectors": {
            "pre_attn": {15: [0.5, 0.6, ...]},
        },
    },
)
```

### Additive Composition

All tiers are additive. The effective vectors per phase are:

```
effective_prefill = global_base + global_prefill + request_base + request_prefill
effective_decode  = global_base + global_decode  + request_base + request_decode
```

Each component is independently pre-scaled by its co-located scale factor before addition.

## Data Flow

See [docs/features/steering.md](../features/steering.md) for the full data flow diagram covering three-tier composition, phase detection, and prefix cache integration.

## Usage Examples

### Python (requests) — Global Three-Tier

```python
import requests

base = "http://localhost:8000"
hidden_size = 3584  # Gemma 3 4B
vector = [0.01] * hidden_size

# Base steering (both phases) with co-located scale
resp = requests.post(f"{base}/v1/steering/set", json={
    "vectors": {
        "post_mlp_pre_ln": {
            "14": {"vector": vector, "scale": 1.5},
        },
    },
})
print(resp.json())

# Prefill-specific (additive)
resp = requests.post(f"{base}/v1/steering/set", json={
    "prefill_vectors": {
        "pre_attn": {"14": vector},
    },
})

# Check status
resp = requests.get(f"{base}/v1/steering")
print(resp.json())

# Clear all tiers
resp = requests.post(f"{base}/v1/steering/clear")
```

### Python (programmatic, no server) — Global

```python
from vllm import LLM

llm = LLM(model="google/gemma-3-4b-it")

# Set global base steering
vec = [10.0] * 3584
llm.collective_rpc(
    "set_steering_vectors",
    kwargs={"vectors": {"post_mlp_pre_ln": {14: vec}}},
)

# Set prefill-specific global
llm.collective_rpc(
    "set_steering_vectors",
    kwargs={"prefill_vectors": {"pre_attn": {14: vec}}},
)

# Generate with steering active
outputs = llm.generate(["Hello"], sampling_params)

# Clear
llm.collective_rpc("clear_steering_vectors")
```

### Python (programmatic) — Per-Request Three-Tier

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-3-4b-it",
    enable_steering=True,
    max_steering_configs=4,
)

# Base + phase-specific
steered = SamplingParams(
    max_tokens=100,
    steering_vectors={"post_mlp_pre_ln": {15: [1.0] * 3072}},
    prefill_steering_vectors={"pre_attn": {15: [0.5] * 3072}},
    decode_steering_vectors={"pre_attn": {15: [-0.5] * 3072}},
)

# Co-located scale
scaled = SamplingParams(
    max_tokens=100,
    steering_vectors={
        "post_mlp_pre_ln": {
            15: {"vector": [0.5] * 3072, "scale": 2.0},
        },
    },
)

normal = SamplingParams(max_tokens=100)

outputs = llm.generate(
    ["Prompt A", "Prompt B", "Prompt C"],
    [steered, scaled, normal],
)
```
