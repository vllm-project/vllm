# Steering API Implementation

This document describes how to expose activation steering via the vLLM API.

[TOC]

## Data Flow

To expose steering via the vLLM API, you need to modify several layers. Here's the complete data flow:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. API Request                                                          │
│     └─ ChatCompletionRequest / CompletionRequest                         │
│        (vllm/entrypoints/openai/chat_completion/protocol.py:150)         │
│                         │                                                │
│                         ▼                                                │
│  2. SamplingParams (or new SteeringParams)                               │
│     (vllm/sampling_params.py:156)                                        │
│                         │                                                │
│                         ▼                                                │
│  3. EngineCoreRequest                                                    │
│     (vllm/v1/engine/__init__.py:67)                                      │
│                         │                                                │
│                         ▼                                                │
│  4. Request (internal)                                                   │
│     (vllm/v1/request.py)                                                 │
│                         │                                                │
│                         ▼                                                │
│  5. SchedulerOutput.scheduled_new_reqs → NewRequestData                  │
│     (vllm/v1/core/sched/output.py:31)                                    │
│                         │                                                │
│                         ▼                                                │
│  6. Model Runner → ForwardContext → Model Forward                        │
│     (vllm/v1/worker/gpu/model_runner.py)                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Implementation Options

### Option A: Add Steering to SamplingParams (Minimal Changes)

The simplest approach - add steering fields directly to `SamplingParams`:

```python
# vllm/sampling_params.py

class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
):
    # ... existing fields ...

    # Steering configuration
    steering_vectors: dict[int, list[float]] | None = None
    """Dict mapping layer indices to steering vectors.
    Each vector should have length == hidden_size."""

    steering_scales: dict[int, float] | None = None
    """Dict mapping layer indices to scale factors. Defaults to 1.0."""

    steering_token_indices: list[int] | None = None
    """Which token positions to steer. None means all tokens."""
```

Then add to API protocol:

```python
# vllm/entrypoints/openai/chat_completion/protocol.py

class ChatCompletionRequest(OpenAIBaseModel):
    # ... existing fields ...

    # --8<-- [start:chat-completion-extra-params]
    # Steering (vLLM extension)
    steering_vectors: dict[int, list[float]] | None = Field(
        default=None,
        description="Layer index to steering vector mapping for activation steering."
    )
    steering_scales: dict[int, float] | None = Field(
        default=None,
        description="Layer index to scale factor mapping. Defaults to 1.0."
    )
    # --8<-- [end:chat-completion-extra-params]

    def to_sampling_params(self, max_tokens: int, default_sampling_params: dict) -> SamplingParams:
        # ... existing code ...
        return SamplingParams(
            # ... existing params ...
            steering_vectors=self.steering_vectors,
            steering_scales=self.steering_scales,
        )
```

### Option B: Create Separate SteeringParams (Cleaner Separation)

Create a dedicated steering config class:

```python
# vllm/steering_params.py (new file)

import msgspec
import torch
from dataclasses import dataclass

class SteeringParams(msgspec.Struct, omit_defaults=True):
    """Parameters for activation steering during inference."""

    vectors: dict[int, list[float]] | None = None
    """Layer index -> steering vector (as list for serialization)."""

    scales: dict[int, float] | None = None
    """Layer index -> scale factor."""

    positions: list[int] | None = None
    """Token positions to steer. None = all."""

    def to_tensors(self, device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
        """Convert vectors to GPU tensors."""
        if self.vectors is None:
            return {}
        return {
            layer_idx: torch.tensor(vec, device=device, dtype=dtype)
            for layer_idx, vec in self.vectors.items()
        }

    def get_scale(self, layer_idx: int) -> float:
        if self.scales is None:
            return 1.0
        return self.scales.get(layer_idx, 1.0)
```

Add to EngineCoreRequest:

```python
# vllm/v1/engine/__init__.py

class EngineCoreRequest(msgspec.Struct, ...):
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    steering_params: SteeringParams | None = None  # NEW
    # ... rest of fields ...
```

### Option C: Use extra_body (No Core Changes)

Use OpenAI's `extra_body` extension mechanism for minimal invasion:

```python
# Client-side usage
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "steering": {
            "vectors": {16: [0.1, -0.2, ...]},  # Layer 16
            "scales": {16: 2.0},
        }
    }
)
```

Handle in the serving layer:

```python
# vllm/entrypoints/openai/chat_completion/serving.py

async def create_chat_completion(self, request: ChatCompletionRequest, ...):
    # Extract steering from extra_body
    steering_config = None
    if hasattr(request, '__pydantic_extra__'):
        steering_config = request.__pydantic_extra__.get('steering')

    # Pass to engine...
```

## Model Runner Integration

Regardless of which option you choose, the model runner needs to apply steering:

```python
# vllm/v1/worker/gpu/model_runner.py

class GPUModelRunner:
    def __init__(self, ...):
        # ... existing init ...
        # Cache for converted steering tensors per request
        self._steering_cache: dict[str, dict[int, torch.Tensor]] = {}

    def _prepare_steering(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, dict[int, tuple[torch.Tensor, float]]] | None:
        """Prepare steering tensors for this batch."""
        steering_by_request = {}

        for req_data in scheduler_output.scheduled_new_reqs:
            steering = req_data.sampling_params.steering_vectors
            if steering:
                # Convert to tensors and cache
                tensors = {}
                for layer_idx, vec in steering.items():
                    tensors[layer_idx] = (
                        torch.tensor(vec, device=self.device, dtype=self.dtype),
                        req_data.sampling_params.steering_scales.get(layer_idx, 1.0)
                    )
                self._steering_cache[req_data.req_id] = tensors
                steering_by_request[req_data.req_id] = tensors

        # Include cached steering for continuing requests
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            if req_id in self._steering_cache:
                steering_by_request[req_id] = self._steering_cache[req_id]

        return steering_by_request if steering_by_request else None

    def execute_model(self, scheduler_output: SchedulerOutput, ...):
        # ... existing setup ...

        steering_by_request = self._prepare_steering(scheduler_output)

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            # Pass steering via additional_kwargs
            additional_kwargs={"steering": steering_by_request} if steering_by_request else None,
        ):
            model_output = self.model(**model_inputs)
```

## Model-Level Steering Application

In the model, read steering from ForwardContext:

```python
# vllm/model_executor/models/llama.py

class LlamaModel(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ):
        # ... embedding logic ...

        # Get steering config from forward context
        ctx = get_forward_context()
        steering_config = ctx.additional_kwargs.get("steering") if ctx.additional_kwargs else None

        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            hidden_states, residual = layer(positions, hidden_states, residual)

            # Apply steering after this layer
            if steering_config:
                self._apply_steering(hidden_states, residual, idx, steering_config)

        # ... rest of forward ...

    def _apply_steering(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_idx: int,
        steering_config: dict[str, dict[int, tuple[torch.Tensor, float]]],
    ):
        """Apply steering vectors to the residual stream."""
        # For simplicity, apply same steering to all tokens in batch
        # For per-request steering, you'd need token-to-request mapping
        for req_id, layer_steering in steering_config.items():
            if layer_idx in layer_steering:
                vec, scale = layer_steering[layer_idx]
                residual.add_(scale * vec)
```

## Client Usage Example

After implementing, clients can use steering like this:

```python
from openai import OpenAI
import numpy as np

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Load a pre-computed steering vector (e.g., from contrastive pairs or SAE)
happiness_vector = np.load("happiness_layer16.npy").tolist()

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "Describe your day."}
    ],
    extra_body={
        "steering_vectors": {16: happiness_vector},
        "steering_scales": {16: 1.5},
    }
)

print(response.choices[0].message.content)
```

## Files to Modify Summary

| File | Changes |
|------|---------|
| `vllm/sampling_params.py` | Add steering fields (Option A) |
| `vllm/steering_params.py` | New file (Option B) |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | Add steering to request |
| `vllm/entrypoints/openai/completion/protocol.py` | Add steering to request |
| `vllm/v1/engine/__init__.py` | Add steering to EngineCoreRequest |
| `vllm/v1/core/sched/output.py` | Add steering to NewRequestData |
| `vllm/v1/worker/gpu/model_runner.py` | Prepare and pass steering tensors |
| `vllm/forward_context.py` | Document steering in additional_kwargs |
| `vllm/model_executor/models/llama.py` | Apply steering in forward pass |

## Testing

```python
# tests/test_steering.py

import pytest
import torch
from vllm import LLM, SamplingParams

def test_steering_basic():
    llm = LLM(model="facebook/opt-125m")  # Small model for testing
    hidden_size = 768  # OPT-125m hidden size

    # Create a random steering vector
    steering_vec = torch.randn(hidden_size).tolist()

    sampling_params = SamplingParams(
        max_tokens=20,
        steering_vectors={6: steering_vec},  # Steer layer 6
        steering_scales={6: 1.0},
    )

    outputs = llm.generate(["Hello, how are you?"], sampling_params)
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0
```

## Related Resources

- [Live Activation Steering](STEERING.md) - Model-level steering implementation
- [Activation Extraction](EXTRACTION.md) - Extracting activations for SAE training
- [Architecture Overview](design/arch_overview.md) - vLLM system architecture
