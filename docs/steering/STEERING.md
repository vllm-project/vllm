# Live Activation Steering

This document describes how to modify activations during inference for steering, representation engineering, or SAE feature manipulation in vLLM.

[TOC]

## Overview

Activation steering allows you to intervene on model hidden states at runtime, enabling:
- **Representation engineering**: Adding "concept vectors" to shift model behavior
- **SAE feature steering**: Amplifying or suppressing specific SAE features
- **Safety interventions**: Modifying activations to reduce harmful outputs
- **Behavioral control**: Adjusting style, tone, or content properties

## Intervention Points

The decoder layer forward pass (`llama.py:315-332`) provides natural intervention points:

```
┌─────────────────────────────────────────────────────────────────┐
│  LlamaDecoderLayer.forward()                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  hidden_states, residual = input_layernorm(hidden_states)       │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ INTERVENTION POINT 1: Pre-attention                     │   │
│  │ Modify hidden_states before attention                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  hidden_states = self_attn(hidden_states)                       │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ INTERVENTION POINT 2: Post-attention                    │   │
│  │ Modify attention output before MLP                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  hidden_states, residual = post_attention_layernorm(...)        │
│                         ↓                                       │
│  hidden_states = mlp(hidden_states)                             │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ INTERVENTION POINT 3: Post-MLP / Post-layer             │   │
│  │ Modify final layer output (most common for steering)    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  return hidden_states, residual                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/models/llama.py:315-332` | `LlamaDecoderLayer.forward()` - primary intervention point |
| `vllm/model_executor/models/llama.py:421-428` | `LlamaModel.forward()` layer loop - model-level steering |
| `vllm/forward_context.py:241` | `ForwardContext.additional_kwargs` - pass steering config |
| `vllm/v1/worker/gpu/model_runner.py:980` | Model execution - inject steering tensors |

## Implementation Approaches

### Approach 1: Modify Decoder Layer

Extend a model's decoder layer to accept steering vectors. This example shows `LlamaDecoderLayer`, but most models have their own implementation you'd need to modify:

```python
# vllm/model_executor/models/llama.py

class LlamaDecoderLayer(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        steering_vector: torch.Tensor | None = None,  # Shape: [hidden_size]
        steering_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        # STEERING: Add steering vector to residual stream
        if steering_vector is not None:
            hidden_states = hidden_states + steering_scale * steering_vector

        return hidden_states, residual
```

### Approach 2: ForwardContext-Based Steering

Pass steering configuration via `ForwardContext.additional_kwargs`:

```python
# Steering config structure
@dataclass
class SteeringConfig:
    vectors: dict[int, torch.Tensor]  # layer_idx -> steering vector
    scales: dict[int, float]          # layer_idx -> scale factor
    positions: torch.Tensor | None    # Which token positions to steer (None = all)

# In model forward, access steering config
from vllm.forward_context import get_forward_context

class LlamaDecoderLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)  # Get layer index from prefix

    def forward(self, positions, hidden_states, residual):
        # ... normal forward ...

        hidden_states = self.mlp(hidden_states)

        # Check for steering in forward context
        ctx = get_forward_context()
        steering = ctx.additional_kwargs.get("steering")
        if steering and self.layer_idx in steering.vectors:
            vec = steering.vectors[self.layer_idx]
            scale = steering.scales.get(self.layer_idx, 1.0)
            if steering.positions is not None:
                # Selective position steering
                mask = steering.positions
                hidden_states[mask] = hidden_states[mask] + scale * vec
            else:
                hidden_states = hidden_states + scale * vec

        return hidden_states, residual
```

### Approach 3: Model-Level Layer Loop Steering

Modify a model's main forward loop to inject steering between layers. This example shows `LlamaModel`, but each model family has its own implementation:

```python
# vllm/model_executor/models/llama.py

class LlamaModel(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        steering_config: dict | None = None,  # {layer_idx: (vector, scale)}
        **extra_layer_kwargs,
    ):
        # ... embedding logic ...

        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            hidden_states, residual = layer(positions, hidden_states, residual)

            # Apply steering after specified layers
            if steering_config and idx in steering_config:
                vec, scale = steering_config[idx]
                # Steer the residual stream (hidden_states + residual)
                residual = residual + scale * vec

        # ... rest of forward ...
```

### Approach 4: Model Runner Integration

For production steering, integrate at the model runner level:

```python
# vllm/v1/worker/gpu/model_runner.py

class GPUModelRunner:
    def __init__(self, ...):
        # ... existing init ...
        self.steering_vectors: dict[int, torch.Tensor] = {}
        self.steering_scales: dict[int, float] = {}

    def set_steering(self, layer_idx: int, vector: torch.Tensor, scale: float = 1.0):
        """Set steering vector for a specific layer."""
        self.steering_vectors[layer_idx] = vector.to(self.device)
        self.steering_scales[layer_idx] = scale

    def clear_steering(self):
        """Remove all steering vectors."""
        self.steering_vectors.clear()
        self.steering_scales.clear()

    def execute_model(self, scheduler_output, ...):
        # ... existing setup ...

        # Pass steering config to model
        model_kwargs = {}
        if self.steering_vectors:
            model_kwargs["steering_config"] = {
                idx: (vec, self.steering_scales.get(idx, 1.0))
                for idx, vec in self.steering_vectors.items()
            }

        with set_forward_context(...):
            model_output = self.model(**model_inputs, **model_kwargs)
```

### Approach 5: PyTorch Hooks (Model-Agnostic)

Use PyTorch's `register_forward_hook` to intercept layer outputs without modifying any model code. This works for all models:

```python
# vllm/v1/worker/gpu/model_runner.py
import re

class GPUModelRunner:
    def __init__(self, ...):
        # ... existing init ...
        self._steering_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._steering_config: dict[int, tuple[torch.Tensor, float]] = {}

    def set_steering(self, layer_idx: int, vector: torch.Tensor, scale: float = 1.0):
        """Set steering vector for a specific layer."""
        self._steering_config[layer_idx] = (vector.to(self.device), scale)
        self._refresh_steering_hooks()

    def clear_steering(self):
        """Remove all steering vectors and hooks."""
        self._clear_steering_hooks()
        self._steering_config.clear()

    def _clear_steering_hooks(self):
        for handle in self._steering_hooks:
            handle.remove()
        self._steering_hooks.clear()

    def _refresh_steering_hooks(self):
        """Attach hooks to layers specified in steering config."""
        self._clear_steering_hooks()

        if not self._steering_config:
            return

        for name, module in self.model.named_modules():
            # Match common layer patterns across models:
            # - "model.layers.16" (Llama, Mistral, Qwen)
            # - "transformer.h.16" (GPT-2, GPT-J)
            # - "model.decoder.layers.16" (OPT, BART)
            match = re.search(r'(?:layers|\.h)\.(\d+)$', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx in self._steering_config:
                    vec, scale = self._steering_config[layer_idx]
                    handle = module.register_forward_hook(
                        self._make_steering_hook(vec, scale)
                    )
                    self._steering_hooks.append(handle)

    def _make_steering_hook(self, vector: torch.Tensor, scale: float):
        """Create a hook function that adds the steering vector."""
        def hook(module, input, output):
            # Handle different output formats:
            # - tuple: (hidden_states, residual) for Llama-style
            # - tensor: hidden_states directly
            if isinstance(output, tuple):
                hidden_states, residual = output
                # Steer the residual stream
                return (hidden_states, residual + scale * vector)
            else:
                return output + scale * vector
        return hook
```

This approach finds layers by name pattern matching, so it works across model architectures without code changes.

## Approach Comparison

| Approach | Model Changes Required | Per-Request | CUDA Graphs | Complexity |
|----------|------------------------|-------------|-------------|------------|
| **1. Decoder Layer** | Each model family | Yes | Yes | Repetitive* |
| **2. ForwardContext** | Each model family | Yes | Yes | Repetitive* |
| **3. Model Loop** | Each model family | Yes | Yes | Repetitive* |
| **4. Model Runner** | Each model family | Yes | Yes | Repetitive* |
| **5. PyTorch Hooks** | **None** | Yes | **No** | **Low** |

*Same ~5-line pattern across 60+ files - tractable with AI coding tools.

### Model Architecture Reality

Most models have their **own** `DecoderLayer` implementation. Contrary to what you might expect, very few inherit from `LlamaDecoderLayer`:

**Models with their OWN DecoderLayer (60+):**
- Qwen2, Qwen3, Qwen2Moe, Qwen3Moe
- Gemma, Gemma2, Gemma3
- Mixtral, DeepseekV2
- OPT, Falcon, Baichuan
- InternLM2, MiniCPM
- Olmo, Olmo2, Starcoder2
- Command-R, Granite, GraniteMoe
- Mamba, Mamba2, Jamba
- And 50+ more...

**Models that DO inherit from LlamaDecoderLayer (few):**
- Mistral, Aria, TeleFLM, TeleChat2

**Models using LlamaForCausalLM directly:**
- Phi3, GLM, GritLM, Ernie4.5, Fairseq2Llama
- Some via registry: Aquila, Cwm

This means **modifying decoder layers requires touching each model family** for full coverage.

### When to Use Each Approach

| Approach | Best For |
|----------|----------|
| **1-4. Model Changes** | Production deployment with CUDA graph performance |
| **5. PyTorch Hooks** | Quick prototyping, can't modify vLLM, testing ideas |

### Key Trade-offs

**Approaches 1-4** (model modifications):
- Full CUDA graph compatibility
- No runtime overhead
- Fine-grained control (pre-attn, post-attn, post-MLP)
- Requires modifying each model family (repetitive but tractable)

**Approach 5 (Hooks)**:
- Works for **ALL models** with zero code changes
- **Incompatible with CUDA graphs** (must disable: `cudagraph_mode = CUDAGraphMode.NONE`)
- Slight overhead from Python hook dispatch
- Limited to post-layer intervention only

### Recommendation

| Use Case | Recommended Approach |
|----------|---------------------|
| **Quick prototyping** | PyTorch Hooks (Approach 5) - zero code changes, works immediately |
| **Production** | Modify models via ForwardContext (Approach 2) |

**On modifying 60+ models**: While this sounds daunting, the change is nearly identical across all decoder layers:

```python
# Same ~5 lines added to every DecoderLayer.forward(), after the MLP:
ctx = get_forward_context()
steering = ctx.additional_kwargs.get("steering") if ctx.additional_kwargs else None
if steering and self.layer_idx in steering:
    vec, scale = steering[self.layer_idx]
    hidden_states = hidden_states + scale * vec
```

With AI coding tools, this is a tractable refactor - exactly the kind of repetitive, pattern-based change they excel at. The one-time cost pays off with:
- Full CUDA graph compatibility (significant throughput benefit)
- No runtime hook dispatch overhead
- Production-ready, maintainable solution

## SAE Feature Steering

For SAE-based steering, decode features back to activation space:

```python
class SAESteering:
    def __init__(self, sae_model, layer_idx: int):
        self.sae = sae_model  # Trained SAE for this layer
        self.layer_idx = layer_idx

    def compute_steering_vector(
        self,
        feature_idx: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Get steering vector for a specific SAE feature."""
        # SAE decoder column = direction in activation space
        return scale * self.sae.decoder.weight[:, feature_idx]

    def steer_features(
        self,
        feature_modifications: dict[int, float],  # feature_idx -> scale
    ) -> torch.Tensor:
        """Combine multiple feature steering vectors."""
        steering = torch.zeros(self.sae.hidden_size, device=self.sae.device)
        for feat_idx, scale in feature_modifications.items():
            steering += scale * self.sae.decoder.weight[:, feat_idx]
        return steering
```

### SAE Integration Example

```python
import torch
from vllm import LLM

class SAESteerableLLM:
    def __init__(self, model_name: str, sae_paths: dict[int, str]):
        """
        Args:
            model_name: HuggingFace model name
            sae_paths: {layer_idx: path_to_sae_weights}
        """
        self.llm = LLM(model=model_name)
        self.saes = {}
        for layer_idx, path in sae_paths.items():
            self.saes[layer_idx] = self._load_sae(path, layer_idx)

    def _load_sae(self, path: str, layer_idx: int):
        # Load your SAE model
        sae = torch.load(path)
        return SAESteering(sae, layer_idx)

    def steer_by_features(
        self,
        feature_interventions: dict[int, dict[int, float]],
        # {layer_idx: {feature_idx: scale}}
    ):
        """Set steering based on SAE features."""
        for layer_idx, features in feature_interventions.items():
            if layer_idx not in self.saes:
                raise ValueError(f"No SAE loaded for layer {layer_idx}")
            steering_vec = self.saes[layer_idx].steer_features(features)
            # Apply to model (implementation depends on approach chosen)
            self._set_layer_steering(layer_idx, steering_vec)
```

## Considerations

### CUDA Graphs Compatibility

vLLM uses CUDA graphs for performance. Steering vectors must:
- Be pre-allocated at graph capture time, OR
- Disable CUDA graphs for steered inference

```python
# Option 1: Pre-allocate steering buffer
self.steering_buffer = torch.zeros(hidden_size, device=device)

# Option 2: Disable CUDA graphs for steering
# In VllmConfig or engine args:
config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
```

### Batch Handling

When steering specific requests in a batch:

```python
def apply_selective_steering(
    hidden_states: torch.Tensor,      # [num_tokens, hidden_size]
    token_to_request: torch.Tensor,   # [num_tokens] -> request_idx
    request_steering: dict[int, torch.Tensor],  # request_idx -> vector
):
    for req_idx, vector in request_steering.items():
        mask = token_to_request == req_idx
        hidden_states[mask] = hidden_states[mask] + vector
    return hidden_states
```

### Performance Impact

| Aspect | Impact |
|--------|--------|
| Steering overhead | Minimal (~1 vector addition per steered layer) |
| Memory per layer | ~16KB for 4096 hidden size in fp32 |
| Latency | Negligible if vectors are on GPU |
| Throughput | No measurable impact |

### Steering Best Practices

1. **Normalize steering vectors**: Use unit vectors and control magnitude via scale
2. **Layer selection**: Middle layers (e.g., layers 12-20 in a 32-layer model) often work best
3. **Scale tuning**: Start with scale=1.0, adjust based on effect strength
4. **Position targeting**: For chat models, steer only assistant tokens

## Complete Example
2
```python
from vllm import LLM, SamplingParams
import torch

class SteerableLLM:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name)
        self.model = self._get_model()
        self.hidden_size = self.model.config.hidden_size

    def _get_model(self):
        return self.llm.llm_engine.model_executor.driver_worker.model_runner.model

    def add_steering_vector(
        self,
        layer_idx: int,
        direction: torch.Tensor,
        scale: float = 1.0,
    ):
        """Register a steering vector for a layer."""
        # Store on model for access during forward
        if not hasattr(self.model, '_steering_vectors'):
            self.model._steering_vectors = {}
        self.model._steering_vectors[layer_idx] = (direction.cuda(), scale)

    def generate_steered(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ):
        """Generate with currently registered steering vectors."""
        return self.llm.generate(prompts, sampling_params)

# Usage
llm = SteerableLLM("meta-llama/Llama-2-7b-hf")

# Create a "happiness" steering vector (e.g., from contrastive pairs or SAE)
happiness_direction = torch.randn(llm.hidden_size)  # Replace with real vector
happiness_direction = happiness_direction / happiness_direction.norm()

# Apply at layer 16
llm.add_steering_vector(layer_idx=16, direction=happiness_direction, scale=2.0)

# Generate with steering
outputs = llm.generate_steered(
    ["The weather today is"],
    SamplingParams(max_tokens=50)
)
```

## Related Resources

- [Steering API Implementation](API.md) - Exposing steering via the vLLM API
- [Activation Extraction](EXTRACTION.md) - Extracting activations for SAE training
- [Architecture Overview](design/arch_overview.md) - vLLM system architecture
- [HuggingFace Integration](design/huggingface_integration.md) - Model loading and configuration
