# Custom Model Integration

This guide shows you how to integrate custom model implementations with vLLM using the Custom Model API. This is particularly useful when you want to:

- Use models from external libraries (e.g., TorchTitan, Megatron-LM)
- Bring your own parallelism implementation
- Integrate training-compatible models with vLLM's inference engine

## Overview

vLLM provides a comprehensive API in `vllm.model_executor.custom_models` for integrating external models:

```python
from vllm.model_executor.custom_models import (
    # Base wrapper class
    VLLMModelForCausalLM,

    # Attention modules
    TrainableFlashAttention,
    TrainableMLA,
    MLAConfig,
    replace_with_trainable_attention,

    # Utilities
    load_external_weights,
    convert_freqs_cis_to_real,
    store_positions_in_context,
    create_mla_kv_cache_spec,
)
```

## Quick Start

Here's a minimal example integrating a custom model:

```python
import torch.nn as nn
from vllm.model_executor.custom_models import (
    VLLMModelForCausalLM,
    replace_with_trainable_attention,
)
from vllm.model_executor.models import ModelRegistry

class MyCustomModelForCausalLM(VLLMModelForCausalLM):
    """vLLM-compatible wrapper for my custom model."""

    supports_pp = False  # Pipeline parallelism support
    supports_multimodal = False

    def __init__(self, vllm_config, parallel_context=None, **kwargs):
        super().__init__()
        assert vllm_config is not None, "vllm_config is required"

        # Import your external model
        from my_library import MyModel, MyModelConfig

        # Map HuggingFace config to your model's config
        hf_config = vllm_config.hf_config
        model_config = MyModelConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            # ... other config mapping
        )

        # Create your model
        self.model = MyModel(model_config)

        # Replace attention with vLLM's trainable attention
        replace_with_trainable_attention(self.model, use_mla=False)

    def get_input_embeddings(self, input_ids):
        return self.model.tok_embeddings(input_ids)

    def forward(self, input_ids, positions=None, **kwargs):
        # Your forward implementation
        return self.model(input_ids)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return self.model.output(hidden_states)

    def load_weights(self, weights_iter):
        # Load weights from HuggingFace checkpoint
        from vllm.model_executor.custom_models import load_external_weights

        name_mapping = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # ... your name mapping
        }

        loaded, skipped = load_external_weights(
            model=self.model,
            weights_iter=weights_iter,
            name_mapping=name_mapping
        )
        print(f"Loaded {loaded} parameters, skipped {skipped}")

def build_my_custom_model(vllm_config, parallel_context):
    """Factory function for vLLM's model registry."""
    model = MyCustomModelForCausalLM(
        vllm_config=vllm_config,
        parallel_context=parallel_context
    )
    return model

# Register with vLLM
ModelRegistry.register_model("MyCustomModel", build_my_custom_model)
```

## Core Components

### 1. VLLMModelForCausalLM

Abstract base class that enforces the vLLM interface:

```python
class VLLMModelForCausalLM(nn.Module, ABC):
    """Abstract base class for custom models."""

    supports_pp: bool = False  # Pipeline parallelism support
    supports_multimodal: bool = False  # Multimodal support

    @abstractmethod
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings."""
        pass

    @abstractmethod
    def forward(self, input_ids, positions=None, **kwargs) -> torch.Tensor:
        """Forward pass returning hidden states."""
        pass

    @abstractmethod
    def compute_logits(self, hidden_states, sampling_metadata=None) -> torch.Tensor:
        """Compute logits from hidden states."""
        pass

    @abstractmethod
    def load_weights(self, weights_iter):
        """Load weights from checkpoint."""
        pass
```

### 2. Attention Replacement

#### TrainableFlashAttention

Use for standard multi-head attention:

```python
from vllm.model_executor.custom_models import replace_with_trainable_attention

# Replace all attention layers in your model
replace_with_trainable_attention(model, use_mla=False)
```

This automatically replaces external attention layers with vLLM's `TrainableFlashAttention`, which:

- Supports PyTorch backward passes (for training/fine-tuning)
- Uses vLLM's optimized KV cache (for inference)
- Auto-registers for KV cache allocation

#### TrainableMLA

Use for Multi-Head Latent Attention (DeepSeek V3):

```python
replace_with_trainable_attention(model, use_mla=True)
```

Features:

- Low-rank compression for Q and KV projections
- Split Q/K into RoPE and non-RoPE parts
- Shared K_PE across all heads (memory efficient)

### 3. Weight Loading

The `load_external_weights()` utility handles weight loading with name mapping:

```python
from vllm.model_executor.custom_models import load_external_weights

# Define HuggingFace → Your Model name mapping
hf_to_model = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    # ... more mappings
}

loaded, skipped = load_external_weights(
    model=self.model,
    weights_iter=weights_iter,
    name_mapping=hf_to_model
)
```

Features:

- Automatic layer number substitution (`{}` in names)
- Returns counts of loaded and skipped parameters
- Handles both regular tensors and sharded weights

### 4. Utility Functions

#### Position Storage

Store vLLM's position indices for RoPE:

```python
from vllm.model_executor.custom_models import store_positions_in_context

def forward(self, input_ids, positions=None, **kwargs):
    # Store positions for attention layers
    store_positions_in_context(positions)

    # Your forward logic
    return self.model(input_ids)
```

#### RoPE Conversion

Convert complex RoPE frequencies to real format:

```python
from vllm.model_executor.custom_models import convert_freqs_cis_to_real

# Convert freqs_cis from complex to cos/sin format
self.model.freqs_cis = convert_freqs_cis_to_real(self.model.freqs_cis)
```

#### MLA KV Cache Spec

Create KV cache specification for MLA:

```python
from vllm.model_executor.custom_models import create_mla_kv_cache_spec

def get_kv_cache_spec(self, vllm_config):
    return create_mla_kv_cache_spec(
        kv_lora_rank=self.config.kv_lora_rank,
        qk_rope_head_dim=self.config.qk_rope_head_dim,
        block_size=vllm_config.cache_config.block_size,
        dtype=vllm_config.model_config.dtype,
    )
```

## Advanced Topics

### Tensor Parallelism

Integrate external tensor parallelism (e.g., TorchTitan, Megatron-LM):

```python
def build_my_model(vllm_config, parallel_context):
    # Create model
    model = MyModelForCausalLM(vllm_config=vllm_config, parallel_context=parallel_context)

    # Apply tensor parallelism if TP > 1
    if parallel_context is not None:
        tp_size = parallel_context.get_tensor_parallel_world_size()
        if tp_size > 1:
            # Use your external TP library
            from torch.distributed.device_mesh import init_device_mesh
            from my_library.parallelize import apply_tensor_parallel

            tp_mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
            apply_tensor_parallel(model.model, tp_mesh=tp_mesh)

    # Convert to dtype after TP
    if hasattr(vllm_config, "model_config"):
        model = model.to(dtype=vllm_config.model_config.dtype)

    return model
```

**Important**: Apply TP after model creation and attention replacement, but before dtype conversion.

### Multi-GPU Best Practices

1. **Deferred Imports**: Import external libraries inside `__init__` to avoid CUDA initialization before vLLM's multiprocessing fork:

```python
def __init__(self, vllm_config, **kwargs):
    super().__init__()

    # Import inside __init__, not at module level
    from torchtitan.models.qwen3 import Qwen3Model

    self.model = Qwen3Model(...)
```
2. **TP Application Order**:
    - ✅ Model creation → Attention replacement → TP → dtype conversion
    - ❌ Attention replacement → TP → Model creation (breaks weight sharding)
3. **Process Group Management**: Use `parallel_context` to get vLLM's process groups:

```python
tp_group = parallel_context.get_tp_process_group()
tp_size = parallel_context.get_tensor_parallel_world_size()
```

## Complete Examples

See the `examples/custom_models/` directory for complete examples:

- **DeepSeek V3**: TorchTitan model with MLA attention and TP
    - `examples/custom_models/deepseek_v3_torchtitan.py`

- **Qwen3**: TorchTitan model with flash attention and TP
    - `examples/custom_models/qwen3_torchtitan.py`

Each example demonstrates:

- Importing external model implementations
- Attention replacement
- Weight loading with name mapping
- Tensor parallelism integration
- Model registration

## Testing

Write pytest tests in `tests/custom_models/`:

```python
import pytest
from examples.custom_models.my_model import MyModelForCausalLM

def test_my_model_creation():
    """Test model creation."""
    # Your test logic
    model = MyModelForCausalLM(...)
    assert model is not None

def test_my_model_forward():
    """Test forward pass."""
    # Your test logic
    hidden_states = model(input_ids)
    assert hidden_states.shape == expected_shape
```

For multi-GPU tests, use pytest fixtures:

```python
@pytest.fixture(scope="module")
def distributed_setup():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    yield rank, world_size

    if "RANK" in os.environ:
        dist.destroy_process_group()
```

## Troubleshooting

### Import Errors

**Problem**: `ImportError: cannot import name 'TrainableFlashAttention'`

**Solution**: Update imports to use `vllm.model_executor.custom_models`:

```python
# ❌ Old
from vllm.model_executor.layers.trainable_attention import TrainableFlashAttention

# ✅ New
from vllm.model_executor.custom_models import TrainableFlashAttention
```

### CUDA Initialization Errors

**Problem**: `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

**Solution**: Defer imports to avoid CUDA init before fork:

```python
# ❌ Module-level import (initializes CUDA too early)
from torchtitan.models.qwen3 import Qwen3Model

class MyModel:
    def __init__(self, ...):
        self.model = Qwen3Model(...)

# ✅ Import inside __init__
class MyModel:
    def __init__(self, ...):
        from torchtitan.models.qwen3 import Qwen3Model
        self.model = Qwen3Model(...)
```

### TP Sharding Issues

**Problem**: Weights not sharded, all ranks have identical weights

**Solution**: Check TP application order:

```python
# ✅ Correct order
model = MyModel(...)  # 1. Create model
replace_with_trainable_attention(model.model, ...)  # 2. Replace attention
apply_tensor_parallel(model.model, ...)  # 3. Apply TP
model = model.to(dtype=...)  # 4. Convert dtype

# ❌ Wrong order (TP has no effect)
apply_tensor_parallel(model.model, ...)  # TP before model is ready
model = MyModel(...)
replace_with_trainable_attention(model.model, ...)
```

## Next Steps

- Review the [basic model guide](basic.md) for vLLM model fundamentals
- Check out [example implementations](../../examples/custom_models/README.md)
- Read about [model registration](registration.md)
- Learn about [testing models](tests.md)
