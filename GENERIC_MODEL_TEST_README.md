# Generic Model Support Test Suite

This test suite demonstrates the implementation of the RFC for vLLM support for generic model definitions.

## Overview

The RFC proposes two key features:

1. **vLLM-provided modules with training support**: A custom flash attention module with backward pass for training
2. **Parallelism setup**: User models can leverage vLLM's initialization functions with a callback-based approach

## Files

### 1. `tests/test_generic_model_support.py`

Comprehensive test suite demonstrating:

- **VLLMFlashAttention**: A flash attention module with both forward and backward passes
  - Forward pass uses PyTorch's `scaled_dot_product_attention` (which can use flash attention)
  - Backward pass is automatically handled by PyTorch autograd
  - Can be used in training loops with gradient computation

- **GenericTransformerForCausalLM**: A simple trainable transformer model
  - Built with vanilla PyTorch
  - Uses the vLLM flash attention module
  - Supports standard training with optimizers and backpropagation

- **GenericTransformerForCausalLMVLLM**: A wrapper for vLLM integration
  - Implements the minimum interface needed for vLLM
  - Can be registered with `ModelRegistry.register_model()`
  - Compatible with the `LLM()` API

#### Running the tests

```bash
# Run basic tests (no vLLM dependencies)
python tests/test_generic_model_support.py

# Run with pytest
pytest tests/test_generic_model_support.py -v

# Run specific tests
pytest tests/test_generic_model_support.py::TestGenericModelSupport::test_flash_attention_backward -v
```

### 2. `examples/generic_model_parallelism.py`

Demonstrates parallelism setup with examples of:

- **ParallelContext**: Mock of vLLM's parallelism context providing rank and world size information
- **ColumnParallelLinear**: Linear layer with column-wise tensor parallelism
- **RowParallelLinear**: Linear layer with row-wise tensor parallelism
- **ParallelTransformerAttention**: Attention layer using tensor parallel projections
- **ParallelMLP**: MLP layer using tensor parallel projections
- **UserModelBuilder**: Example of callback-based model registration

#### Running the examples

```bash
python examples/generic_model_parallelism.py
```

## Key Concepts

### 1. Flash Attention with Backward Pass

The `VLLMFlashAttention` module demonstrates how vLLM can provide optimized attention implementations that work for both inference and training:

```python
from test_generic_model_support import VLLMFlashAttention

# Create attention module
attn = VLLMFlashAttention(hidden_size=256, num_heads=4)

# Use in training
attn.train()
hidden_states = torch.randn(2, 16, 256, requires_grad=True)
output = attn(hidden_states)
loss = output.sum()
loss.backward()  # Backward pass works!
```

**Benefits:**
- Users get optimized attention (flash attention) for free
- Works seamlessly with PyTorch training loops
- No need to implement custom backward passes

### 2. Parallelism Callback Pattern

The proposed pattern allows users to leverage vLLM's parallelism setup:

```python
# User defines a model builder that receives parallel context
class MyModelBuilder:
    @staticmethod
    def build(config, parallel_context):
        # parallel_context provides:
        # - get_tensor_parallel_rank()
        # - get_pipeline_parallel_rank()
        # - get_tensor_parallel_world_size()
        # - etc.

        return MyModel(config, parallel_context)

# Register with vLLM
ModelRegistry.register_model(
    "MyCustomModel",
    MyModelBuilder.build,
)

# Use with LLM API
llm = LLM(
    model="path/to/model",
    tensor_parallel_size=4,  # vLLM handles process group setup
)
```

**Benefits:**
- vLLM handles process group initialization
- Users get tensor/pipeline parallelism without manual setup
- Works with vLLM's existing parallelism infrastructure

### 3. Model Registration

The test demonstrates the complete flow:

```python
# 1. Build a PyTorch model
model = GenericTransformerForCausalLM(config)

# 2. Train it (optional)
optimizer = torch.optim.Adam(model.parameters())
for batch in dataloader:
    loss = train_step(model, batch)
    loss.backward()
    optimizer.step()

# 3. Wrap for vLLM
vllm_model = GenericTransformerForCausalLMVLLM(config)

# 4. Register with vLLM
ModelRegistry.register_model("GenericTransformer", vllm_model)

# 5. Use with LLM API
llm = LLM(model="path/to/model")
outputs = llm.generate(["Hello world"])
```

## Test Results

All basic tests pass:

```
Testing Flash Attention Forward...
✓ Flash Attention Forward works

Testing Flash Attention Backward...
✓ Flash Attention Backward works

Testing Model Forward...
✓ Model Forward works

Testing Model Training...
✓ Model Training works

Testing vLLM Wrapper...
✓ vLLM Wrapper works

==================================================
All basic tests passed!
==================================================
```

## Next Steps

To fully implement the RFC, the following would be needed:

1. **Flash Attention Backend**:
   - Integrate actual flash attention kernels (FlashAttention-2, xformers, etc.)
   - Implement custom backward pass for specific optimizations
   - Add to `vllm/model_executor/layers/`

2. **Parallelism Utilities**:
   - Expose vLLM's parallel initialization to user callbacks
   - Provide utilities like `ColumnParallelLinear`, `RowParallelLinear`
   - Document the callback signature

3. **Model Registry Enhancements**:
   - Support callback-based registration
   - Pass `parallel_context` to user model builders
   - Add helper utilities for common patterns

4. **Documentation**:
   - Add user guide for custom model registration
   - Document minimum interface requirements
   - Provide more examples

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  User's PyTorch Model                   │
│  (Uses VLLMFlashAttention, ParallelLinear, etc.)        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ wraps
                        ▼
┌─────────────────────────────────────────────────────────┐
│              vLLM Integration Wrapper                   │
│  (Implements vLLM interface: forward, load_weights)     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ registers via
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  ModelRegistry                          │
│  ModelRegistry.register_model("MyModel", wrapper)       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ used by
                        ▼
┌─────────────────────────────────────────────────────────┐
│                     LLM() API                           │
│  llm = LLM(model="...", tensor_parallel_size=4)         │
└─────────────────────────────────────────────────────────┘
```

## Related Files

- RFC: `RFC.md`
- Model Registry: `vllm/model_executor/models/registry.py`
- Attention Backend: `vllm/attention/backends/abstract.py`

## Questions?

For questions or feedback on this RFC implementation, please refer to:
- RFC Issue: #28326
- Documentation: https://docs.vllm.ai/en/latest/contributing/model/basic/
