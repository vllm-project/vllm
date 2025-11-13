# Generic Model Support Tests

This directory contains tests for the generic model support RFC (#28326).

## Running the Tests

### Basic Tests (No vLLM imports)

The basic tests can run standalone without vLLM dependencies:

```bash
python tests/models/test_generic_models.py
```

### Full Test Suite with pytest

The full test suite requires proper vLLM setup. You may need to set `LD_PRELOAD` for cublas:

```bash
# On systems that require LD_PRELOAD for cublas (device-specific)
export LD_PRELOAD="/path/to/libcublasLt.so:/path/to/libcublas.so"
python -m pytest tests/models/test_generic_models.py -v
```

Or run specific tests:

```bash
export LD_PRELOAD="/path/to/libcublasLt.so:/path/to/libcublas.so"
python -m pytest tests/models/test_generic_models.py::TestGenericModelSupport::test_flash_attention_backward -v
```

### Test Coverage

The test suite includes:

1. **test_flash_attention_forward** - Validates flash attention forward pass
2. **test_flash_attention_backward** - Validates training with backward pass (**KEY TEST**)
3. **test_model_forward** - Tests full model forward pass
4. **test_model_training** - Tests end-to-end training loop (**KEY TEST**)
5. **test_vllm_wrapper** - Tests vLLM interface compatibility
6. **test_model_registration** - Tests model registration (requires vLLM setup, skipped by default)
7. **test_llm_api_integration** - Tests LLM() API integration (requires GPU, skipped by default)

## What These Tests Demonstrate

###  1. Flash Attention with Backward Pass

`VLLMFlashAttention` shows how vLLM can provide optimized modules that work for both inference and training:

- Forward pass uses PyTorch's flash attention when available
- Backward pass works automatically via autograd
- Can be used in standard training loops

### 2. Model Registration Pattern

`GenericTransformerForCausalLMVLLM` demonstrates the minimum interface needed for vLLM:

```python
class MyModelVLLM(nn.Module):
    # Required interface
    def __init__(self, vllm_config, prefix=""):
        ...

    def get_input_embeddings(self, input_ids):
        ...

    def forward(self, input_ids, positions):
        # Returns hidden states (not logits)
        ...

    def compute_logits(self, hidden_states):
        # Computes logits from hidden states
        ...

    def load_weights(self, weights):
        ...
```

### 3. End-to-End Workflow

```python
# 1. Train with PyTorch
model = GenericTransformerForCausalLM(config)
train(model)  # Uses VLLMFlashAttention with backward pass

# 2. Wrap for vLLM
vllm_model = GenericTransformerForCausalLMVLLM(config)

# 3. Register
ModelRegistry.register_model("MyModel", vllm_model)

# 4. Fast inference
llm = LLM(model="path/to/model", tensor_parallel_size=4)
outputs = llm.generate(prompts)
```

## Key Innovations

1. **Training-Compatible Modules** - vLLM modules work for both training and inference
2. **Minimal Interface** - Simple requirements to integrate custom models
3. **Backward Pass Support** - Full gradient computation for RL and fine-tuning
4. **Parallelism Ready** - Designed to support tensor/pipeline parallelism

## Related Files

- RFC: `RFC.md`
- Parallelism Examples: `examples/generic_model_parallelism.py`
- Documentation: `GENERIC_MODEL_TEST_README.md`, `IMPLEMENTATION_SUMMARY.md`
