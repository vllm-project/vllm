# Custom Model Integration Examples

This directory contains examples demonstrating how to integrate custom models with vLLM.

## Available Examples

### TorchTitan Integration

- **`deepseek_v3_torchtitan.py`**: DeepSeek V3 model using TorchTitan's implementation with vLLM's MLA attention
- **`qwen3_torchtitan.py`**: Qwen3 model using TorchTitan's implementation with vLLM's flash attention

These examples show how to:
1. Import external model implementations (e.g., from TorchTitan)
2. Replace attention layers with vLLM's trainable attention
3. Register custom models with vLLM's model registry
4. Apply tensor parallelism for multi-GPU inference
5. Load weights from HuggingFace checkpoints

## Using These Examples

### With vLLM's LLM API

```python
from vllm import LLM

# Import and register your custom model first
from examples.custom_models import deepseek_v3_torchtitan  # noqa

# Create LLM with your custom model
llm = LLM(
    model="deepseek-ai/DeepSeek-V3-Base",
    trust_remote_code=True,
    tensor_parallel_size=8,
)

outputs = llm.generate(["Hello world!"])
```

### Standalone Testing

Each example can be run standalone for testing:

```bash
# Test DeepSeek V3
python examples/custom_models/deepseek_v3_torchtitan.py

# Test Qwen3
python examples/custom_models/qwen3_torchtitan.py
```

## Key Components

All examples use vLLM's custom model API:

- **`VLLMModelForCausalLM`**: Base class enforcing vLLM interface
- **`replace_with_trainable_attention()`**: Replace attention layers with vLLM's trainable attention
- **`load_external_weights()`**: Load weights with name mapping
- **`TrainableFlashAttention`** / **`TrainableMLA`**: vLLM's trainable attention implementations

See the [documentation](../../docs/source/contributing/model/custom.md) for more details.
