# Custom Model Integration Examples

This directory contains examples demonstrating how to integrate custom models with vLLM.

## Available Examples

### TorchTitan Integration

- **`deepseek_v3_torchtitan.py`**: DeepSeek V3 model using TorchTitan's implementation with vLLM's MLA attention
- **`qwen3_torchtitan.py`**: Qwen3 model using TorchTitan's implementation with vLLM's flash attention
- **`benchmark_deepseek_v3.py`**: Benchmark comparing custom DeepSeek V3 with vLLM's built-in implementation

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

### Benchmarking

Benchmark custom models against vLLM's built-in implementations:

```bash
# Benchmark custom model (default)
PYTHONPATH=/home/bwasti/oss/vllm python examples/custom_models/benchmark_deepseek_v3.py \
    --num-requests 10 \
    --max-batch-size 4 \
    --max-model-len 8192

# Benchmark vLLM's built-in DeepSeek V3
PYTHONPATH=/home/bwasti/oss/vllm python examples/custom_models/benchmark_deepseek_v3.py \
    --use-builtin \
    --num-requests 10 \
    --max-batch-size 4

# Compare both side-by-side
PYTHONPATH=/home/bwasti/oss/vllm python examples/custom_models/benchmark_deepseek_v3.py \
    --run-both \
    --num-requests 10 \
    --max-batch-size 4

# Full benchmark with TP=8
PYTHONPATH=/home/bwasti/oss/vllm python examples/custom_models/benchmark_deepseek_v3.py \
    --tp 8 \
    --num-requests 100 \
    --max-batch-size 32 \
    --max-model-len 8192
```

The benchmark measures:
- **Throughput**: Tokens generated per second
- **Requests/sec**: Request processing rate
- **Latency**: P50/P90/P99 latency distribution
- **Initialization time**: Model loading time

**Comparing Custom vs Built-in:**

Use `--run-both` to run both benchmarks sequentially and see a comparison table:

```bash
PYTHONPATH=/home/bwasti/oss/vllm python examples/custom_models/benchmark_deepseek_v3.py \
    --run-both \
    --num-requests 10
```

This will output:
```
Metric                    Custom               Built-in             Speedup
--------------------------------------------------------------------------------
Throughput (tok/s)        1028.11              945.23               1.09x
Requests/sec              8.03                 7.39                 1.09x
Avg Latency (ms)          124.50               135.23               1.09x
...
```

## Key Components

All examples use vLLM's custom model API:

- **`VLLMModelForCausalLM`**: Base class enforcing vLLM interface
- **`replace_with_trainable_attention()`**: Replace attention layers with vLLM's trainable attention
- **`load_external_weights()`**: Load weights with name mapping
- **`TrainableFlashAttention`** / **`TrainableMLA`**: vLLM's trainable attention implementations

See the [documentation](../../docs/source/contributing/model/custom.md) for more details.
