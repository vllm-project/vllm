# Pytest Marking Guide for vLLM

This guide explains how to mark tests with pytest marks for intelligent test selection in CI.

## Why Mark Tests?

Test marks enable:
1. **Intelligent CI test selection** - Run only relevant tests based on code changes
2. **Local development** - Run specific test categories during development
3. **Better test organization** - Clear categorization of what each test covers
4. **Reduced CI time** - Skip irrelevant tests while maintaining coverage

## Quick Start

```python
import pytest

# Single mark
@pytest.mark.attention
def test_flash_attention():
    ...

# Multiple marks for cross-cutting concerns
@pytest.mark.attention
@pytest.mark.distributed_comm
@pytest.mark.kernels
def test_distributed_flash_attention():
    ...
```

## Available Marks

### Core Functional Areas

| Mark | Description | Example Tests |
|------|-------------|---------------|
| `attention` | Attention mechanisms and backends | Flash attention, PagedAttention, MLA |
| `kernels` | CUDA/compute kernels | Custom CUDA kernels, Triton kernels |
| `quantization` | Quantization methods | FP8, INT8, GPTQ, AWQ, SmoothQuant |
| `compilation` | Torch.compile integration | Full-graph mode, inductor backends |
| `engine` | Engine core and scheduling | Request scheduling, batch management |
| `model_loading` | Model loading and weights | Weight loading, safetensors, sharding |
| `sampling` | Sampling algorithms | Top-k, top-p, temperature, logits processors |

### Distributed & Parallelism

| Mark | Description | Example Tests |
|------|-------------|---------------|
| `distributed_comm` | Communication primitives | NCCL, custom all-reduce, P2P ops |
| `tensor_parallel` | Tensor parallelism | TP sharding, column/row parallel |
| `pipeline_parallel` | Pipeline parallelism | PP stages, activation checkpointing |
| `expert_parallel` | Expert parallelism (MoE) | Expert load balancing, routing |
| `data_parallel` | Data parallelism | DP strategies, gradient sync |

### Model & Architecture Types

| Mark | Description | Example Tests |
|------|-------------|---------------|
| `models_language` | Language models | LLaMA, GPT, Mistral, Qwen |
| `models_multimodal` | Multimodal models | LLaVA, Qwen-VL, Whisper |
| `models_moe` | Mixture-of-Experts | Mixtral, DeepSeek-MoE |
| `lora` | LoRA adapters | LoRA inference, multi-adapter |

### API & Serving

| Mark | Description | Example Tests |
|------|-------------|---------------|
| `entrypoints` | API entrypoints | LLM class, AsyncLLMEngine |
| `openai_api` | OpenAI API compatibility | Chat completions, completions API |
| `offline_inference` | Offline/batch inference | Batch processing, throughput |
| `pooling` | Pooling/embedding endpoints | Embedding models, pooling modes |

### Advanced Features

| Mark | Description | Example Tests |
|------|-------------|---------------|
| `spec_decode` | Speculative decoding | Draft models, acceptance rate |
| `kv_cache` | KV cache management | Cache offloading, compression |
| `tool_calling` | Tool use and function calling | Function schemas, tool routing |
| `reasoning` | Reasoning models | Chain-of-thought, reasoning parsers |
| `structured_output` | Structured output | JSON mode, grammar constraints |
| `multimodal` | Multimodal processing | Image/audio processing, encoders |

### V1 Engine

| Mark | Description | Example Tests |
|------|-------------|---------------|
| `v1` | V1 engine implementation | V1-specific tests |
| `v1_core` | V1 core components | V1 scheduler, executor |
| `v1_distributed` | V1 distributed execution | V1 TP/PP implementation |

### Platform-Specific

| Mark | Description | Example Tests |
|------|-------------|---------------|
| `cuda` | CUDA-specific functionality | CUDA graphs, CUDA kernels |
| `rocm` | ROCm/AMD-specific | ROCm kernels, AMD optimizations |
| `tpu` | TPU-specific | TPU kernels, JAX integration |
| `cpu_test` | CPU-only tests (existing) | CPU backend tests |

### Test Categories

| Mark | Description | Example Tests |
|------|-------------|---------------|
| `correctness` | Correctness validation | Output verification, numerical accuracy |
| `benchmark` | Performance benchmarks | Throughput, latency measurements |
| `e2e` | End-to-end integration | Full stack integration tests |

## Marking Guidelines

### 1. Mark by Functionality, Not Directory

**❌ Bad** - Marking based on file location:
```python
# Just because it's in tests/kernels/ doesn't mean it only tests kernels
@pytest.mark.kernels
def test_attention_with_custom_kernel():
    # This also tests attention!
    ...
```

**✅ Good** - Mark based on what's actually tested:
```python
@pytest.mark.attention
@pytest.mark.kernels
def test_attention_with_custom_kernel():
    # Tests both attention behavior AND kernel implementation
    ...
```

### 2. Use Multiple Marks for Cross-Cutting Tests

Many tests cover multiple areas. Mark them all:

```python
@pytest.mark.quantization
@pytest.mark.models_language
@pytest.mark.correctness
def test_fp8_quantized_llama_accuracy():
    # Tests quantization + model implementation + correctness
    ...
```

### 3. Combine Functional + Infrastructure Marks

```python
@pytest.mark.distributed_comm  # Functional area
@pytest.mark.distributed       # Infrastructure (requires 2+ GPUs)
@pytest.mark.slow_test        # Infrastructure (long runtime)
def test_nccl_all_reduce_large_tensor():
    ...
```

### 4. Mark Test Files with `pytestmark`

For files where ALL tests share marks:

```python
# tests/kernels/attention/test_flash_attention.py
import pytest

# All tests in this file get these marks
pytestmark = [pytest.mark.attention, pytest.mark.kernels]

def test_flash_attention_forward():
    ...

def test_flash_attention_backward():
    ...
```

### 5. Model Tests Get Multiple Marks

```python
@pytest.mark.models_language      # Model type
@pytest.mark.core_model           # Run on every PR (existing)
@pytest.mark.quantization         # If testing quantized version
@pytest.mark.tensor_parallel      # If testing TP
def test_llama_tp2_fp8():
    ...
```

## Directory to Mark Mapping

This is a **starting guide** - always mark based on functionality, not just directory:

| Directory | Primary Mark(s) | Common Secondary Marks |
|-----------|----------------|------------------------|
| `tests/attention/` | `attention` | `kernels`, `distributed_comm` |
| `tests/kernels/` | `kernels` | `attention`, `quantization`, `cuda` |
| `tests/distributed/` | `distributed_comm` | `tensor_parallel`, `pipeline_parallel`, `expert_parallel` |
| `tests/quantization/` | `quantization` | `kernels`, `models_language` |
| `tests/compile/` | `compilation` | `kernels`, `attention` |
| `tests/lora/` | `lora` | `models_language`, `quantization` |
| `tests/models/language/` | `models_language` | `correctness`, various others |
| `tests/models/multimodal/` | `models_multimodal` | `correctness`, `multimodal` |
| `tests/entrypoints/openai/` | `openai_api`, `entrypoints` | `e2e` |
| `tests/v1/` | `v1` | `v1_core`, `v1_distributed`, etc. |
| `tests/basic_correctness/` | `correctness` | `e2e` |

## Running Tests Locally

```bash
# Run all attention tests
pytest -m attention

# Run attention tests, excluding slow ones
pytest -m "attention and not slow_test"

# Run distributed attention tests
pytest -m "attention and distributed_comm"

# Run all tests except quantization
pytest -m "not quantization"

# Run multiple categories
pytest -m "attention or kernels"

# Complex queries
pytest -m "models_language and core_model and not slow_test"
```

## CI Integration

The CI pipeline uses these marks to select tests based on changed files:

```yaml
# Example: If vllm/attention/ changed, run:
pytest -m "attention or kernels"

# If vllm/distributed/ changed, run:
pytest -m "distributed_comm or tensor_parallel or pipeline_parallel or expert_parallel"
```

## Migration Strategy

We're gradually marking all tests. Priority order:

1. ✅ **High-impact areas** - attention, distributed, kernels (mark first)
2. **Model tests** - language and multimodal models
3. **API tests** - entrypoints and OpenAI API
4. **V1 engine** - V1 implementation tests
5. **Remaining tests** - quantization, compilation, etc.

### For New Tests

**Always mark new tests** from day one:

```python
@pytest.mark.attention
@pytest.mark.kernels
def test_new_attention_kernel():
    ...
```

### For Existing Tests

When you modify an existing test, add appropriate marks:

```python
# Before
def test_flash_attention():
    ...

# After
@pytest.mark.attention
@pytest.mark.kernels
def test_flash_attention():
    ...
```

## Mark Validation

We provide a validation script to check mark coverage:

```bash
# Check which tests are missing marks
python .buildkite/scripts/validate_test_marks.py

# Check marks for specific directory
python .buildkite/scripts/validate_test_marks.py tests/attention/
```

## Best Practices

1. **Be specific** - Use the most specific marks that apply
2. **Be complete** - Mark all relevant functional areas
3. **Update marks** - When test scope changes, update marks
4. **Check marks** - Before submitting PR, ensure tests are marked
5. **Review marks** - During code review, verify mark appropriateness

## Examples from Real Tests

### Simple kernel test
```python
@pytest.mark.kernels
@pytest.mark.attention
def test_flash_attention_kernel_forward():
    ...
```

### Distributed model test
```python
@pytest.mark.models_language
@pytest.mark.tensor_parallel
@pytest.mark.distributed_comm
@pytest.mark.distributed  # Requires 2+ GPUs (infrastructure)
def test_llama_tp2():
    ...
```

### Multimodal with LoRA
```python
@pytest.mark.models_multimodal
@pytest.mark.lora
@pytest.mark.multimodal
@pytest.mark.correctness
def test_llava_with_lora_adapters():
    ...
```

### V1 speculative decoding
```python
@pytest.mark.v1
@pytest.mark.spec_decode
@pytest.mark.e2e
def test_v1_eagle_speculative_decoding():
    ...
```

## Questions?

- See existing marked tests for examples
- Check `pyproject.toml` for full mark definitions
- Ask in #vllm-ci Slack channel
