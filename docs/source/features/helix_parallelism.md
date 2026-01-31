# Helix Parallelism

Helix parallelism is an advanced decode context parallelism strategy that uses All-to-All communication with LSE-weighted reduction instead of the standard AllGather+ReduceScatter pattern.

## Overview

Helix decouples attention parallelism (TPA) from FFN parallelism (TP):
- **TPA (Tensor Parallel for Attention)**: How attention heads are distributed
- **KVP (KV Parallel)**: How sequence is sharded for decode context parallelism
- **TP**: Standard tensor parallelism for FFN layers

**Key benefit**: Eliminates KV cache duplication for long-context decoding scenarios.

## Modes of Operation

### 1. Standard DCP (`--helix-mode` NOT set)

Standard decode context parallelism using AllGather Q followed by AllGather+ReduceScatter for output combination.

```
Communication: AllGather Q → Compute → AllGather+ReduceScatter
```

### 2. Helix MLA (`--helix-mode` with MLA models)

For MLA (Multi-head Latent Attention) models like DeepSeek-V2, TPA=1 because there's effectively one KV head (latent).

```
TPA = TP / DCP = 1 (when DCP = TP)
KVP = DCP

Communication: AllGather Q → Compute → All-to-All + LSE Reduce
```

### 3. Helix GQA (`--helix-mode` with GQA models)

For GQA (Grouped Query Attention) models like Llama and Qwen, TPA > 1 allows head sharding.

```
TPA = TP / DCP > 1
KVP = DCP

Communication: Local Q (no AllGather) → Compute → All-to-All + LSE Reduce
```

## Code Path Analysis

### Configuration Flow

```
helix_mode=False                    → Standard DCP
helix_mode=True + use_mla=True      → Helix MLA (TPA=1)
helix_mode=True + use_mla=False     → Helix GQA (TPA>1)
```

### Detailed Code Paths

#### Standard DCP (`helix_mode=False`)

| Component | Behavior |
|-----------|----------|
| Groups | DCP group only |
| `is_helix_gqa_mode()` | False |
| `get_attention_tp_world_size()` | Returns full TP size |
| `get_attention_tp_rank()` | Returns full TP rank |
| Flash Attn | AllGather Q → `cp_lse_ag_out_rs()` |
| FlashInfer | AllGather Q → `cp_lse_ag_out_rs()` |
| MLA Backend | AllGather Q → `cp_lse_ag_out_rs()` |
| QKVParallelLinear | Standard TP distribution |
| Attention layer | `num_output_heads = num_heads` |

#### Helix MLA (`helix_mode=True`, TPA=1)

| Component | Behavior |
|-----------|----------|
| Groups | DCP group, `_HELIX_KVP=None` (falls back to DCP) |
| `is_helix_gqa_mode()` | False |
| `get_attention_tp_world_size()` | Returns full TP size |
| `get_attention_tp_rank()` | Returns full TP rank |
| Flash Attn | AllGather Q → `helix_alltoall_lse_reduce()` |
| FlashInfer | AllGather Q → `helix_alltoall_lse_reduce()` |
| MLA Backend | AllGather Q → `helix_alltoall_lse_reduce()` |
| QKVParallelLinear | Standard TP (MLA has different architecture) |
| Attention layer | `num_output_heads = num_heads` |

#### Helix GQA (`helix_mode=True`, TPA>1)

| Component | Behavior |
|-----------|----------|
| Groups | DCP group + `_HELIX_KVP` group |
| `is_helix_gqa_mode()` | True |
| `get_attention_tp_world_size()` | Returns TPA size |
| `get_attention_tp_rank()` | Returns TPA rank (`tp_rank // kvp_size`) |
| Flash Attn | Local Q (no AllGather) → `helix_alltoall_lse_reduce()` + head scatter |
| FlashInfer | Local Q (no AllGather) → `helix_alltoall_lse_reduce()` + head scatter |
| QKVParallelLinear | TPA-based distribution (KVP ranks share weights) |
| LlamaAttention | TPA-based head distribution |
| Attention layer | `num_output_heads = num_heads // KVP_SIZE` |

## Usage

### Basic Usage

```bash
# Standard DCP (baseline)
vllm serve <model> \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2

# Helix mode
vllm serve <model> \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2 \
    --helix-mode
```

### GQA Model (e.g., Llama, Qwen, Nemotron)

```bash
# TP=8, DCP=2 → TPA=4, KVP=2
vllm serve nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2 \
    --helix-mode \
    --cp-kv-cache-interleave-size 16
```

### MLA Model (e.g., DeepSeek-V2)

```bash
# TP=8, DCP=8 → TPA=1, KVP=8
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8 \
    --helix-mode \
    --attention-backend FLASHMLA \
    --cp-kv-cache-interleave-size 64 \
    --trust-remote-code
```

## Constraints

### TPA Constraints (GQA models)

1. **TPA ≤ K**: TPA must not exceed the number of KV heads
2. **K % TPA == 0**: KV heads must be evenly divisible by TPA
3. **Q % TPA == 0**: Q heads must be evenly divisible by TPA

### MLA Constraint

- TPA=1 is only valid for MLA models (where effective K=1)
- GQA models cannot use TPA=1 due to Q-KV head binding issues

## Testing

### Run Functional Tests

```bash
# All functional tests
pytest tests/distributed/test_helix_functional.py -v -s

# Specific test classes
pytest tests/distributed/test_helix_functional.py::TestHelixGQA -v -s
pytest tests/distributed/test_helix_functional.py::TestHelixMLA -v -s
pytest tests/distributed/test_helix_functional.py::TestStandardDCP -v -s
pytest tests/distributed/test_helix_functional.py::TestHelixVsDCPConsistency -v -s

# Quick smoke test
pytest tests/distributed/test_helix_functional.py::TestQuickSmoke -v -s
```

### Run Unit Tests (No GPU Required)

```bash
pytest tests/distributed/test_helix_config.py -v
```

### Run Integration Tests (GSM8K Evaluation)

```bash
pytest tests/distributed/test_helix_parallel.py -v -s
```

## Requirements

- 4+ GPUs with compute capability 9.0+ (Hopper/Blackwell)
- vLLM with Helix support
- For MLA models: `--trust-remote-code` flag
