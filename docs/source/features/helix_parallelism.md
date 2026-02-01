# Helix Parallelism

Helix parallelism is an advanced decode context parallelism strategy that uses All-to-All communication with LSE-weighted reduction instead of the standard AllGather+ReduceScatter pattern.

## Overview

Helix decouples attention parallelism (TPA) from FFN parallelism (TP):
- **TPA (Tensor Parallel for Attention)**: How attention heads are distributed
- **KVP (KV Parallel)**: How sequence is sharded for decode context parallelism
- **TP**: Standard tensor parallelism for FFN layers

**Key benefit**: Eliminates KV cache duplication for long-context decoding scenarios.

## Backend Compatibility

### GQA Models (Llama, Nemotron, Qwen, etc.)

| Backend | Standard TP | Standard DCP | Helix GQA |
|---------|-------------|--------------|-----------|
| FLASH_ATTN | ✓ | ✓ | ✓ |
| FLASHINFER | ✓ | ✓ | ✗ (known limitation) |

**Note**: FlashInfer + Helix GQA produces incorrect output. Use `--attention-backend FLASH_ATTN` for Helix GQA.

### MLA Models (DeepSeek, etc.)

| Backend | Hopper | Blackwell | DCP Support | Helix MLA |
|---------|--------|-----------|-------------|-----------|
| FLASHMLA | ✓ | ✗ | ✓ | ✓ |
| CUTLASS_MLA | ✓ | ✓ | ✓ | ✓ |
| FLASHINFER_MLA | ✓ | ✓ | ✗ (no LSE) | ✗ |

**Note**: On Blackwell with DCP enabled, vLLM auto-selects CUTLASS_MLA (FlashInferMLA lacks LSE support required for DCP).

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

**Important**: Use `--attention-backend FLASH_ATTN` for Helix GQA. FlashInfer is not supported.

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
# Must use FLASH_ATTN (FlashInfer not supported for Helix GQA)
vllm serve nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2 \
    --helix-mode \
    --attention-backend FLASH_ATTN \
    --cp-kv-cache-interleave-size 16
```

### MLA Model (e.g., DeepSeek-V2)

```bash
# TP=8, DCP=8 → TPA=1, KVP=8
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8 \
    --helix-mode \
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

### Backend Constraints

- **Helix GQA**: Requires `FLASH_ATTN` backend (FlashInfer not supported)
- **Helix MLA on Blackwell**: Auto-selects `CUTLASS_MLA` (FlashInferMLA lacks DCP support)

## Known Limitations

1. **FlashInfer + Helix GQA**: Produces incorrect output. Root cause under investigation. Workaround: use `--attention-backend FLASH_ATTN`.

2. **Standard DCP + GQA**: Requires `TP > num_kv_heads`. For example, Llama-3.1-8B has 8 KV heads, so standard DCP needs TP > 8 (i.e., 16+ GPUs). Helix removes this constraint.

3. **FlashInferMLA + DCP**: FlashInferMLA doesn't return LSE for decode, which is required for DCP. On Blackwell, vLLM auto-selects CUTLASS_MLA when DCP is enabled.

## Testing

### Run Unit Tests (No GPU Required)

```bash
pytest tests/distributed/test_helix_config.py -v
```

### Standalone Functional Test Script

For comprehensive functional testing, use the standalone test script in Docker:

```bash
# Run all tests
./test_helix.sh --all

# Run specific suite
./test_helix.sh --suite helix-gqa
./test_helix.sh --suite helix-mla

# List available suites
./test_helix.sh --list
```

## Requirements

- 4+ GPUs with compute capability 9.0+ (Hopper/Blackwell)
- vLLM with Helix support
- For MLA models: `--trust-remote-code` flag
- For Helix GQA: `--attention-backend FLASH_ATTN` (FlashInfer not supported)
