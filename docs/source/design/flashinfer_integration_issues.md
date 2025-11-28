# FlashInfer Integration Issues in vLLM

This document details the current state of FlashInfer integration in vLLM, including known issues, broken functionality, and recommendations for the FlashInfer team.

**Environment tested:**
- vLLM: v0.8.3rc2.dev6045
- FlashInfer: 0.5.2, 0.5.3 (both tested)
- flashinfer-cubin: 0.5.2, 0.5.3 (both tested)
- PyTorch: 2.9.0
- CUDA: 12.8 (runtime), FlashInfer packages compiled against CUDA ≤12.6
- Hardware: NVIDIA H200 (SM 9.0 / Hopper)

**Summary of issues by FlashInfer version:**

| Issue | FlashInfer 0.5.2 | FlashInfer 0.5.3 | Fixed? |
|-------|------------------|------------------|--------|
| AllReduce Fusion JIT (`std::optional` bug) | ❌ Broken | ❌ Broken | **No** |
| FP8 MoE CUDA Version (needs 12.7+) | ❌ Broken | ❌ Broken | **No** |
| MXFP4 MoE on SM90 | ❌ Broken | ❌ Broken | **No** |
| Attention Sinks on Hopper | ❌ Not supported | ❌ Not supported | **No** |

---

## Table of Contents

1. [Overview of FlashInfer Integration](#overview-of-flashinfer-integration)
2. [Working Features](#working-features)
3. [Known Issues](#known-issues)
   - [Issue 1: AllReduce Fusion JIT Compilation Failure](#issue-1-allreduce-fusion-jit-compilation-failure)
   - [Issue 2: FP8 MoE CUDA Version Mismatch](#issue-2-fp8-moe-cuda-version-mismatch)
   - [Issue 3: MXFP4 MoE SM90 Backend Broken](#issue-3-mxfp4-moe-sm90-backend-broken)
   - [Issue 4: Attention Sinks Not Supported on Hopper](#issue-4-attention-sinks-not-supported-on-hopper)
4. [Skipped Tests](#skipped-tests)
5. [Environment Variables Reference](#environment-variables-reference)
6. [Missing Operators for Complete FlashInfer Backend](#missing-operators-for-complete-flashinfer-backend)
7. [Recommendations for FlashInfer Team](#recommendations-for-flashinfer-team)

---

## Overview of FlashInfer Integration

vLLM integrates FlashInfer for multiple operators:

| Operator | Environment Variable | Status on Hopper (SM90) | Status on Blackwell (SM100) |
|----------|---------------------|------------------------|----------------------------|
| Attention | `VLLM_ATTENTION_BACKEND=FLASHINFER` | ✅ Works (no sinks) | ✅ Works |
| Attention with Sinks | - | ❌ Needs TRTLLM | ✅ Via TRTLLM |
| Top-k/Top-p Sampling | `VLLM_USE_FLASHINFER_SAMPLER` | ✅ Works | ✅ Works |
| RMSNorm | `VLLM_USE_FLASHINFER_NORM` | ✅ Works | ✅ Works |
| Activations (SiLU, GELU) | `VLLM_USE_FLASHINFER_ACTIVATION` | ✅ Works | ✅ Works |
| MoE FP16/BF16 | `VLLM_USE_FLASHINFER_MOE_FP16` | ✅ Works | ✅ Works |
| MoE FP8 | `VLLM_USE_FLASHINFER_MOE_FP8` | ❌ CUDA version issue | ✅ Works |
| MoE MXFP4 | `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | ❌ Broken | ✅ Works |
| AllReduce Fusion | `VLLM_USE_FLASHINFER_ALLREDUCE` | ❌ JIT failure | ❌ JIT failure |
| All2All | `VLLM_ALL2ALL_BACKEND=flashinfer_all2allv` | ✅ Works | ✅ Works |

---

## Working Features

These FlashInfer features work correctly on Hopper (H100/H200):

### 1. Attention (without sinks)
- File: `vllm/v1/attention/backends/flashinfer.py`
- Works for standard models like Llama, Qwen without attention sinks

### 2. Sampling
- File: `vllm/v1/sample/ops/topk_topp_sampler.py`
- Uses `flashinfer.sampling.top_k_top_p_sampling_from_probs`

### 3. RMSNorm
- File: `vllm/model_executor/layers/layernorm.py`
- Uses `flashinfer.norm.rmsnorm` and `flashinfer.norm.fused_add_rmsnorm`
- **Note:** `fused_add_rmsnorm` is in-place and returns `None`. The vLLM integration must return `(x, residual)` after the call.

### 4. Activations
- File: `vllm/model_executor/layers/activation.py`
- Uses `flashinfer.activation.silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`

### 5. MoE FP16/BF16
- File: `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
- Uses FlashInfer CUTLASS MoE for unquantized models

---

## Known Issues

### Issue 1: AllReduce Fusion JIT Compilation Failure

**Severity:** High  
**Affected versions:** FlashInfer 0.5.2, 0.5.3 (including with cubins installed)  
**Environment variable:** `VLLM_USE_FLASHINFER_ALLREDUCE`  
**Compilation pass:** `enable_fi_allreduce_fusion`

#### Symptom

When enabling `enable_fi_allreduce_fusion=True` in vLLM's compilation config, JIT compilation fails with:

```
/data-fast/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh(487): error: namespace "std" has no member "optional"
/data-fast/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh(487): error: identifier "batchIdx" is undefined
/data-fast/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh(707): error: identifier "AllReduceFusionPattern" is undefined
```

#### Root Cause: C++ Namespace Mismatch in CUDA Header

**File:** `flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh`

The header file has a namespace mismatch:

```cpp
// Line 10 - includes CUDA's libcudacxx optional
#include <cuda/std/optional>

// Line 487-489 - uses std::optional (wrong namespace!)
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    std::optional<int> batchIdx,  // ERROR: should be cuda::std::optional
    int rowIdx,
    int colIdx, 
    std::optional<int> numRows,   // ERROR: should be cuda::std::optional
    ...
)
```

**The problem:** CUDA's libcudacxx uses `cuda::std::optional`, not `std::optional`. When NVCC compiles this header, it cannot find `std::optional` because `<optional>` (the C++17 standard library header) is not included.

This causes a cascade of errors:
1. `std::optional` is undefined → compilation fails at line 487
2. `batchIdx` parameter becomes undefined 
3. All subsequent code using `AllReduceFusionPattern` (defined at line 690) fails because compilation never reaches that point

#### Installing cubins Does NOT Fix This

Even with `flashinfer-cubin==0.5.2` installed, the issue persists because:
- Cubins provide pre-compiled kernels for specific operations
- The JIT compilation wrapper still needs to compile code that includes the broken header
- The header file bug affects all code paths that include it

#### Code Path

```
vllm/config/compilation.py
  -> PassConfig(enable_fi_allreduce_fusion=True)
  -> vllm/compilation/pass_manager.py:103
  -> AllReduceFusionPass(config)
  -> vllm/compilation/collective_fusion.py
  -> call_trtllm_fused_allreduce_norm()
  -> flashinfer.comm.trtllm_allreduce_fusion()
  -> JIT compilation of wrapper code
  -> #include "flashinfer/comm/trtllm_allreduce_fusion.cuh"
  -> COMPILATION FAILS
```

#### How to Reproduce

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

# Enable the allreduce fusion pass
pass_config = PassConfig(
    enable_fi_allreduce_fusion=True,
    enable_noop=True,
)
compilation_config = CompilationConfig(pass_config=pass_config)

# This will fail during model loading
llm = LLM(
    model='Qwen/Qwen3-30B-A3B-Instruct-2507',
    tensor_parallel_size=2,
    compilation_config=compilation_config,
)
```

#### Workaround

vLLM does NOT auto-enable this feature even when `VLLM_USE_FLASHINFER=1` is set:

```python
# vllm/envs.py - VLLM_USE_FLASHINFER_ALLREDUCE does NOT fallback to VLLM_USE_FLASHINFER
"VLLM_USE_FLASHINFER_ALLREDUCE": lambda: bool(
    int(os.getenv("VLLM_USE_FLASHINFER_ALLREDUCE", "0"))
),
```

**Note:** Running vLLM in eager mode (`enforce_eager=True`) does NOT help avoid this issue. The `enable_fi_allreduce_fusion` pass modifies the model's forward pass in a way that expects torch.compile to perform the actual fusion. Without compilation, the modified code path breaks.

#### Fix Required in FlashInfer

**Option A (Preferred):** Change `std::optional` to `cuda::std::optional` in the header:

```cpp
// Line 487-489 - fix namespace
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    cuda::std::optional<int> batchIdx,  // FIXED
    int rowIdx,
    int colIdx, 
    cuda::std::optional<int> numRows,   // FIXED
    ...
)
```

**Option B:** Add a using declaration at the top of the file:

```cpp
#include <cuda/std/optional>
// Add this line:
namespace std { using cuda::std::optional; }
```

**Option C:** Include standard C++17 `<optional>` and ensure compilation with `-std=c++17`:

```cpp
#include <optional>  // C++17 standard library
```

---

### Issue 2: FP8 MoE CUDA Version Mismatch

**Severity:** Medium  
**Affected versions:** FlashInfer 0.5.2, 0.5.3 (both compiled against CUDA ≤12.6)  
**Affected models:** Qwen3-30B-A3B-Instruct-2507-FP8, other FP8 MoE models  
**Environment variable:** `VLLM_USE_FLASHINFER_MOE_FP8`

#### Symptom

```python
NotImplementedError: FP8 block scaling not implemented for CUDA 12.6 or lower.
```

#### Root Cause

FlashInfer's FP8 MoE kernel requires CUDA 12.7+, but both FlashInfer 0.5.2 and 0.5.3 packages were compiled against CUDA ≤12.6. Even though the host system has CUDA 12.8, FlashInfer checks its own compile-time CUDA version.

**Tested:** Issue persists in both FlashInfer 0.5.2 and 0.5.3.

#### Code Path

```
vllm/model_executor/layers/fused_moe/fused_moe.py
  -> FusedMoE.forward_impl()
  -> FlashInfer FP8 MoE kernel
  -> Runtime check fails
```

#### Detection Methods

The runtime CUDA version detection is inconsistent:
- `torch.cuda.runtime_version()` returns 12080 (12.8.0)
- `torch.version.cuda` returns "12.8"
- FlashInfer internal check returns CUDA 12.6 (compile-time)

#### Workaround

Skip FP8 MoE tests on systems where FlashInfer was compiled against CUDA < 12.7.

#### Recommendation for FlashInfer Team

1. Document the minimum CUDA version requirement for FP8 MoE clearly
2. Consider providing wheels compiled against different CUDA versions
3. Improve the error message to indicate it's a compile-time vs runtime version mismatch

---

### Issue 3: MXFP4 MoE SM90 Backend Broken

**Severity:** High  
**Affected versions:** FlashInfer 0.5.2, 0.5.3 (SM90 kernels not implemented)  
**Affected models:** GPT-OSS-120B and other MXFP4 models on Hopper  
**Environment variable:** `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16`

#### Symptom

```
File "flashinfer/data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_sm100_binding.cu", line 214
RuntimeError: Check failed: (false) is false: Could not construct fused moe op with the requested 
input combination Activation: bfloat16, Weight: uint8, Output: bfloat16
```

#### Root Cause

vLLM's `mxfp4.py` has a backend called `SM90_FI_MXFP4_BF16` that claims to support FlashInfer MXFP4 MoE on Hopper (SM90):

```python
# vllm/model_executor/layers/quantization/mxfp4.py:96-102
if (
    current_platform.is_device_capability(90)  # Hopper
    and has_flashinfer()
    and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16
):
    logger.info_once("Using FlashInfer MXFP4 BF16 backend for SM90")
    return Mxfp4Backend.SM90_FI_MXFP4_BF16
```

However, when this backend is selected, the code path leads to `flashinfer_cutlass_fused_moe` which only has SM100 implementations:
- The CUDA file is literally named `flashinfer_cutlass_fused_moe_sm100_binding.cu`
- There is no corresponding `sm90_binding.cu` file

#### Code Path

```
vllm/model_executor/layers/quantization/mxfp4.py
  -> get_mxfp4_backend() returns SM90_FI_MXFP4_BF16
  -> Mxfp4MoEMethod.apply()
  -> flashinfer_cutlass_fused_moe()
  -> flashinfer/fused_moe/core.py:cutlass_fused_moe()
  -> get_cutlass_fused_moe_module(device_arch)
  -> Loads sm100_binding.cu on SM90 hardware
  -> Kernel initialization fails
```

#### Workaround

Disable FlashInfer MXFP4 MoE on Hopper to fall back to Marlin/Triton:

```bash
export VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=0
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=0
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=0
```

#### Recommendation for FlashInfer Team

1. **Option A:** Implement SM90 CUTLASS kernels for MXFP4 MoE
2. **Option B:** Remove/gate the SM90 code path in vLLM if FlashInfer doesn't plan to support it
3. Add architecture detection in `get_cutlass_fused_moe_module()` to fail gracefully on unsupported architectures

---

### Issue 4: Attention Sinks Not Supported on Hopper

**Severity:** Medium  
**Affected versions:** FlashInfer 0.5.2, 0.5.3 (requires TRTLLM on SM100)  
**Affected models:** GPT-OSS-120B and other models using attention sinks  
**File:** `vllm/v1/attention/backends/flashinfer.py`

#### Symptom

```
ValueError: Selected backend AttentionBackendEnum.FLASHINFER is not valid for this configuration. 
Reason: ['sink setting not supported']
```

#### Root Cause

FlashInfer's attention sink support requires TRTLLM attention, which is only available on Blackwell (SM100):

```python
# vllm/v1/attention/backends/flashinfer.py:353-366
@classmethod
def supports_sink(cls) -> bool:
    """FlashInfer supports sinks when TRTLLM attention is available (SM100)."""
    from vllm.utils.flashinfer import (
        force_use_trtllm_attention,
        supports_trtllm_attention,
    )
    if force_use_trtllm_attention() is False:
        return False
    return supports_trtllm_attention()

# vllm/utils/flashinfer.py:257-267
def supports_trtllm_attention() -> bool:
    if vllm_is_batch_invariant():
        return False
    # Requires SM100 and NVIDIA artifactory to be accessible
    return current_platform.is_device_capability(100) and has_nvidia_artifactory()
```

#### Workaround

Use FlashAttention3 (`FLASH_ATTN`) for attention on Hopper when using models with attention sinks:

```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### Recommendation for FlashInfer Team

1. Consider implementing attention sink support for SM90 without requiring TRTLLM
2. Document that sink support requires SM100 + TRTLLM

---

## Skipped Tests

### Test: Qwen3-30B-A3B-Instruct-2507-FP8

**Reason:** FlashInfer FP8 MoE requires CUDA 12.7+, but FlashInfer 0.5.2 and 0.5.3 were both compiled against CUDA ≤12.6

```python
# tests/kernels/run_flashinfer_test.py
if model_name == "qwen":
    print(f"\n⚠ Skipping {result_name}: FlashInfer FP8 MoE kernel requires "
          "CUDA 12.7+ (FlashInfer package compiled against older CUDA)")
    results[result_name] = RESULT_SKIPPED
    continue
```

### Test: GPT-OSS-120B on Hopper (partial skip)

**Reason:** FlashInfer MXFP4 MoE and FlashInfer attention (with sinks) not supported on SM90

The test runs but uses fallback backends:
- Attention: FlashAttention3 instead of FlashInfer
- MoE: Triton/Marlin instead of FlashInfer CUTLASS

---

## Environment Variables Reference

### Master Switch

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_USE_FLASHINFER` | Enable FlashInfer for all supported operators | `0` |

### Individual Feature Flags

| Variable | Description | Auto-enabled by master? | Notes |
|----------|-------------|------------------------|-------|
| `VLLM_ATTENTION_BACKEND` | Attention backend selection | Yes → `FLASHINFER` | |
| `VLLM_USE_FLASHINFER_SAMPLER` | Top-k/Top-p sampling | Yes | |
| `VLLM_USE_FLASHINFER_NORM` | RMSNorm | Yes | |
| `VLLM_USE_FLASHINFER_ACTIVATION` | SiLU/GELU activations | Yes | |
| `VLLM_USE_FLASHINFER_MOE_FP16` | FP16/BF16 MoE | Yes | |
| `VLLM_USE_FLASHINFER_MOE_FP8` | FP8 MoE | Yes | Requires CUDA 12.7+ |
| `VLLM_USE_FLASHINFER_MOE_FP4` | FP4 MoE (NVFP4) | Yes | |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_BF16` | MXFP4 MoE with BF16 activation | Yes | **Broken on SM90** |
| `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` | MXFP4 MoE with MXFP8 activation | Yes | SM100 only |
| `VLLM_USE_FLASHINFER_ALLREDUCE` | AllReduce fusion | **No** | JIT compilation broken |
| `VLLM_ALL2ALL_BACKEND` | All2All communication backend | Yes → `flashinfer_all2allv` | |

---

## Missing Operators for Complete FlashInfer Backend

For vLLM to use FlashInfer as a **complete backend** without relying on vLLM's custom CUDA kernels, FlashInfer would need to implement the following operators:

### Critical (Required for basic inference)

| Operator | Current Status | vLLM Implementation |
|----------|---------------|---------------------|
| **Embedding** | ❌ Not available | `vllm/model_executor/layers/vocab_parallel_embedding.py` uses PyTorch's `F.embedding` |
| **Dense Linear (FP16/BF16)** | ⚠️ Partial | FlashInfer has FP8/FP4 GEMM but standard linear uses cuBLAS/Triton |
| **LayerNorm** | ✅ Available | `flashinfer.norm.layernorm` (in addition to RMSNorm) |
| **Softmax** | ✅ Available | `flashinfer.sampling.softmax` |
| **RoPE** | ✅ Available | `flashinfer.rope.*` |

### Quantization Kernels (Required for quantized models)

| Quantization | Current Status | Notes |
|--------------|---------------|-------|
| **AWQ** | ❌ Not available | vLLM uses Marlin/custom kernels |
| **GPTQ** | ❌ Not available | vLLM uses Marlin/ExLlama/custom kernels |
| **GGUF** | ❌ Not available | vLLM uses custom kernels |
| **Marlin** | ❌ Not available | vLLM's optimized W4A16 kernels |
| **FP8 (compressed-tensors)** | ⚠️ Partial | FlashInfer has FP8 GEMM but not all compression schemes |
| **W8A8** | ⚠️ Partial | FlashInfer has `mm_fp8` but not CUTLASS W8A8 |

### Architecture-Specific Kernels

| Kernel | Current Status | Used By |
|--------|---------------|---------|
| **MLA Attention** | ✅ Available | DeepSeek-V2, DeepSeek-V3 (`flashinfer.mla`, `flashinfer.xqa_mla`) |
| **Mamba/SSM** | ❌ Not available | Mamba, Jamba, Zamba (`vllm/model_executor/layers/mamba/ops/`) |
| **GDN Attention** | ❌ Not available | Specialized attention variants |
| **KDA Attention** | ❌ Not available | Key-value decomposed attention |
| **Tree Attention** | ❌ Not available | Speculative decoding |

### Communication Kernels

| Kernel | Current Status | Notes |
|--------|---------------|-------|
| **AllReduce** | ⚠️ JIT broken | `flashinfer.comm.trtllm_allreduce_fusion` exists but has C++ bugs |
| **All2All** | ✅ Available | `flashinfer_all2allv` backend works |
| **Custom AllReduce (one-shot)** | ⚠️ Partial | FlashInfer has `trtllm_custom_all_reduce` |
| **Quick AllReduce** | ❌ Not available | vLLM's `quick_all_reduce` is separate |

### LoRA Kernels

| Kernel | Current Status | Notes |
|--------|---------------|-------|
| **Punica BGMV** | ❌ Not available | Batched GEMV for LoRA (`vllm/lora/punica_wrapper/`) |
| **Punica SGMV** | ❌ Not available | Segmented GEMV for LoRA |
| **LoRA expand/shrink** | ❌ Not available | Used in multi-LoRA inference |

### KV Cache Operations

| Operation | Current Status | Notes |
|-----------|---------------|-------|
| **Append paged KV cache** | ✅ Available | `flashinfer.append_paged_kv_cache` |
| **Append MLA KV cache** | ✅ Available | `flashinfer.append_paged_mla_kv_cache` |
| **Copy blocks** | ❌ Not available | vLLM uses custom kernel for block copying |
| **Reshape/swap blocks** | ❌ Not available | vLLM uses custom kernels |

### Miscellaneous

| Kernel | Current Status | Notes |
|--------|---------------|-------|
| **Fused cross-entropy loss** | ❌ Not available | Used for training/fine-tuning |
| **Rotary embedding (batched)** | ✅ Available | `flashinfer.rope.*` |
| **Speculative sampling** | ✅ Available | `flashinfer.chain_speculative_sampling` |
| **Top-k/Top-p sampling** | ✅ Available | `flashinfer.sampling.*` |
| **Min-p sampling** | ✅ Available | `flashinfer.min_p_sampling_from_probs` |

### Summary: What's Needed for Full FlashInfer Backend

To run vLLM entirely on FlashInfer kernels (no vLLM custom ops), FlashInfer would need:

1. **Embedding kernels** (vocab-parallel embedding lookup)
2. **Dense linear (non-quantized)** - Or defer to cuBLAS, which is current behavior
3. **AWQ/GPTQ/Marlin quantization kernels** for W4A16 models
4. **Mamba/SSM kernels** for state-space models
5. **LoRA kernels** (Punica BGMV/SGMV) for multi-adapter inference
6. **KV cache block copy/swap** kernels
7. **Fix AllReduce fusion JIT bugs** (high priority)

**Note:** Many of these are specialized kernels that may not make sense for FlashInfer's scope. The goal is not necessarily to replace everything, but to identify gaps for users who want maximum FlashInfer utilization.

---

## Recommendations for FlashInfer Team

### Priority 1: Fix AllReduce Fusion JIT (C++ Namespace Bug)

The `trtllm_allreduce_fusion` kernel is a critical feature for multi-GPU performance. The JIT compilation failure affects all GPU architectures.

**Bug:** `std::optional` vs `cuda::std::optional` namespace mismatch in CUDA header.

**File to fix:** `flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh`

**Specific lines to fix:**
- Line 487: `std::optional<int> batchIdx` → `cuda::std::optional<int> batchIdx`
- Line 489: `std::optional<int> numRows` → `cuda::std::optional<int> numRows`

**Or add at the top of the file (after includes):**
```cpp
namespace std { using cuda::std::optional; }
```

### Priority 2: MXFP4 MoE SM90 Support

Either implement SM90 CUTLASS kernels or clearly document that MXFP4 MoE is Blackwell-only. Currently vLLM claims SM90 support but it doesn't work.

**Suggested actions:**
- Add `flashinfer_cutlass_fused_moe_sm90_binding.cu` implementation, OR
- Return an error from `get_cutlass_fused_moe_module()` for SM90, OR
- Coordinate with vLLM team to remove the `SM90_FI_MXFP4_BF16` backend

### Priority 3: FP8 MoE CUDA Version

Provide clearer error messages distinguishing between:
- Compile-time CUDA version of the FlashInfer package
- Runtime CUDA version of the system

Consider shipping multiple wheel variants for different CUDA versions.

### Priority 4: Attention Sinks on Hopper

Consider implementing attention sink support that doesn't require TRTLLM, enabling FlashInfer attention for models like GPT-OSS on Hopper hardware.

---

## Test Script

A test script is available at `tests/kernels/run_flashinfer_test.py` to verify FlashInfer integration:

```bash
# Test all FlashInfer features
VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model all

# Test specific model
VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model qwen
VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model llama
VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model gpt-oss

# Test with FP8 quantization
VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model llama --fp8
```

---

*Document last updated: November 28, 2024*  
*vLLM version: 0.8.3rc2.dev6045*  
*FlashInfer versions tested: 0.5.2, 0.5.3*

