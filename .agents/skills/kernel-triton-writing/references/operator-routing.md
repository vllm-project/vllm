<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileComment: Copied from NVIDIA TensorRT-LLM at commit 395985c025c8d1cf5aa842bc752b337ba88721b6.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!-- markdownlint-disable MD040 MD060 -->

# Operator Routing Decision Reference

Detailed decision rules for determining whether an operator should be implemented
as a custom Triton kernel or handled by existing libraries.

## Decision Procedure

Follow these rules in order. Stop at the first match.

1. **Single element-wise op** (e.g., `relu(x)`, `sigmoid(x)`) -- SKIP. PyTorch
   already optimal, no fusion benefit.
2. **Standalone matmul** (e.g., `torch.matmul(a, b)`) -- SKIP. cuBLAS is highly
   optimized and hard to beat.
3. **Standard attention** (e.g., `F.scaled_dot_product_attention`) -- SKIP. Use
   FlashAttention.
4. **Element-wise chain (2+ ops)** (e.g., `gelu(dropout(x))`, `silu(x) * y`) --
   USE TRITON. Fuse memory-bound ops into compute-bound kernel.
5. **Reduction op** (e.g., LayerNorm, RMSNorm, Softmax) -- USE TRITON. Custom
   single-pass implementation beats generic PyTorch decomposition.
6. **Matmul + element-wise epilogue** (e.g., `matmul(a, b) + bias`,
   `matmul + gelu`) -- USE TRITON. Epilogue fusion avoids memory round-trip.
7. **Matmul + reduction** (e.g., `matmul -> softmax`, `matmul -> layernorm`) --
   USE TRITON. Common transformer pattern with clear fusion benefit.
8. **Custom attention variant** -- Check FlashAttention support first. Only use
   Triton if the variant is unsupported.
9. **Sparse operations** -- Triton can help, but evaluate specialized libraries
   (cuSPARSE, Triton block-sparse) first.
10. **Very small tensors** -- Launch overhead may dominate. Benchmark before
    committing.
11. **Default** -- Analyze operator code and shapes, then decide.

## Output Format

Report the routing decision as:

```markdown
## Routing Decision: [OPERATOR_NAME]

**Decision:** USE TRITON | SKIP TRITON | EVALUATE FURTHER

**Pattern:** [e.g., Element-wise chain, Reduction, Matmul+epilogue]

**Rationale:** [Why -- reference fusion benefit or lack thereof]

**Next Steps:**
- [USE TRITON] Proceed to Phase 1 (Analyze the Operator)
- [SKIP] Recommend alternative (cuBLAS, FlashAttention, PyTorch)
- [EVALUATE] Profile operator, analyze shapes, then re-decide
```

## Examples

### Fused GELU + Dropout

```python
def fused_op(x, p=0.1):
    return F.dropout(F.gelu(x), p=p)
```

**Decision:** USE TRITON | **Pattern:** Element-wise chain (2 ops)
Fusing eliminates one intermediate tensor write+read (~2x memory traffic reduction).

### Simple ReLU

```python
def simple_relu(x):
    return F.relu(x)
```

**Decision:** SKIP TRITON | **Pattern:** Single element-wise op
No fusion benefit. PyTorch ReLU is already a single memory-bound kernel.

### RMSNorm

```python
def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

**Decision:** USE TRITON | **Pattern:** Reduction op
Triton fuses square, mean, sqrt, divide, multiply in a single pass over the data.

### Linear + GELU

```python
def linear_gelu(x, weight, bias):
    return F.gelu(F.linear(x, weight, bias))
```

**Decision:** USE TRITON | **Pattern:** Matmul + element-wise epilogue
Fusing GELU into the matmul epilogue avoids an extra full tensor read+write.

## Edge Cases

- **Dynamic shapes or data-dependent branching** -- Triton requires static grid
  dimensions at launch. If shapes change per-sample, fall back to PyTorch eager
  or `torch.compile`.
- **Operators already in `torch.compile` fusion groups** -- Check whether
  `torch.compile` already fuses the pattern before writing a manual kernel.
  A manual Triton kernel is only justified if it measurably outperforms the
  compiler-generated version.
- **Mixed precision boundaries** -- Triton handles dtype casting well, but verify
  that the fused kernel preserves numerical behavior (especially around
  loss scaling and FP16/BF16 reductions).
