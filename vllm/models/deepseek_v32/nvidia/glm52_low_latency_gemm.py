# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 decode GEMM selection for unquantized BF16 on SM100/SM103/SM107.

Dispatch is by local ``(N, K)`` weight shape and token count ``M``.
:data:`GLM52_PROJECTIONS` was measured on B300 (SM103),
:data:`GLM52_PROJECTIONS_SM107` on VRNVL72 (SM107) and
:data:`GLM52_PROJECTIONS_SM100` on B200 (SM100). The SM107 table also covers
the TP4 projections (q_b, o_proj, shared expert) and the replicated indexer
projections, which run as BF16 GEMMs at decode batch sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform

Backend = Literal["cute", "dsv3_fused_a"]
ResolvedCall = tuple[Backend, SkinnyGemmConfig | None]


@dataclass(frozen=True, slots=True)
class GLM52ProjectionSpec:
    n: int
    k: int
    cute_configs: tuple[tuple[int, SkinnyGemmConfig], ...]
    dsv3_tokens: frozenset[int] = frozenset()

    def build_plan(self) -> dict[int, ResolvedCall]:
        plan: dict[int, ResolvedCall] = {
            num_tokens: ("cute", config) for num_tokens, config in self.cute_configs
        }
        plan.update(
            (num_tokens, ("dsv3_fused_a", None)) for num_tokens in self.dsv3_tokens
        )
        return plan


GLM52_QKV_A_PROJECTION = GLM52ProjectionSpec(
    n=2624,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 4, static_k=6144)),
        (2, SkinnyGemmConfig(2, 128, 2)),
    ),
    dsv3_tokens=frozenset(range(3, 17)),
)

GLM52_Q_B_PROJECTION = GLM52ProjectionSpec(
    n=2048,
    k=2048,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 4, static_k=2048)),
        (2, SkinnyGemmConfig(2, 64, 2, k_unroll=2)),
    ),
    dsv3_tokens=frozenset(range(3, 17)),
)

# The MTP eh_proj is a plain nn.Linear, so it gets its plan through
# build_glm52_plan rather than a quant_method swap. cuBLAS wins from M=4 on SM103.
GLM52_EH_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=12288,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=12288)),
        (2, SkinnyGemmConfig(2, 64, 2)),
        (3, SkinnyGemmConfig(3, 64, 2)),
    ),
)

GLM52_PROJECTIONS = {
    (spec.n, spec.k): spec
    for spec in (
        GLM52_QKV_A_PROJECTION,
        GLM52_Q_B_PROJECTION,
        GLM52_EH_PROJECTION,
    )
}

# SM107 (VRNVL72) table: each M is listed only where it beat cuBLAS; other M use
# cuBLAS. Times are table backend / cuBLAS at M=4 unless noted.
# q_a + kv_a, replicated: 5.84 / 6.91 us.
_SM107_QKV_A_PROJECTION = GLM52ProjectionSpec(
    n=2624,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 4, k_unroll=2)),
        (2, SkinnyGemmConfig(2, 128, 2, k_unroll=2)),
        (3, SkinnyGemmConfig(3, 128, 2, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 128, 2, k_unroll=2)),
        (5, SkinnyGemmConfig(5, 128, 2)),
    ),
    dsv3_tokens=frozenset(range(6, 9)),
)

# MTP eh_proj (draft M = number of requests): 12.90 / 16.96 us at M=1.
_SM107_EH_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=12288,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=12288)),
        (2, SkinnyGemmConfig(2, 64, 4, k_unroll=4)),
        (3, SkinnyGemmConfig(3, 64, 2, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 128, 4)),
    ),
)

# q_b_proj at TP4 and the indexer wq_b (replicated): 4.31 / 5.71 us.
_SM107_Q_B_TP4_PROJECTION = GLM52ProjectionSpec(
    n=4096,
    k=2048,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 4, k_unroll=2)),
        (2, SkinnyGemmConfig(2, 128, 4, k_unroll=2)),
        (3, SkinnyGemmConfig(3, 128, 4, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 64, 4, k_unroll=4)),
        (5, SkinnyGemmConfig(5, 64, 2, k_unroll=2)),
        (6, SkinnyGemmConfig(6, 64, 2, k_unroll=2)),
    ),
)

# o_proj at TP4: CuTe wins only for M<=3 (draft passes): 5.90 / 7.77 us at M=1.
_SM107_O_PROJ_TP4_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=4096,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 3, k_unroll=4)),
        (2, SkinnyGemmConfig(2, 64, 4, k_unroll=4)),
        (3, SkinnyGemmConfig(3, 128, 4)),
    ),
)

# indexer wk + weights_proj, replicated: 2.83 / 5.43 us.
_SM107_WK_WEIGHTS_PROJECTION = GLM52ProjectionSpec(
    n=160,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 1, k_unroll=2, static_k=6144)),
        (2, SkinnyGemmConfig(2, 256, 1, k_unroll=2, static_k=6144)),
        (3, SkinnyGemmConfig(3, 256, 1, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 256, 1, k_unroll=2)),
        (5, SkinnyGemmConfig(5, 128, 2, k_unroll=2)),
        (6, SkinnyGemmConfig(6, 256, 1, k_unroll=2)),
        (8, SkinnyGemmConfig(8, 256, 1, k_unroll=2)),
        (12, SkinnyGemmConfig(12, 256, 1, k_unroll=2)),
        (16, SkinnyGemmConfig(16, 256, 1, k_unroll=2)),
    ),
)

# shared expert gate_up at TP4: 3.96 / 5.40 us.
_SM107_SHARED_GATE_UP_TP4_PROJECTION = GLM52ProjectionSpec(
    n=1024,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 2, k_unroll=2)),
        (2, SkinnyGemmConfig(2, 256, 2, k_unroll=2)),
        (3, SkinnyGemmConfig(3, 256, 2, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 128, 1, k_unroll=2)),
        (5, SkinnyGemmConfig(5, 128, 2, k_unroll=2)),
        (6, SkinnyGemmConfig(6, 128, 1, k_unroll=2)),
        (8, SkinnyGemmConfig(8, 128, 2, k_unroll=2)),
    ),
)

# shared expert down at TP4: 3.19 / 4.28 us.
_SM107_SHARED_DOWN_TP4_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=512,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 64, 6, k_unroll=4)),
        (2, SkinnyGemmConfig(2, 64, 4, k_unroll=4)),
        (3, SkinnyGemmConfig(3, 64, 4)),
        (4, SkinnyGemmConfig(4, 32, 4, k_unroll=2)),
        (6, SkinnyGemmConfig(6, 32, 4, k_unroll=2)),
    ),
)

# q_b_proj at TP8: the SM103 entry, re-measured on VRNVL72 and kept (CuTe at M=1-2
# 3.3 / 5.0 us; dsv3_fused_a_gemm at M=3-16 4.0-4.2 / 4.9-5.1 us).
_SM107_Q_B_TP8_PROJECTION = GLM52_Q_B_PROJECTION

GLM52_PROJECTIONS_SM107 = {
    (spec.n, spec.k): spec
    for spec in (
        _SM107_QKV_A_PROJECTION,
        _SM107_Q_B_TP8_PROJECTION,
        _SM107_EH_PROJECTION,
        _SM107_Q_B_TP4_PROJECTION,
        _SM107_O_PROJ_TP4_PROJECTION,
        _SM107_WK_WEIGHTS_PROJECTION,
        _SM107_SHARED_GATE_UP_TP4_PROJECTION,
        _SM107_SHARED_DOWN_TP4_PROJECTION,
    )
}


# SM100 (B200) table: M=1-4, 6 and 8 were measured; each M is listed only where it
# beat cuBLAS by >= 5 % (CUPTI, CUDA graph, cold and hot L2), others use cuBLAS.
# Times are table backend / cuBLAS, cold L2. o_proj at TP4 and the shared expert
# down projections stay on cuBLAS.
# q_a + kv_a, replicated: 8.42 / 12.00 us at M=1.
_SM100_QKV_A_PROJECTION = GLM52ProjectionSpec(
    n=2624,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 64, 4, k_unroll=4)),
        (2, SkinnyGemmConfig(2, 64, 2, k_unroll=4)),
        (3, SkinnyGemmConfig(3, 128, 2, k_unroll=2, vector_width=4)),
        (4, SkinnyGemmConfig(4, 64, 2, k_unroll=2)),
    ),
    dsv3_tokens=frozenset({6, 8}),
)

# q_b_proj at TP8: 4.16 / 6.05 us at M=1.
_SM100_Q_B_TP8_PROJECTION = GLM52ProjectionSpec(
    n=2048,
    k=2048,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 4, static_k=2048)),
        (2, SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4)),
        (3, SkinnyGemmConfig(3, 128, 2, k_unroll=4, vector_width=4)),
    ),
    dsv3_tokens=frozenset({4, 6, 8}),
)

# q_b_proj at TP4 and the indexer wq_b (replicated): 5.76 / 6.88 us at M=1.
_SM100_Q_B_TP4_PROJECTION = GLM52ProjectionSpec(
    n=4096,
    k=2048,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4)),
        (2, SkinnyGemmConfig(2, 64, 2, k_unroll=4)),
        (3, SkinnyGemmConfig(3, 64, 2, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=4)),
    ),
)

# o_proj at TP8: 6.94 / 7.97 us at M=1.
_SM100_O_PROJ_TP8_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=2048,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 64, 4, k_unroll=2)),
        (2, SkinnyGemmConfig(2, 64, 4, k_unroll=2)),
    ),
)

# indexer wk + weights_proj, replicated: 2.72 / 6.30 us at M=1.
_SM100_WK_WEIGHTS_PROJECTION = GLM52ProjectionSpec(
    n=160,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 1, k_unroll=2, static_k=6144)),
        (2, SkinnyGemmConfig(2, 256, 1, k_unroll=2, static_k=6144)),
        (3, SkinnyGemmConfig(3, 256, 1, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 256, 1, k_unroll=2)),
        (6, SkinnyGemmConfig(6, 256, 1, k_unroll=2)),
        (8, SkinnyGemmConfig(8, 256, 1, k_unroll=2)),
    ),
)

# shared expert gate_up at TP8: 4.19 / 7.71 us at M=1.
_SM100_SHARED_GATE_UP_TP8_PROJECTION = GLM52ProjectionSpec(
    n=512,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 1, k_unroll=2)),
        (2, SkinnyGemmConfig(2, 256, 1, k_unroll=2)),
        (3, SkinnyGemmConfig(3, 256, 1, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 256, 1, k_unroll=2)),
        (6, SkinnyGemmConfig(6, 128, 1, k_unroll=2)),
        (8, SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4)),
    ),
)

# shared expert gate_up at TP4: 5.28 / 8.67 us at M=1.
_SM100_SHARED_GATE_UP_TP4_PROJECTION = GLM52ProjectionSpec(
    n=1024,
    k=6144,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 2, k_unroll=2)),
        (2, SkinnyGemmConfig(2, 256, 2, k_unroll=2)),
        (3, SkinnyGemmConfig(3, 128, 1, k_unroll=2)),
        (4, SkinnyGemmConfig(4, 128, 1, k_unroll=2)),
        (6, SkinnyGemmConfig(6, 128, 2, k_unroll=4, vector_width=4)),
        (8, SkinnyGemmConfig(8, 128, 2, k_unroll=2)),
    ),
)

# MTP eh_proj (draft M = number of requests): 26.34 / 37.31 us at M=1.
_SM100_EH_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=12288,
    cute_configs=(
        (1, SkinnyGemmConfig(1, 256, 1, k_unroll=2)),
        (2, SkinnyGemmConfig(2, 128, 1, k_unroll=4)),
        (3, SkinnyGemmConfig(3, 128, 2, k_unroll=4, vector_width=4)),
        (4, SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4)),
    ),
)

GLM52_PROJECTIONS_SM100 = {
    (spec.n, spec.k): spec
    for spec in (
        _SM100_QKV_A_PROJECTION,
        _SM100_Q_B_TP8_PROJECTION,
        _SM100_Q_B_TP4_PROJECTION,
        _SM100_O_PROJ_TP8_PROJECTION,
        _SM100_WK_WEIGHTS_PROJECTION,
        _SM100_SHARED_GATE_UP_TP8_PROJECTION,
        _SM100_SHARED_GATE_UP_TP4_PROJECTION,
        _SM100_EH_PROJECTION,
    )
}


def _low_latency_table() -> dict[tuple[int, int], GLM52ProjectionSpec] | None:
    """Measured dispatch table for the current device, or None if unsupported."""
    if current_platform.is_device_capability((10, 7)):
        return GLM52_PROJECTIONS_SM107
    if current_platform.is_device_capability((10, 3)):
        return GLM52_PROJECTIONS
    if current_platform.is_device_capability((10, 0)):
        return GLM52_PROJECTIONS_SM100
    return None


def _is_supported_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_supported_row_major(x)
        and _is_supported_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def run_glm52_plan(
    plan: dict[int, ResolvedCall] | None,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    if plan is None or not _runtime_ok(x, weight):
        return None
    entry = plan.get(x.shape[0])
    if entry is None:
        return None

    backend, config = entry
    if backend == "cute":
        if not shape_dynamic_skinny_gemm.is_available():
            return None
        return shape_dynamic_skinny_gemm(x, weight, config)

    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        return None
    output = torch.empty(
        (x.shape[0], weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )
    ops.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)
    return output


def _request_warmup(dtype: torch.dtype, configs: set[SkinnyGemmConfig]) -> None:
    if configs and shape_dynamic_skinny_gemm.is_available():
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, configs)


def build_glm52_plan(
    weight: torch.Tensor | None, dtype: torch.dtype
) -> dict[int, ResolvedCall] | None:
    """Plan for a weight the walk below cannot reach (a plain ``nn.Linear``)."""
    if dtype != torch.bfloat16:
        return None
    table = _low_latency_table()
    if table is None:
        return None
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return None
    spec = table.get(tuple(weight.shape))
    if spec is None:
        return None
    _request_warmup(dtype, {config for _, config in spec.cute_configs})
    return spec.build_plan()


class GLM52LowLatencyLinearMethod(UnquantizedLinearMethod):
    def __init__(self, plan: dict[int, ResolvedCall]) -> None:
        super().__init__()
        self._plan = plan

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None:
            output = run_glm52_plan(self._plan, x, layer.weight)
            if output is not None:
                return output
        return super().apply(layer, x, bias)


def enable_glm52_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    if dtype != torch.bfloat16:
        return
    table = _low_latency_table()
    if table is None:
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        if (
            not isinstance(child, LinearBase)
            or type(child.quant_method) is not UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        spec = table.get(tuple(weight.shape))
        if spec is None:
            continue
        child.quant_method = GLM52LowLatencyLinearMethod(spec.build_plan())
        warmup_configs.update(config for _, config in spec.cute_configs)

    _request_warmup(dtype, warmup_configs)
