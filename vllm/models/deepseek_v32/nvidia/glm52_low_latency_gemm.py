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

import functools
from dataclasses import dataclass
from typing import Any, Literal

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

Backend = Literal["cute", "dsv3_fused_a", "fi_warp_splitk"]
# FlashInfer warp split-K tactic: (output_tile, token_tile, k_tile, stages,
# b_loader_warps).
WarpSplitKConfig = tuple[int, int, int, int, int]
ResolvedCall = tuple[Backend, SkinnyGemmConfig | WarpSplitKConfig | None]


@dataclass(frozen=True, slots=True)
class GLM52ProjectionSpec:
    n: int
    k: int
    cute_configs: tuple[tuple[int, SkinnyGemmConfig], ...]
    dsv3_tokens: frozenset[int] = frozenset()
    warp_splitk_configs: tuple[tuple[int, WarpSplitKConfig], ...] = ()

    def build_plan(self) -> dict[int, ResolvedCall]:
        plan: dict[int, ResolvedCall] = {
            num_tokens: ("cute", config) for num_tokens, config in self.cute_configs
        }
        plan.update(
            (num_tokens, ("dsv3_fused_a", None)) for num_tokens in self.dsv3_tokens
        )
        plan.update(
            (num_tokens, ("fi_warp_splitk", config))
            for num_tokens, config in self.warp_splitk_configs
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


def _warp_splitk_rows(
    measured: dict[int, WarpSplitKConfig], first_m: int = 1
) -> tuple[tuple[int, WarpSplitKConfig], ...]:
    """Serve every M up to the largest swept M with the tactic swept at the
    smallest swept M' >= M."""
    swept = sorted(measured)
    return tuple(
        (m, measured[next(s for s in swept if s >= m)])
        for m in range(first_m, swept[-1] + 1)
    )


# SM107 (VRNVL72) table, swept on one VRNVL72 GPU with CUDA graph replay, cold L2
# and PDL. FlashInfer warp split-K reduces in-kernel, so no split-K reduce kernel
# follows it. Times are table backend / cuBLAS at M=8.
# q_a + kv_a, replicated: 3.53 / 6.14 us.
_SM107_QKV_A_PROJECTION = GLM52ProjectionSpec(
    n=2624,
    k=6144,
    cute_configs=(),
    warp_splitk_configs=_warp_splitk_rows(
        {
            1: (16, 8, 256, 16, 2),
            2: (16, 8, 256, 16, 2),
            4: (16, 8, 256, 16, 2),
            8: (16, 8, 256, 15, 2),
            12: (16, 16, 256, 12, 2),
            16: (16, 16, 256, 12, 2),
            24: (16, 32, 256, 9, 2),
            32: (32, 16, 256, 9, 2),
        }
    ),
)

# MTP eh_proj: 11.83 / 15.19 us.
_SM107_EH_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=12288,
    cute_configs=(),
    warp_splitk_configs=_warp_splitk_rows(
        {
            8: (32, 8, 256, 11, 2),
            16: (32, 16, 256, 9, 2),
            32: (32, 32, 256, 6, 2),
        }
    ),
)

# q_b_proj at TP4 and the indexer wq_b (replicated): 2.45 / 4.93 us.
_SM107_Q_B_TP4_PROJECTION = GLM52ProjectionSpec(
    n=4096,
    k=2048,
    cute_configs=(),
    warp_splitk_configs=_warp_splitk_rows(
        {
            1: (32, 8, 256, 8, 2),
            2: (32, 8, 128, 16, 2),
            8: (32, 8, 256, 8, 2),
            12: (32, 16, 128, 14, 2),
            16: (32, 16, 256, 8, 2),
            32: (32, 32, 128, 11, 2),
        }
    ),
)

# o_proj at TP4: 4.41 / 7.09 us.
_SM107_O_PROJ_TP4_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=4096,
    cute_configs=(),
    warp_splitk_configs=_warp_splitk_rows(
        {
            1: (16, 8, 256, 9, 2),
            2: (16, 8, 256, 9, 1),
            4: (16, 8, 256, 9, 2),
            8: (32, 8, 256, 11, 2),
            16: (32, 16, 256, 9, 2),
            32: (32, 32, 256, 6, 2),
        }
    ),
)

# indexer wk + weights_proj, replicated: CuTe up to M=8 (2.44 / 4.27 us), then
# warp split-K (2.83 / 3.74 us at M=16).
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
    ),
    warp_splitk_configs=_warp_splitk_rows(
        {16: (16, 8, 256, 14, 2), 32: (16, 8, 256, 16, 2)}, first_m=9
    ),
)

# shared expert gate_up at TP4: 2.91 / 4.55 us.
_SM107_SHARED_GATE_UP_TP4_PROJECTION = GLM52ProjectionSpec(
    n=1024,
    k=6144,
    cute_configs=(),
    warp_splitk_configs=_warp_splitk_rows(
        {
            1: (16, 8, 256, 15, 2),
            24: (16, 8, 256, 16, 2),
            32: (16, 16, 256, 10, 2),
        }
    ),
)

# shared expert down at TP4: 1.75 / 3.15 us.
_SM107_SHARED_DOWN_TP4_PROJECTION = GLM52ProjectionSpec(
    n=6144,
    k=512,
    cute_configs=(),
    warp_splitk_configs=_warp_splitk_rows(
        {
            24: (32, 8, 128, 4, 2),
            32: (32, 8, 128, 3, 1),
        }
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


@functools.cache
def _warp_splitk_module() -> Any | None:
    """FlashInfer's CuTe DSL warp split-K BF16 GEMM, called directly so every
    (N, K, M) runs its measured tactic: `flashinfer.mm_bf16` takes no tactic
    and would use its autotuner or heuristic default instead."""
    try:
        from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk
    except Exception:
        return None
    return dense_bf16_gemm_warp_splitk


def _run_warp_splitk(
    x: torch.Tensor, weight: torch.Tensor, config: WarpSplitKConfig
) -> torch.Tensor | None:
    module = _warp_splitk_module()
    if module is None or x.data_ptr() % 32 or weight.data_ptr() % 32:
        return None
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    return module.run_warp_splitk_dense(
        x, weight.t(), output, True, module.WarpSplitKTactic(*config)
    )


class _WarpSplitKWarmup:
    """Compiles the planned warp split-K kernels before CUDA graph capture."""

    def __init__(self) -> None:
        self._units: set[tuple[int, int, WarpSplitKConfig]] = set()
        self._registered = False

    def request(self, n: int, k: int, configs: set[WarpSplitKConfig]) -> None:
        if not configs or _warp_splitk_module() is None:
            return
        self._units.update((n, k, config) for config in configs)
        if self._registered:
            return
        from vllm.model_executor.warmup.cutedsl_warmup import (
            register_cutedsl_warmup_provider,
        )

        register_cutedsl_warmup_provider(self)
        self._registered = True

    @staticmethod
    def _compile(n: int, k: int, config: WarpSplitKConfig) -> None:
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        x = torch.zeros((1, k), dtype=torch.bfloat16, device=device)
        weight = torch.zeros((n, k), dtype=torch.bfloat16, device=device)
        _run_warp_splitk(x, weight, config)

    def get_cutedsl_warmup_compile_units(self):
        from vllm.model_executor.warmup.cutedsl_warmup import CuTeDSLCompileUnit

        return tuple(
            CuTeDSLCompileUnit(
                name="FlashInfer warp split-K BF16 GEMM",
                key=("fi-warp-splitk", n, k, config),
                compile=functools.partial(self._compile, n, k, config),
            )
            for n, k, config in sorted(self._units)
        )


_warp_splitk_warmup = _WarpSplitKWarmup()


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
    if backend == "fi_warp_splitk":
        assert isinstance(config, tuple)
        return _run_warp_splitk(x, weight, config)
    if backend == "cute":
        assert isinstance(config, SkinnyGemmConfig)
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


def _request_warmup(dtype: torch.dtype, spec: GLM52ProjectionSpec) -> None:
    configs = {config for _, config in spec.cute_configs}
    if configs and shape_dynamic_skinny_gemm.is_available():
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, configs)
    _warp_splitk_warmup.request(
        spec.n, spec.k, {config for _, config in spec.warp_splitk_configs}
    )


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
    _request_warmup(dtype, spec)
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

    planned: set[GLM52ProjectionSpec] = set()
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
        planned.add(spec)

    for spec in planned:
        _request_warmup(dtype, spec)
