# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 decode GEMM selection for unquantized BF16 on SM103.

Dispatch is purely by local ``(N, K)`` shape and token count ``M`` -- the module
name plays no role. Each measured shape maps to a :class:`ProjectionSpec`
listing the token counts where ``dsv3_fused_a_gemm`` beats the default
unquantized GEMM. The static part of the decision is resolved once per module
at install time into a small ``{M: backend}`` plan, so the per-forward path is a
single dict lookup. Every shape and token count absent from the table falls
through to :func:`default_unquantized_gemm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform

Backend = Literal["dsv3_fused_a"]


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    n: int
    k: int
    dsv3_tokens: frozenset[int] = frozenset()
    name: str = ""  # optional debug label; never used for dispatch


_M1_TO_16 = frozenset(range(1, 17))

# Keyed by local (N, K). Where two projections share a shape (only 1536x7168:
# shared_gate_up_proj and mla_g_proj) the entry is unified.
KIMI_K3_PROJECTIONS: dict[tuple[int, int], ProjectionSpec] = {
    (1536, 128): ProjectionSpec(1536, 128, _M1_TO_16, name="f_b_proj"),
    (3072, 128): ProjectionSpec(3072, 128, _M1_TO_16, name="f_b_proj"),
    # 1536x7168 is shared by shared_gate_up_proj and mla_g_proj. dsv3 M1..16 is
    # only crash-safe once the mla_g aux-stream/PDL capture fix lands (subtask
    # task_7388aba1); the fallback if it cannot be fixed is dsv3_tokens=_M1.
    (1536, 7168): ProjectionSpec(
        1536, 7168, _M1_TO_16, name="shared_gate_up_proj/mla_g_proj"
    ),
    (2112, 7168): ProjectionSpec(2112, 7168, _M1_TO_16, name="fused_qkv_a_proj"),
    (2304, 1536): ProjectionSpec(2304, 1536, _M1_TO_16, name="q_b_proj"),
    (4608, 1536): ProjectionSpec(4608, 1536, _M1_TO_16, name="q_b_proj"),
    (3584, 7168): ProjectionSpec(
        3584, 7168, frozenset(range(2, 9)), name="routed_expert_down_proj"
    ),
    (7168, 768): ProjectionSpec(7168, 768, _M1_TO_16, name="shared_down_proj"),
    # TP16. Measured on B300 over M=1..16 with the same >=5% threshold as the
    # entries above. The replicated projections (2112x7168, 3584x7168) keep
    # their shapes at TP16, and o_proj lands on 7168x768, which
    # shared_down_proj already covers.
    (3216, 7168): ProjectionSpec(
        3216,
        7168,
        # Both gaps in this range are measured, not oversights: dsv3 is only
        # 4% ahead at M6..M8, and at M16 cuBLAS switches to a faster kernel
        # (11.42us vs dsv3's 11.83us) after trailing it by 6-8% at M9..M15.
        frozenset(range(9, 16)),
        name="in_proj_qkvgfab",
    ),
    (768, 7168): ProjectionSpec(
        768, 7168, frozenset(range(5, 17)), name="mla_g_proj/shared_gate_up_proj"
    ),
    (1152, 1536): ProjectionSpec(1152, 1536, frozenset(range(2, 17)), name="q_b_proj"),
    (768, 128): ProjectionSpec(768, 128, _M1_TO_16, name="f_b_proj"),
    # dsv3 drops under 5% from M9 on for this shape.
    (7168, 384): ProjectionSpec(
        7168, 384, frozenset(range(1, 9)), name="shared_down_proj"
    ),
    (4224, 7168): ProjectionSpec(
        4224, 7168, frozenset(range(4, 9)), name="dense_gate_up_proj"
    ),
}


def select_kimi_k3_backend(num_tokens: int, n: int, k: int) -> Backend | None:
    """Backend for a local ``(N, K)`` at ``num_tokens``, or None to fall back."""
    spec = KIMI_K3_PROJECTIONS.get((n, k))
    if spec is None or num_tokens not in spec.dsv3_tokens:
        return None
    return "dsv3_fused_a"


def _build_plan(spec: ProjectionSpec) -> dict[int, Backend]:
    return {
        num_tokens: "dsv3_fused_a"
        for num_tokens in range(1, 17)
        if num_tokens in spec.dsv3_tokens
    }


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _run_plan(
    plan: dict[int, Backend], x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    if plan.get(x.shape[0]) is None:
        return None
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        return None
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    ops.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)
    return output


def try_low_latency_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    """Run the shape-selected low-latency kernel, or None to fall back.

    Resolves the plan from the shape table on each call; production installs a
    precomputed plan (see :func:`enable_kimi_k3_low_latency_gemm`) and does not
    use this path.
    """
    if envs.VLLM_BATCH_INVARIANT or not _is_sm103() or not _runtime_ok(x, weight):
        return None
    spec = KIMI_K3_PROJECTIONS.get((weight.shape[0], weight.shape[1]))
    if spec is None:
        return None
    return _run_plan(_build_plan(spec), x, weight)


class KimiK3LowLatencyLinearMethod(UnquantizedLinearMethod):
    """Try the precomputed plan, else defer to the base method."""

    def __init__(self, plan: dict[int, Backend]) -> None:
        super().__init__()
        self._plan = plan

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            bias is None
            and not envs.VLLM_BATCH_INVARIANT
            and _runtime_ok(x, layer.weight)
        ):
            output = _run_plan(self._plan, x, layer.weight)
            if output is not None:
                return output
        return super().apply(layer, x, bias)


def enable_kimi_k3_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Install the shape-selected low-latency GEMM on matching linears.

    Modules are matched purely by type, an exactly-unquantized method, and a
    local ``(N, K)`` present in :data:`KIMI_K3_PROJECTIONS`.
    """
    if dtype != torch.bfloat16 or not _is_sm103():
        return

    for child in module.modules():
        if (
            not isinstance(child, LinearBase)
            or type(child.quant_method) is not UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        spec = KIMI_K3_PROJECTIONS.get((weight.shape[0], weight.shape[1]))
        if spec is None:
            continue
        child.quant_method = KimiK3LowLatencyLinearMethod(_build_plan(spec))
