# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE gate linear layer — hw-agnostic vendored copy.

Vendored from
``vllm/model_executor/layers/fused_moe/router/gate_linear.py``.
Differences vs. the upstream copy:

  * The DSV3 specialized router-GEMM tier (``ops.dsv3_router_gemm``)
    and the FP32 specialized router-GEMM tier
    (``ops.fp32_router_gemm``) are dropped — both are CUDA-binary
    kernels gated on Hopper/Blackwell device capability that the OOT
    host (which inherits ``_enum=CUDA``) cannot run. They also gate on
    ``num_experts`` / ``hidden_size`` dimensions that DSv4 doesn't
    match (DSv4 uses ``n_routed_experts=128``,
    ``hidden_size=2048``-class).
  * The cuBLAS ``bf16×bf16→fp32`` ``torch.mm`` path is kept — it's a
    generic torch op, not a vendor-specific fast path.
  * The ``PluggableLayer.register("gate_linear")`` decorator is
    skipped so the upstream layer keeps owning the registry name.
  * Inherits from the local hw-agnostic ``ReplicatedLinear``.
"""

import torch
from torch.nn.parameter import Parameter

from vllm.platforms import current_platform

from ...custom_op import PluggableLayer
from ..linear import ReplicatedLinear


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """MoE gate linear layer.

    Tier 1 (kept): cuBLAS ``bf16×bf16→fp32`` via ``torch.mm`` when the
    weight is bf16 and the requested output dtype is fp32. Generic
    torch op, portable across backends.

    Tier 2 (fallback): ``F.linear`` via ``ReplicatedLinear`` with an
    optional dtype cast to/from ``out_dtype``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        # If fp32 compute is required and we don't have a specialized
        # kernel available (always the case here — those kernels were
        # dropped), store weights in fp32 so the fallback linear path
        # computes in fp32.
        if force_fp32_compute:
            params_dtype = torch.float32

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.out_dtype = out_dtype

        # Generic cuBLAS bf16→fp32 ``torch.mm`` path is fine on any
        # CUDA-alike device (including the OOT plugin piggybacking on
        # CUDA): the op routes through PyTorch's standard dispatch.
        self.allow_cublas_router_gemm = (
            current_platform.is_cuda_alike()
            and not bias
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        """Set output dtype for the router logits after init."""
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

        if not self.allow_cublas_router_gemm and out_dtype == torch.float32:
            self.allow_cublas_router_gemm = (
                current_platform.is_cuda_alike() and self.weight.dtype == torch.bfloat16
            )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Tier 1: cuBLAS bf16×bf16→fp32.
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)
            return output, None

        # Tier 2: F.linear via ReplicatedLinear.
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
