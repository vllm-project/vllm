# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fusion pass: GEMM (scaled_mm) + static FP8 quantization.

Matches the graph pattern where a scaled matrix multiply produces BF16/FP16
output that is immediately quantized to FP8 via static_scaled_fp8_quant,
and replaces it with a single fused kernel.

On ROCm: uses torch._scaled_mm with FP8 output dtype via hipBLASLt.
"""

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()

# Static quant op (same on all platforms)
STATIC_FP8_QUANT_OP = torch.ops._C.static_scaled_fp8_quant.default

# Platform-specific scaled_mm and fused ops
SCALED_MM_OP = None
FUSED_OP = None

if current_platform.is_rocm():
    # Ensure the fused op is registered
    import vllm.model_executor.kernels.linear.scaled_mm.rocm_fused_gemm_fp8_quant  # noqa: F401, E501

    FUSED_OP = torch.ops.vllm.rocm_scaled_mm_static_fp8_quant.default

    if hasattr(torch.ops.vllm, "rocm_per_tensor_float_w8a8_scaled_mm_impl"):
        SCALED_MM_OP = torch.ops.vllm.rocm_per_tensor_float_w8a8_scaled_mm_impl.default


class GemmStaticFP8QuantPattern:
    """
    Matches: scaled_mm(a, b, out_dtype, As, Bs, bias) → BF16/FP16
         +   static_scaled_fp8_quant(result, input, scale, group_shape) → FP8

    Replaces with: fused_op(a, b, As, Bs, output_scale, bias) → FP8
    """

    def __init__(
        self,
        mm_out_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.mm_out_dtype = mm_out_dtype
        self.device = device

    def _empty(self, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(*shape, dtype=dtype, device=self.device)

    def pattern(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        bias: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        # Step 1: scaled_mm → BF16/FP16
        mm_result = auto_functionalized(
            SCALED_MM_OP,
            A=a,
            B=b,
            out_dtype=self.mm_out_dtype,
            As=a_scales,
            Bs=b_scales,
            bias=bias,
        )
        mm_out = mm_result[1]

        # Step 2: static_scaled_fp8_quant → FP8
        quant_result = auto_functionalized(
            STATIC_FP8_QUANT_OP,
            result=self._empty(1, 1, dtype=FP8_DTYPE),
            input=mm_out,
            scale=output_scale,
            group_shape=None,
        )
        return quant_result[1]

    def replacement(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        bias: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        fused_result = auto_functionalized(
            FUSED_OP,
            a=a,
            b=b,
            a_scales=a_scales,
            b_scales=b_scales,
            output_scale=output_scale,
            bias=bias,
        )
        return fused_result[1]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        inputs = [
            self._empty(1, 1, dtype=FP8_DTYPE),  # a
            self._empty(1, 1, dtype=FP8_DTYPE),  # b
            self._empty(1, 1, dtype=torch.float32),  # a_scales
            self._empty(1, 1, dtype=torch.float32),  # b_scales
            self._empty(1, dtype=torch.float32),  # bias
            self._empty(1, dtype=torch.float32),  # output_scale
        ]

        register_replacement(
            self.pattern,
            self.replacement,
            inputs,
            fwd_only,
            pm_pass,
        )


class GemmQuantFusionPass(VllmPatternMatcherPass):
    """
    Compilation pass that fuses GEMM + static FP8 quantization.

    Supported platforms:
    - ROCm (MI300X+): via torch._scaled_mm with FP8 output dtype
      (hipBLASLt natively supports FP8 output since ROCm 6.0)
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns = PatternMatcherPass(pass_name="gemm_quant_fusion_pass")

        if SCALED_MM_OP is None or FUSED_OP is None:
            logger.debug(
                "GEMM + FP8 quant fusion: no fused op available "
                "for current platform, skipping"
            )
            return

        for out_dtype in (torch.bfloat16, torch.float16):
            GemmStaticFP8QuantPattern(out_dtype, self.device).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "GemmQuantFusion: replaced %s patterns",
            self.matched_count,
        )

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, GemmStaticFP8QuantPattern)
