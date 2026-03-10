# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
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


class GemmStaticFP8QuantPattern:
    """Pattern: CUTLASS GEMM followed by static per-tensor FP8 quantization."""

    def __init__(self, out_dtype: torch.dtype, device: str | None) -> None:
        self.out_dtype = out_dtype
        self.device = device

    def get_inputs(self) -> list[torch.Tensor]:
        a = torch.empty((8, 16), device=self.device, dtype=FP8_DTYPE)
        b = (
            torch.empty((16, 16), device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        a_scales = torch.empty((8, 1), device=self.device, dtype=torch.float32)
        b_scales = torch.empty((1, 16), device=self.device, dtype=torch.float32)
        output_scale = torch.empty(1, device=self.device, dtype=torch.float32)
        mm_out = torch.empty((8, 16), device=self.device, dtype=self.out_dtype)
        quant_out = torch.empty((8, 16), device=self.device, dtype=FP8_DTYPE)
        return [a, b, a_scales, b_scales, output_scale, mm_out, quant_out]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a: torch.Tensor,
            b: torch.Tensor,
            a_scales: torch.Tensor,
            b_scales: torch.Tensor,
            output_scale: torch.Tensor,
            mm_out: torch.Tensor,
            quant_out: torch.Tensor,
        ) -> torch.Tensor:
            mm_at = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=mm_out,
                a=a,
                b=b,
                a_scales=a_scales,
                b_scales=b_scales,
                bias=None,
            )
            quant_at = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.static_scaled_fp8_quant.default,
                result=quant_out,
                input=mm_at[1],
                scale=output_scale,
                group_shape=None,
            )
            return quant_at[1]

        def replacement(
            a: torch.Tensor,
            b: torch.Tensor,
            a_scales: torch.Tensor,
            b_scales: torch.Tensor,
            output_scale: torch.Tensor,
            mm_out: torch.Tensor,
            quant_out: torch.Tensor,
        ) -> torch.Tensor:
            del mm_out
            fused_at = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm_static_fp8_quant.default,
                out=quant_out,
                a=a,
                b=b,
                a_scales=a_scales,
                b_scales=b_scales,
                output_scale=output_scale,
                bias=None,
            )
            return fused_at[1]

        register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            fwd_only,
            pm_pass,
            skip_duplicates=True,
        )


class GemmQuantFusionPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns = PatternMatcherPass(pass_name="gemm_quant_fusion_pass")

        if not current_platform.has_device_capability(89):
            logger.warning_once(
                "GEMM + static FP8 quant fusion pass is disabled for device "
                "capability < 89."
            )
            return

        for out_dtype in (torch.bfloat16, torch.float16):
            GemmStaticFP8QuantPattern(out_dtype, self.device).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, GemmStaticFP8QuantPattern)
