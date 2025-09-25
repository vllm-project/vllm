# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (PatternMatcherPass, fwd_only,
                                             register_replacement)
from torch._ops import OpOverload

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey, kFp8StaticTensorSym, kNvfp4Quant, kStaticTensorScale)
from vllm.platforms import current_platform

from .fusion import QUANT_OPS, empty_bf16, empty_fp32, empty_i32
from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

SILU_MUL_OP = torch.ops._C.silu_and_mul.default

FUSED_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.silu_and_mul_quant.default,  # noqa: E501
}
silu_and_mul_nvfp4_quant_supported = (current_platform.is_cuda() and hasattr(
    torch.ops._C, "silu_and_mul_nvfp4_quant"))
if silu_and_mul_nvfp4_quant_supported:
    FUSED_OPS[
        kNvfp4Quant] = torch.ops._C.silu_and_mul_nvfp4_quant.default  # noqa: E501


class ActivationQuantPattern(ABC):
    """
    The base class for Activation+Quant fusions.
    Should not be used directly.
    """

    def __init__(
        self,
        quant_key: QuantKey,
    ):
        self.quant_key = quant_key
        self.quant_dtype = quant_key.dtype

        assert self.quant_key in QUANT_OPS, \
            f"unsupported quantization scheme {self.quant_key}"
        self.QUANT_OP = QUANT_OPS[self.quant_key]

        assert self.quant_key in FUSED_OPS, \
            f"unsupported fusion scheme {self.quant_key}"
        self.FUSED_OP = FUSED_OPS[self.quant_key]

    def empty_quant(self, *args, **kwargs):
        kwargs = {'dtype': self.quant_dtype, 'device': "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    @abstractmethod
    def register(self, pm_pass: PatternMatcherPass):
        raise NotImplementedError


class SiluMulFp8StaticQuantPattern(ActivationQuantPattern):
    """
    Fusion for SiluMul+Fp8StaticQuant Pattern
    """

    def __init__(self, symmetric: bool = True):
        quant_key = QuantKey(dtype=FP8_DTYPE,
                             scale=kStaticTensorScale,
                             symmetric=symmetric)
        super().__init__(quant_key)

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(result: torch.Tensor, result_silu_mul: torch.Tensor,
                    input: torch.Tensor, scale: torch.Tensor):
            at1 = auto_functionalized(SILU_MUL_OP,
                                      result=result_silu_mul,
                                      input=input)
            at2 = auto_functionalized(self.QUANT_OP,
                                      result=result,
                                      input=at1[1],
                                      scale=scale)
            return at2[1]

        def replacement(result: torch.Tensor, result_silu_mul: torch.Tensor,
                        input: torch.Tensor, scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP,
                                     result=result,
                                     input=input,
                                     scale=scale)
            return at[1]

        inputs = [
            self.empty_quant(5, 4),  # result
            empty_bf16(5, 4),  # result_silu_mul
            empty_bf16(5, 4),  # input
            empty_fp32(1, 1)  # scale
        ]

        register_replacement(pattern, replacement, inputs, fwd_only, pm_pass)


class SiluMulNvfp4QuantPattern(ActivationQuantPattern):
    """
    Fusion for SiluMul+Nvfp4Quant Pattern
    """

    def __init__(self):
        super().__init__(kNvfp4Quant)

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(result: torch.Tensor, output_scale: torch.Tensor,
                    result_silu_mul: torch.Tensor, input: torch.Tensor,
                    scale: torch.Tensor):
            at1 = auto_functionalized(SILU_MUL_OP,
                                      result=result_silu_mul,
                                      input=input)
            at2 = auto_functionalized(self.QUANT_OP,
                                      output=result,
                                      input=at1[1],
                                      output_scale=output_scale,
                                      input_scale=scale)
            return at2[1], at2[2]

        def replacement(result: torch.Tensor, output_scale: torch.Tensor,
                        result_silu_mul: torch.Tensor, input: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP,
                                     result=result,
                                     result_block_scale=output_scale,
                                     input=input,
                                     input_global_scale=scale)
            return at[1], at[2]

        inputs = [
            self.empty_quant(5, 32),  # result
            empty_i32(128, 4),  # output_scale
            empty_bf16(5, 64),  # result_silu_mul
            empty_bf16(5, 64),  # input
            empty_fp32(1, 1)  # scale
        ]

        register_replacement(pattern, replacement, inputs, fwd_only, pm_pass)


class ActivationQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="activation_quant_fusion_pass")

        pattern_silu_mul_fp8 = SiluMulFp8StaticQuantPattern()
        pattern_silu_mul_fp8.register(self.patterns)

        if silu_and_mul_nvfp4_quant_supported:
            pattern_silu_mul_nvfp4 = SiluMulNvfp4QuantPattern()
            pattern_silu_mul_nvfp4.register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self):
        return VllmInductorPass.hash_source(self, ActivationQuantPattern,
                                            SiluMulFp8StaticQuantPattern,
                                            SiluMulNvfp4QuantPattern)
