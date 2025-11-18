# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import helion
import helion.language as hl
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)
from torch._ops import OpOverload

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Quant,
)
from vllm.platforms import current_platform

from .fusion import QUANT_OPS, empty_bf16, empty_fp32, empty_i32
from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherQuantFP8, MatcherSiluAndMul
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


# TODO(gmagogsfm): Instead of speciying one config, we should
# use Helion bound kernel to generate many kernels according to
@torch.library.custom_op(
    "my_helion_lib::silu_mul_fp8", mutates_args=(), device_types="cuda"
)
@helion.kernel(
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    config=helion.Config(
        block_sizes=[1, 2048],
        flatten_loops=[True],
        indexing=["tensor_descriptor", "pointer", "tensor_descriptor", "pointer"],
        l2_groupings=[32],
        load_eviction_policies=["first", "first", "first"],
        loop_orders=[[0, 1]],
        num_stages=7,
        num_warps=4,
        pid_type="persistent_interleaved",
        range_flattens=[None],
        range_multi_buffers=[None],
        range_num_stages=[1],
        range_unroll_factors=[0],
        range_warp_specializes=[],
    ),
)
def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU-and-mul with FP8 quantization.

    Takes an input tensor, splits it along the last dimension into two halves,
    applies SiLU activation to the first half, multiplies with the second half,
    and quantizes the result to FP8.

    Operation: quantize_fp8(SiLU(input[..., :d]) * input[..., d:2*d])
    where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Args:
        input (Tensor): Input tensor with last dimension = 2*d
        scale (Tensor): Scalar scale factor for FP8 quantization

    Returns:
        Tensor: Output tensor with shape [..., d] and dtype float8_e4m3fn
    """
    d = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (d,)

    out = torch.empty(output_shape, device=input.device, dtype=torch.float8_e4m3fn)

    input_part_a = input[..., :d]
    input_part_b = input[..., d:]

    assert scale.numel() == 1, "Scale must be a scalar Tensor"

    for tile_idx in hl.tile(out.shape):
        a_vals = input_part_a[tile_idx].to(torch.float32)
        sigmoid_a = torch.sigmoid(a_vals)
        silu_result = a_vals * sigmoid_a
        silu_result = silu_result.to(input.dtype)
        b_vals = input_part_b[tile_idx]
        result = silu_result * b_vals
        result_f32 = result.to(torch.float32)
        scale_val = hl.load(scale, [0])
        inv_scale = 1.0 / scale_val
        result_scaled = result_f32 * inv_scale
        out[tile_idx] = result_scaled.to(out.dtype)

    return out


@silu_mul_fp8.register_fake
def silu_mul_fp8_fake(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Fake/meta implementation for silu_mul_fp8.
    Defines the input/output shape relationship without actual computation.

    Shape contract:
    - input: [..., 2*d]
    - scale: scalar (numel == 1)
    - returns: [..., d] with dtype float8_e4m3fn
    """
    d = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (d,)

    return torch.empty(output_shape, device=input.device, dtype=torch.float8_e4m3fn)


FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

SILU_MUL_OP = torch.ops._C.silu_and_mul.default

FUSED_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.silu_and_mul_quant.default,  # noqa: E501
}
silu_and_mul_nvfp4_quant_supported = current_platform.is_cuda() and hasattr(
    torch.ops._C, "silu_and_mul_nvfp4_quant"
)
if silu_and_mul_nvfp4_quant_supported:
    FUSED_OPS[kNvfp4Quant] = torch.ops._C.silu_and_mul_nvfp4_quant.default  # noqa: E501


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

        assert self.quant_key in QUANT_OPS, (
            f"unsupported quantization scheme {self.quant_key}"
        )
        self.QUANT_OP = QUANT_OPS[self.quant_key]

        assert self.quant_key in FUSED_OPS, (
            f"unsupported fusion scheme {self.quant_key}"
        )
        self.FUSED_OP = FUSED_OPS[self.quant_key]

        self.silu_and_mul_matcher = MatcherSiluAndMul()

    def empty_quant(self, *args, **kwargs):
        kwargs = {"dtype": self.quant_dtype, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    @abstractmethod
    def register(self, pm_pass: PatternMatcherPass):
        raise NotImplementedError


class SiluMulFp8StaticQuantPattern(ActivationQuantPattern):
    """
    Fusion for SiluMul+Fp8StaticQuant Pattern
    """

    def __init__(self, use_helion: bool = False):
        super().__init__(kFp8StaticTensorSym)
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)
        self.use_helion = use_helion

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            scale: torch.Tensor,
        ):
            result_silu_mul = self.silu_and_mul_matcher(input)
            result_quant = self.quant_matcher(result_silu_mul, scale)
            return result_quant[0]

        def replacement(
            input: torch.Tensor,
            scale: torch.Tensor,
        ):
            if self.use_helion:
                return torch.ops.my_helion_lib.silu_mul_fp8(input, scale)
            else:
                d = input.shape[-1] // 2
                output_shape = input.shape[:-1] + (d,)
                result = torch.empty(
                    output_shape, device=input.device, dtype=self.quant_dtype
                )
                at = auto_functionalized(
                    self.FUSED_OP, result=result, input=input, scale=scale
                )
                return at[1]

        inputs = [
            *self.silu_and_mul_matcher.inputs(),  # input
            self.quant_matcher.inputs()[1],  # scale
        ]
        pattern(*inputs)

        register_replacement(pattern, replacement, inputs, fwd_only, pm_pass)


class SiluMulNvfp4QuantPattern(ActivationQuantPattern):
    """
    Fusion for SiluMul+Nvfp4Quant Pattern
    """

    def __init__(self):
        super().__init__(kNvfp4Quant)

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            result: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            scale: torch.Tensor,
        ):
            result_silu_mul = self.silu_and_mul_matcher(input)
            at = auto_functionalized(
                self.QUANT_OP,
                output=result,
                input=result_silu_mul,
                output_scale=output_scale,
                input_scale=scale,
            )
            return at[1], at[2]

        def replacement(
            result: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                result_block_scale=output_scale,
                input=input,
                input_global_scale=scale,
            )
            return at[1], at[2]

        inputs = [
            self.empty_quant(5, 32),  # result
            empty_i32(128, 4),  # output_scale
            empty_bf16(5, 64),  # input
            empty_fp32(1, 1),  # scale
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
    def __init__(self, config: VllmConfig, use_helion: bool):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="activation_quant_fusion_pass"
        )

        # TODO(gmagogsfm): Add a global flag to enable Helion kernels.
        pattern_silu_mul_fp8 = SiluMulFp8StaticQuantPattern(use_helion)
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
        return VllmInductorPass.hash_source(
            self,
            ActivationQuantPattern,
            SiluMulFp8StaticQuantPattern,
            SiluMulNvfp4QuantPattern,
        )
