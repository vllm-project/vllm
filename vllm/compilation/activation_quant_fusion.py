# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)
from torch._ops import OpOverload

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
    GroupShape,
    ScaleDesc,
)
from vllm.platforms import current_platform

from .fusion import QUANT_OPS, empty_bf16, empty_fp32, empty_i32
from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherQuantFP8, MatcherSiluAndMul
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

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
    FUSED_OPS[kNvfp4Dynamic] = torch.ops._C.silu_and_mul_nvfp4_quant.default  # noqa: E501


class ActivationQuantPattern(ABC):
    """
    The base class for Activation+Quant fusions.
    Should not be used directly.
    """

    def __init__(
        self,
        quant_key: QuantKey,
    ) -> None:
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

    def empty_quant(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": self.quant_dtype, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    @abstractmethod
    def register(self, pm_pass: PatternMatcherPass) -> None:
        raise NotImplementedError


class SiluMulFp8StaticQuantPattern(ActivationQuantPattern):
    """
    Fusion for SiluMul+Fp8StaticQuant Pattern
    """

    def __init__(self) -> None:
        super().__init__(kFp8StaticTensorSym)
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        scale = self.quant_matcher.inputs()[1]
        return [
            *self.silu_and_mul_matcher.inputs(),  # input
            scale,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            result_silu_mul = self.silu_and_mul_matcher(input)
            result_quant = self.quant_matcher(result_silu_mul, scale)
            return result_quant[0]

        def replacement(
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            d = input.shape[-1] // 2
            output_shape = input.shape[:-1] + (d,)
            result = torch.empty(
                output_shape, device=input.device, dtype=self.quant_dtype
            )
            at = auto_functionalized(
                self.FUSED_OP, result=result, input=input, scale=scale
            )
            return at[1]

        inps = self.get_inputs()
        pattern(*inps)

        register_replacement(pattern, replacement, inps, fwd_only, pm_pass)


class SiluMulNvfp4QuantPattern(ActivationQuantPattern):
    """
    Fusion for SiluMul+Nvfp4Quant Pattern
    """

    def __init__(self) -> None:
        super().__init__(kNvfp4Dynamic)

    def get_inputs(self) -> list[torch.Tensor]:
        result = self.empty_quant(5, 32)
        output_scale = empty_i32(128, 4)
        input_ = empty_bf16(5, 64)
        scale = empty_fp32(1, 1)
        return [result, output_scale, input_, scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            result: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_silu_mul = self.silu_and_mul_matcher(input)
            at = auto_functionalized(
                self.QUANT_OP,
                output=result,
                input=result_silu_mul,
                output_scale=output_scale,
                input_scale=scale,
                is_sf_swizzled_layout=True,
            )
            return at[1], at[2]

        def replacement(
            result: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                result_block_scale=output_scale,
                input=input,
                input_global_scale=scale,
            )
            return at[1], at[2]

        register_replacement(pattern, replacement, self.get_inputs(), fwd_only, pm_pass)

# In activation_quant_fusion.py - REPLACE SiluMulBlockQuantPattern class

class SiluMulBlockQuantPattern:
    """
    Pattern for fusing inline SiLU+Mul with block quantization.
    
    Matches:
        gate, up = split(input, dim=-1)
        silu = sigmoid(gate) * gate  # Inline SiLU
        result = silu * up
        quantize(result)
    Into:
        silu_and_mul_per_block_quant(input, group_size)
    """
    
    def __init__(
        self, 
        group_shape: GroupShape,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
    ):
        assert group_shape[0] == 1, (
            f"SiluMulBlockQuantPattern only supports per-token quantization "
            f"(group_m=1), got group_shape={group_shape}"
        )
        
        self.group_shape = group_shape
        self.group_size = group_shape[1]
        self.has_col_major_scales = has_col_major_scales
        self.is_e8m0 = is_e8m0
        self.quant_dtype = FP8_DTYPE
        
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None
        
        # Create quant matcher
        scale = ScaleDesc(torch.float32, False, group_shape)
        quant_key = QuantKey(dtype=FP8_DTYPE, scale=scale, symmetric=True)
        self.quant_matcher = MatcherQuantFP8(
            quant_key, 
            has_col_major_scales=has_col_major_scales,
            is_e8m0=is_e8m0
        )
    
    def register(self, pm_pass: PatternMatcherPass) -> None:
        """Register pattern that matches inline SiLU+Mul operations."""
        
        # DEFINE THE PATTERN TO MATCH (native ops)
        def pattern(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Match: split → sigmoid → mul → mul → quantize
            hidden = input.shape[-1] // 2
            gate, up = input.split(hidden, dim=-1)
            
            # Inline SiLU: sigmoid(gate) * gate
            silu = torch.sigmoid(gate) * gate
            
            # Multiply with up
            silu_out = silu * up
            
            # Quantize
            result, scale = self.quant_matcher(silu_out)
            return result, scale
        
        # DEFINE THE REPLACEMENT (fused operation)
        def replacement(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            print(f"NATIVE OPS FUSION TRIGGERED! input.shape={input.shape}, group_size={self.group_size}")
            
            if self.model_dtype is not None:
                input = input.to(dtype=self.model_dtype)
            
            # Calculate output shape
            output_shape = list(input.shape)
            output_shape[-1] = output_shape[-1] // 2
            
            # Allocate tensors
            result = torch.empty(
                output_shape, 
                device=input.device, 
                dtype=self.quant_dtype
            )
            
            scale = self.quant_matcher.make_scale(
                torch.empty(output_shape, device=input.device),
                transposed=self.has_col_major_scales
            )
            
            # Call fused kernel
            at = auto_functionalized(
                torch.ops._C.silu_and_mul_per_block_quant.default,
                result=result,
                input=input,
                scale=scale,
                group_size=self.group_size,
                scale_ub=None,
                is_scale_transposed=self.has_col_major_scales,
            )
            
            return at[1], at[2]
        
        # Create example inputs for pattern matching
        # Pattern matcher needs to see what the pattern looks like
        input_example = torch.empty(5, 32, dtype=torch.float16, device="cuda")
        
        # REGISTER THE PATTERN
        register_replacement(
            pattern,
            replacement,
            [input_example],  # Example inputs
            fwd_only,
            pm_pass,
        )
        
class ActivationQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="activation_quant_fusion_pass"
        )

        pattern_silu_mul_fp8 = SiluMulFp8StaticQuantPattern()
        pattern_silu_mul_fp8.register(self.patterns)

        if silu_and_mul_nvfp4_quant_supported:
            pattern_silu_mul_nvfp4 = SiluMulNvfp4QuantPattern()
            pattern_silu_mul_nvfp4.register(self.patterns)

        # =====================================================================
        # NEW: Register block quantization patterns
        # =====================================================================
        if current_platform.is_cuda():
            # Register patterns for different group sizes and layouts
            count = 0
            for group_shape in [GroupShape(1, 128), GroupShape(1, 64)]:
                for has_col_major_scales in [True, False]:
                    for is_e8m0 in [True, False]:
                        pattern_silu_mul_block = SiluMulBlockQuantPattern(
                            group_shape=group_shape,
                            has_col_major_scales=has_col_major_scales,
                            is_e8m0=is_e8m0,
                        )
                        pattern_silu_mul_block.register(self.patterns)
                        count += 1
                        print(f"  Registered pattern {count}: group_size={group_shape[1]}, col_major={has_col_major_scales}, e8m0={is_e8m0}")

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            ActivationQuantPattern,
            SiluMulFp8StaticQuantPattern,
            SiluMulNvfp4QuantPattern,
            SiluMulBlockQuantPattern,
        )
