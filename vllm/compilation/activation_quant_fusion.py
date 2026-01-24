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

from vllm.config import VllmConfig
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

class SiluMulBlockQuantPattern:
    """
    This pattern fuses silu_and_mul & block quantization.
    Handles all group_size, col_major, and e8m0 variants in one pattern.
    """
    def __init__(self):
        self.quant_dtype = FP8_DTYPE
        
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None
        
        from .matcher_utils import MatcherSiluAndMul, MatcherQuantFP8
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        
        # Create a single matcher for group_size=128 as the pattern template
        # The actual replacement will handle all variants
        scale = ScaleDesc(torch.float32, False, GroupShape(1, 128))
        quant_key = QuantKey(dtype=FP8_DTYPE, scale=scale, symmetric=True)
        self.quant_matcher = MatcherQuantFP8(quant_key, has_col_major_scales=False, is_e8m0=False)
    
    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Write the FULL pattern explicitly - no matchers
            d = input.shape[-1] // 2
            gate = input[..., :d]
            silu = torch.nn.functional.silu(gate)
            up = input[..., d:]
            silu_out = silu * up
            
            # Match the in-place quantization pattern
            x_q = torch.empty(silu_out.shape, dtype=FP8_DTYPE, device=input.device)
            num_groups = silu_out.shape[-1] // 128
            x_s = torch.empty((silu_out.shape[0], num_groups), dtype=torch.float32, device=input.device)
            
            torch.ops._C.per_token_group_fp8_quant(silu_out, x_q, x_s, 128, 1e-10, -448.0, 448.0, False)
            
            return x_q, x_s
        
        def replacement(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            print(f"🔥 FUSED KERNEL TRIGGERED! input.shape={input.shape}")
            
            output_shape = list(input.shape)
            output_shape[-1] = output_shape[-1] // 2
            
            result = torch.empty(output_shape, device=input.device, dtype=self.quant_dtype)
            num_groups = output_shape[-1] // 128
            scale = torch.empty((output_shape[0], num_groups), dtype=torch.float32, device=input.device)
            
            torch.ops._C.silu_and_mul_per_block_quant.default(
                result, input, scale, 128, None, False
            )
            
            return result, scale
        
        input = torch.empty(5, 256, dtype=torch.float16, device='cuda')
        pattern(input)
        
        register_replacement(pattern, replacement, [input], fwd_only, pm_pass)
        
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
            print(f"Registering block quant pattern...")
            pattern_silu_mul_block = SiluMulBlockQuantPattern()
            pattern_silu_mul_block.register(self.patterns)
            print(f"✓ Registered block quant pattern")

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
