# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, NamedTuple

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape, QuantKey, ScaleDesc, kFp8DynamicTensorSym, kFp8DynamicTokenSym,
    kFp8StaticTensorSym, kNvfp4Quant, kStaticTensorScale)
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float32, device="cuda")


def empty_i32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int32, device="cuda")


RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym:
    torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym:
    torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym:
    torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}
if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Quant] = torch.ops._C.scaled_fp4_quant.default


class FusedRMSQuantKey(NamedTuple):
    """
    Named tuple for identifying the type of RMSNorm + quant fusion.
    quant: type of quantization
    fused_add: does the op also perform the residual add
    """
    quant: QuantKey
    fused_add: bool

    def __str__(self):
        return (f"FusedQuantKey({self.quant}, with"
                f"{'' if self.fused_add else 'out'} residual)")


FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {
    FusedRMSQuantKey(kFp8StaticTensorSym, False):
    torch.ops._C.rms_norm_static_fp8_quant.default,  # noqa: E501
    FusedRMSQuantKey(kFp8StaticTensorSym, True):
    torch.ops._C.fused_add_rms_norm_static_fp8_quant.default,  # noqa: E501
    FusedRMSQuantKey(kFp8DynamicTokenSym, False):
    torch.ops._C.rms_norm_dynamic_per_token_quant.default,  # noqa: E501
    FusedRMSQuantKey(kFp8DynamicTokenSym, True):
    torch.ops._C.rms_norm_dynamic_per_token_quant.default,  # noqa: E501
}

if current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER:
    AITER_RMS_GROUP_QUANT_OP = \
        torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant.default
    AITER_RMS_ADD_GROUP_QUANT_OP = \
        torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant.default
    
    BLOCK_LINEAR_OP = torch.ops.vllm.apply_w8a8_block_fp8_linear.default
    AITER_BLOCK_LINEAR_OP = \
        torch.ops.vllm.rocm_aiter_gemm_w8a8_blockscale.default
    
    AITER_RMS_OP = torch.ops.vllm.rocm_aiter_rms_norm.default
    AITER_RMS_ADD_OP = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default
    
    import aiter as rocm_aiter
    rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8
    rocm_aiter_fp8_quant_group_size = 128

class RMSNormQuantPattern:

    def __init__(self, epsilon: float, key: FusedRMSQuantKey):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype

        assert key.quant in QUANT_OPS, \
            f"unsupported quantization scheme {key.quant}"
        self.QUANT_OP = QUANT_OPS[key.quant]

        assert key in FUSED_OPS, \
            f"unsupported fused rmsnorm+quant op for {key}"
        self.FUSED_OP = FUSED_OPS[key]


class RMSNormStaticQuantPattern(RMSNormQuantPattern):

    def __init__(self,
                 epsilon: float,
                 quant_dtype: torch.dtype,
                 symmetric=True):
        fused_key = FusedRMSQuantKey(fused_add=False,
                                     quant=QuantKey(dtype=quant_dtype,
                                                    scale=kStaticTensorScale,
                                                    symmetric=symmetric))
        super().__init__(epsilon, fused_key)

    def register(self, pm_pass: PatternMatcherPass):
        # Cannot use methods, as the self argument affects tracing
        def pattern(result: torch.Tensor, result_rms: torch.Tensor,
                    input: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at1 = auto_functionalized(RMS_OP,
                                      result=result_rms,
                                      input=input,
                                      weight=weight,
                                      epsilon=self.epsilon)
            at2 = auto_functionalized(self.QUANT_OP,
                                      result=result,
                                      input=at1[1],
                                      scale=scale)

            # result
            return at2[1]

        def replacement(result: torch.Tensor, result_rms: torch.Tensor,
                        input: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP,
                                     result=result,
                                     input=input,
                                     weight=weight,
                                     scale=scale,
                                     epsilon=self.epsilon)

            # result
            return at[1]

        inputs = [
            torch.empty(5, 4, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # result_rms
            empty_bf16(5, 4),  # input
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1)  # scale
        ]

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only,
                                pm_pass)


class FusedAddRMSNormStaticQuantPattern(RMSNormQuantPattern):

    def __init__(self,
                 epsilon: float,
                 quant_dtype: torch.dtype,
                 symmetric=True):
        key = FusedRMSQuantKey(fused_add=True,
                               quant=QuantKey(dtype=quant_dtype,
                                              scale=kStaticTensorScale,
                                              symmetric=symmetric))
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(result: torch.Tensor, input: torch.Tensor,
                    residual: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at = auto_functionalized(RMS_ADD_OP,
                                     input=input,
                                     residual=residual,
                                     weight=weight,
                                     epsilon=self.epsilon)
            at1 = auto_functionalized(self.QUANT_OP,
                                      result=result,
                                      input=at[1],
                                      scale=scale)

            # result, residual
            return at1[1], at[2]

        def replacement(result: torch.Tensor, input: torch.Tensor,
                        residual: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP,
                                     result=result,
                                     input=input,
                                     residual=residual,
                                     weight=weight,
                                     scale=scale,
                                     epsilon=self.epsilon)

            # result, residual
            return at[1], at[2]

        inputs = [
            torch.empty(5, 4, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # input
            empty_bf16(5, 4),  # residual
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1)  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class RMSNormDynamicQuantPattern(RMSNormQuantPattern):

    def __init__(self,
                 epsilon: float,
                 quant_dtype: torch.dtype,
                 group_shape: GroupShape = GroupShape.PER_TOKEN,
                 symmetric=True):
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(fused_add=False,
                               quant=QuantKey(dtype=quant_dtype,
                                              scale=scale,
                                              symmetric=symmetric))
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(result: torch.Tensor, result_rms: torch.Tensor,
                    input: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at1 = auto_functionalized(RMS_OP,
                                      result=result_rms,
                                      input=input,
                                      weight=weight,
                                      epsilon=self.epsilon)
            at2 = auto_functionalized(self.QUANT_OP,
                                      result=result,
                                      input=at1[1],
                                      scale=scale,
                                      scale_ub=None)

            # result, scale
            return at2[1], at2[2]

        def replacement(result: torch.Tensor, result_rms: torch.Tensor,
                        input: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP,
                                     result=result,
                                     input=input,
                                     weight=weight,
                                     scale=scale,
                                     epsilon=self.epsilon,
                                     scale_ub=None,
                                     residual=None)

            # result, scale
            return at[1], at[2]

        inputs = [
            torch.empty(5, 4, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # result_rms
            empty_bf16(5, 4),  # input
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1)  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class FusedAddRMSNormDynamicQuantPattern(RMSNormQuantPattern):

    def __init__(self,
                 epsilon: float,
                 quant_dtype: torch.dtype,
                 group_shape: GroupShape = GroupShape.PER_TOKEN,
                 symmetric=True):
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(fused_add=True,
                               quant=QuantKey(dtype=quant_dtype,
                                              scale=scale,
                                              symmetric=symmetric))
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(result: torch.Tensor, input: torch.Tensor,
                    residual: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at = auto_functionalized(RMS_ADD_OP,
                                     input=input,
                                     residual=residual,
                                     weight=weight,
                                     epsilon=self.epsilon)
            at1 = auto_functionalized(self.QUANT_OP,
                                      result=result,
                                      input=at[1],
                                      scale=scale,
                                      scale_ub=None)

            # result, residual, scale
            return at1[1], at[2], at1[2]

        def replacement(result: torch.Tensor, input: torch.Tensor,
                        residual: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(self.FUSED_OP,
                                     result=result,
                                     input=input,
                                     weight=weight,
                                     scale=scale,
                                     epsilon=self.epsilon,
                                     scale_ub=None,
                                     residual=residual)

            # result, residual, scale
            return at[1], at[3], at[2]

        inputs = [
            torch.empty(5, 4, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # input
            empty_bf16(5, 4),  # residual
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1)  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )
        
        
class AiterRMSGroupQuantFP8Pattern():

    def __init__(self,
                 epsilon: float,
                 quant_dtype: torch.dtype):
        self.epsilon = epsilon
        self.quant_dtype = quant_dtype

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(input: torch.Tensor, weight: torch.Tensor, #result_rms: torch.Tensor,
                    linear_weight: torch.Tensor,
                    linear_weight_scale: torch.Tensor):
            at1 = AITER_RMS_OP(x=input,
                               weight=weight,
                               variance_epsilon=self.epsilon)
            
            at2 = BLOCK_LINEAR_OP(input=at1,
                                    weight=linear_weight,
                                    block_size=[128, 128],
                                    weight_scale=linear_weight_scale,
                                    input_scale=None,
                                    bias=None,
                                    cutlass_block_fp8_supported=False,
                                    use_aiter_and_is_supported=True)

            return at2

        def replacement(input: torch.Tensor, weight: torch.Tensor,
                        linear_weight: torch.Tensor,
                        linear_weight_scale: torch.Tensor):
            at1 = AITER_RMS_GROUP_QUANT_OP(x=input,
                                           residual=None,
                                           weight=weight,
                                           variance_epsilon=self.epsilon)
            
            at2 = AITER_BLOCK_LINEAR_OP(A=at1[0],
                                        B=linear_weight,
                                        As=at1[1],
                                        Bs=linear_weight_scale,
                                        block_size=[128, 128],
                                        output_dtype=input.dtype)

            return at2

        inputs = [
            empty_bf16(5, 4),  # input
            empty_bf16(1, 5),  # weight
            torch.empty((2, 5), device="cuda", dtype=FP8_DTYPE), # linear_weight
            empty_fp32(1, 1),  # linear_weight_scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass)


class AiterFusedAddRMSGroupQuantPattern():

    def __init__(self,
                 epsilon: float,
                 quant_dtype: torch.dtype):
        self.epsilon = epsilon
        self.quant_dtype = quant_dtype

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                    linear_weight: torch.Tensor,
                    linear_weight_scale: torch.Tensor):
            at1 = AITER_RMS_ADD_OP(x=input,
                                   residual=residual,
                                   weight=weight,
                                   variance_epsilon=self.epsilon)
            
            at2 = BLOCK_LINEAR_OP(input=at1[0],
                                  weight=linear_weight,
                                  block_size=[128, 128],
                                  weight_scale=linear_weight_scale,
                                  input_scale=None,
                                  bias=None,
                                  cutlass_block_fp8_supported=False,
                                  use_aiter_and_is_supported=True)
            # result, residual
            return at2, at1[1]

        def replacement(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                        linear_weight: torch.Tensor,
                        linear_weight_scale: torch.Tensor):

            at1 = AITER_RMS_ADD_GROUP_QUANT_OP(x=input,
                                               residual=residual,
                                               weight=weight,
                                               variance_epsilon=self.epsilon)
            
            at2 = AITER_BLOCK_LINEAR_OP(A=at1[0],
                                        B=linear_weight,
                                        As=at1[1],
                                        Bs=linear_weight_scale,
                                        block_size=[128, 128],
                                        output_dtype=input.dtype)
            # result, residual
            return at2, at1[2]

        inputs = [
            empty_bf16(5, 4),  # input
            empty_bf16(5, 4),  # residual
            empty_bf16(1, 5),  # weight
            torch.empty((2, 5), device="cuda", dtype=FP8_DTYPE), # linear_weight
            empty_fp32(1, 1), # linear_weight_scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass)


class RMSNormQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses rms_norm & quant custom ops into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rmsnorm_quant_fusion_pass")

        for epsilon in [1e-5, 1e-6]:
            # Fuse rms_norm + static fp8 quant
            RMSNormStaticQuantPattern(epsilon,
                                      FP8_DTYPE).register(self.patterns)

            # Fuse fused_add_rms_norm + static fp8 quant
            FusedAddRMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns)

            # Fuse rms_norm + dynamic per-token fp8 quant
            RMSNormDynamicQuantPattern(epsilon,
                                       FP8_DTYPE).register(self.patterns)

            # Fuse fused_add_rms_norm + dynamic per-token fp8 quant
            FusedAddRMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns)
            
            if envs.VLLM_ROCM_USE_AITER:
                # Fuse rms_norm + dynamic group fp8 quant
                AiterRMSGroupQuantFP8Pattern(epsilon, FP8_DTYPE).register(
                    self.patterns)
                
                AiterFusedAddRMSGroupQuantPattern(epsilon, FP8_DTYPE).register(
                    self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        return self.hash_source(self, RMSNormQuantPattern,
                                RMSNormStaticQuantPattern,
                                RMSNormDynamicQuantPattern,
                                FusedAddRMSNormStaticQuantPattern,
                                FusedAddRMSNormDynamicQuantPattern,
                                AiterRMSGroupQuantFP8Pattern,
                                AiterFusedAddRMSGroupQuantPattern)
