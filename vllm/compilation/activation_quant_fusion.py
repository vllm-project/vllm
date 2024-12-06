from typing import Optional

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (PatternMatcherPass, fwd_only,
                                             register_replacement)

from vllm.config import CompilationConfig
from vllm.logger import init_logger

from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


def silu_mul_pattern_static(result: torch.Tensor, result_silu_mul: torch.Tensor,
                            input: torch.Tensor, scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops._C.silu_and_mul.default,
                              result=result_silu_mul,
                              input=input)
    at2 = auto_functionalized(torch.ops._C.static_scaled_fp8_quant.default,
                              result=result,
                              input=at1[1],
                              scale=scale)
    return at2[1]


def silu_mul_replacement_static(result: torch.Tensor,
                                result_silu_mul: torch.Tensor,
                                input: torch.Tensor, scale: torch.Tensor):
    at = auto_functionalized(torch.ops._C.silu_and_mul_quant.default,
                             result=result,
                             input=input,
                             scale=scale)
    return at[1]


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp8(*args, **kwargs):
    fp8 = torch.float8_e4m3fn
    return torch.empty(*args, **kwargs, dtype=fp8, device="cuda")


def empty_fp32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float32, device="cuda")


class ActivationQuantFusionPass(VllmInductorPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.
    It also manually processes multi-output matches, as those are broken in
    the torch pattern matcher.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    _instance: 'Optional[ActivationQuantFusionPass]' = None

    @classmethod
    def instance(cls, config: CompilationConfig.PassConfig):
        """
        Get the singleton instance of the ActivationQuantFusionPass.
        If the instance exists, the config is updated but
        initialization is not repeated.
        """
        if cls._instance is None:
            cls._instance = ActivationQuantFusionPass(config)
        else:
            cls._instance.config = config
        return cls._instance

    def __init__(self, config: CompilationConfig.PassConfig):
        assert self.__class__._instance is None, \
            "ActivationQuantFusionPass singleton instance already exists"
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="activation_quant_fusion_pass")

        inputs = [
            empty_fp8(5, 4),  # Quant output
            empty_bf16(5, 4),  # Silu_and_mul output
            empty_bf16(5, 4),  # Input
            empty_fp32(1, 1)  # Scale
        ]
        register_replacement(silu_mul_pattern_static,
                             silu_mul_replacement_static, inputs, fwd_only,
                             self.patterns)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_act_quant_fusion")

        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns in ActivationQuantFusionPass",
                     count)
        self.dump_graph(graph, "after_pattern_match")

        self.dump_graph(graph, "after_act_quant_fusion")
        self.end_and_log()
