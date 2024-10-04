import logging

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.logger import init_logger

logger = init_logger(__name__)
logger.setLevel(logging.DEBUG)  # TODO


# DYNAMIC
@torch.library.custom_op("vllm::fused_rms_norm_quant_dynamic", mutates_args=['result', 'scale', 'azp'])
def fused_rms_norm_quant_dynamic(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                                 azp: torch.Tensor, epsilon: float) -> None:
    print("vllm::fused_rms_norm_quant_dynamic")
    result_rms = torch.empty_like(input)
    torch.ops._C.rms_norm(result_rms, input, weight, epsilon)
    torch.ops._C.dynamic_scaled_int8_quant(result, result_rms, scale, azp)


@torch.library.register_fake("vllm::fused_rms_norm_quant_dynamic")
def _(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, azp: torch.Tensor,
      epsilon: float) -> None:
    return


# TODO epsilon
def rms_pattern(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops._C.rms_norm.default, result=result_rms, input=input, weight=weight,
                              epsilon=1e-6)
    at2 = auto_functionalized(torch.ops._C.dynamic_scaled_int8_quant.default, result=result, input=at1[1], scale=scale,
                              azp=None)

    # result, scale
    # TODO azp
    return at2[1:2]


def rms_replacement(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
    at = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.fused_rms_norm_quant_dynamic.default, result=result,
                                                    input=input, weight=weight,
                                                    epsilon=1e-6, scale=scale, azp=None)

    # result, scale
    # TODO azp
    return at[1:2]


# STATIC
@torch.library.custom_op("vllm::fused_rms_norm_quant_static", mutates_args=['result'])
def fused_rms_norm_quant_static(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                                azp: torch.Tensor, epsilon: float) -> None:
    print("vllm::fused_rms_norm_quant_static")
    result_rms = torch.empty_like(input)
    torch.ops._C.rms_norm(result_rms, input, weight, epsilon)
    torch.ops._C.static_scaled_int8_quant(result, result_rms, scale, azp)


@torch.library.register_fake("vllm::fused_rms_norm_quant_static")
def _(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, azp: torch.Tensor,
      epsilon: float) -> None:
    return


def rms_pattern_static(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                       scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops._C.rms_norm.default, result=result_rms, input=input, weight=weight,
                              epsilon=1e-5)
    at2 = auto_functionalized(torch.ops._C.static_scaled_int8_quant.default, result=result, input=at1[1], scale=scale,
                              azp=None)

    # result
    return at2[1]


def rms_replacement_static(result: torch.Tensor, result_rms: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                           scale: torch.Tensor):
    at = auto_functionalized(torch.ops.vllm.fused_rms_norm_quant_static.default, result=result, input=input,
                             weight=weight,
                             epsilon=1e-5, scale=scale, azp=None)

    # result
    return at[1]


@torch.library.custom_op("vllm::fused_rms_norm_residual_quant_static", mutates_args=['result', 'input', 'residual'])
def fused_rms_norm_residual_quant_static(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor,
                                         weight: torch.Tensor, scale: torch.Tensor, azp: torch.Tensor,
                                         epsilon: float) -> None:
    # print("vllm::fused_rms_norm_residual_quant_static")
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)
    torch.ops._C.static_scaled_int8_quant(result, input, scale, azp)


@torch.library.register_fake("vllm::fused_rms_norm_residual_quant_static")
def _(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
      azp: torch.Tensor, epsilon: float) -> None:
    return


def rms_pattern_residual_static(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                                scale: torch.Tensor):
    at1 = auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input=input, residual=residual, weight=weight,
                              epsilon=1e-5)
    at2 = auto_functionalized(torch.ops._C.static_scaled_int8_quant.default, result=result, input=at1[1], scale=scale,
                              azp=None)

    # result, residual
    return at2[1], at1[2]


def rms_replacement_residual_static(result: torch.Tensor, input: torch.Tensor, residual: torch.Tensor,
                                    weight: torch.Tensor, scale: torch.Tensor):
    at = auto_functionalized(torch.ops.vllm.fused_rms_norm_residual_quant_static.default, result=result, input=input,
                             residual=residual, weight=weight, epsilon=1e-5, scale=scale, azp=None)
    # result, residual
    return at[1], at[3]


my_patterns = PatternMatcherPass()


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_int8(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int8, device="cuda")


def get_patterns():
    my_patterns = PatternMatcherPass()

    inputs = [empty_int8(5, 4), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(1, 5), torch.empty(1, 1, device="cuda")]
    register_replacement(rms_pattern, rms_replacement, inputs, fwd_only, my_patterns)
    register_replacement(rms_pattern_static, rms_replacement_static, inputs, fwd_only, my_patterns)

    # with residual
    inputs = [empty_int8(5, 4), empty_bf16(5, 4), empty_bf16(5, 4), empty_bf16(1, 5), torch.empty(1, 1, device="cuda")]
    register_replacement(rms_pattern_residual_static, rms_replacement_residual_static, inputs, fwd_only, my_patterns)

    return my_patterns


def get_fusion_pass():
    patterns = get_patterns()

    def fusion_pass(graph: torch.fx.Graph):
        """
        Use the pattern matcher
        """
        # logger.info("Graph before fusion pass:")
        with open("before_fusion.py", "w") as f:
            print(graph.python_code(root_module="self", verbose=True).src, file=f)
        count = patterns.apply(graph)
        logger.info(f"Replaced {count} patterns")
        with open("after_fusion.py", "w") as f:
            print(graph.python_code(root_module="self", verbose=True).src, file=f)

    return fusion_pass
