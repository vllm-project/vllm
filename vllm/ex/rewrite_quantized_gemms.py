import torch

from .utils import ModuleInputGenerator, graph_print_tabular, is_call, call_method_class

from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.subgraph_rewriter import replace_pattern
from torch.fx import symbolic_trace, subgraph_rewriter
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch
from torch.fx.passes.utils.matcher_with_name_node_map_utils import SubgraphMatcherWithNameNodeMap
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set, Type

from vllm.logger import init_logger
from vllm import _custom_ops as custom_ops

import traceback

logger = init_logger(__name__)

###############################################################################
#
# Rewrite quantized gemms
#
###############################################################################


def pattern(x, weight, weight_scale):
    f_x = x.to(torch.float32)
    f_w = weight.to(torch.float32) * weight_scale
    f_out = torch.nn.functional.linear(f_x, f_w.transpose(1, 0))
    return f_out.to(x.dtype)


def pattern2(f_x, f_w):
    f_w_t = f_w.transpose(1, 0)
    f_out = torch.nn.functional.linear(f_x, f_w_t)
    return f_out.to(f_x.dtype)


def fake_static_scaled_int8_quant(x_q, x, scale):
    x_q = x * scale
    return x_q


def fake_cutlass_scaled_mm_dq(out, a, b, a_scales, b_scales):
    return out


#torch._dynamo.disallow_in_graph(vllm_ops.static_scaled_int8_quant)
#torch._dynamo.disallow_in_graph(vllm_ops.cutlass_scaled_mm_dq)

#torch._dynamo.allow_in_graph(vllm_ops.static_scaled_int8_quant)
#torch._dynamo.allow_in_graph(vllm_ops.cutlass_scaled_mm_dq)


def _cutlass_scaled_mm_dq(a: torch.Tensor, b: torch.Tensor,
                          a_scales: torch.Tensor, b_scales: torch.Tensor,
                          out_dtype: Type[torch.dtype]) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[1]
    #out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    out = torch.empty_like(a)
    out.resize_((m, n))
    out = out.to(out_dtype)

    torch.ops._C.cutlass_scaled_mm_dq(out, a, b, a_scales, b_scales)
    #fake_cutlass_scaled_mm_dq(out, a, b, a_scales, b_scales)

    return out


def pattern3(x, w, w_scale, xtype):
    f_x = x.to(torch.float32)
    f_w = w.to(torch.float32) * w_scale
    f_out = torch.nn.functional.linear(f_x, f_w.transpose(1, 0))
    return f_out.to(xtype)


class replacement_class(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.x_scale = torch.empty((1, 1), dtype=torch.float32,
                                   device='cuda')  # temp hack

    def forward(self, x, w, w_scale, x_type):
        x_q = custom_ops.static_scaled_int8_quant(x, self.x_scale)
        return custom_ops.cutlass_scaled_mm_dq(x_q, w, self.x_scale, w_scale,
                                               x_type)


def llama_mlp_pattern(l_x_, weight, weight_1):
    gate_up = torch._C._nn.linear(l_x_, weight, None)
    getitem = gate_up[(Ellipsis, slice(None, 11008, None))]
    getitem_1 = gate_up[(Ellipsis, slice(11008, None, None))]
    silu = torch.nn.functional.silu(getitem)
    input_parallel = silu * getitem_1
    x_1 = torch._C._nn.linear(input_parallel, weight_1, None)
    return x_1


# TODO: try to register things differently?
def rewrite_quantized_gemms(
        mod: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor]) -> torch.fx.GraphModule:
    return mod
    # Disable for now
    pattern_graph = symbolic_trace(pattern3).graph

    #x = torch.empty((16,16), dtype=torch.float16)
    #x_scale = torch.empty((1,1), dtype=torch.float32, device='cuda')

    #w = torch.empty((x.size(0),16), dtype=torch.int8, device='cuda')
    #w_scale = torch.empty((w.size(1),), dtype=torch.float32, device='cuda')

    #x_type = torch.float16

    #m, n, k = 512, 512, 512

    #x = torch.empty((m, k), device="cuda", dtype=torch.float16)
    #x_scale = torch.empty((m, 1), device="cuda", dtype=torch.float32)

    #w = torch.empty((n, k), device="cuda", dtype=torch.int8).transpose(1, 0)
    #w_scale = torch.empty((1, n), device="cuda", dtype=torch.float32)

    #    replace_graph = symbolic_trace(replacement, {'x':x, 'w': w, 'x_scale':x_scale, 'w_scale':w_scale, 'x_type': x_type}).graph  # provide sample inputs?

    # See https://github.com/pytorch/pytorch/issues/93002
    # https://github.com/pytorch/pytorch/issues/120124
    #with torch._C.DisableTorchFunction():
    replacement = replacement_class()
    replace_graph = symbolic_trace(replacement).graph

    rep_matches = replace_pattern(mod, pattern_graph, replacement)
    print(f"root MATCHES {rep_matches}")

    if len(rep_matches) > 0:
        logger.debug(
            f"Rewritten module {mod}:\n{graph_print_tabular(mod.graph)}")

    llama_mod = symbolic_trace(llama_mlp_pattern)
    print(f"llama mod {llama_mod}")
    llama_mlp_pattern_graph = llama_mod.graph
    matcher = SubgraphMatcher(llama_mlp_pattern_graph, ignore_literals=True)
    matches = matcher.match(mod.graph)
    print(f"LLAMA MATCHES {matches}")

    #    for name, subm in mod.named_modules():
    #        matches = matcher.match(subm.graph)
    #        print(f"sub {name} MATCHES {matches}")

    return mod
