import copy
import torch
import unittest.mock

from .utils import ModuleInputGenerator, graph_print_tabular, is_call, call_method_class

from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.subgraph_rewriter import replace_pattern
from torch.fx import subgraph_rewriter #symbolic_trace
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch
from torch.fx.passes.utils.matcher_with_name_node_map_utils import SubgraphMatcherWithNameNodeMap
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set, Type, Union

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

def pattern4(x, w, x_scale, w_scale, xtype):
    torch._check(x_scale.numel() == 1)
    f_x = x.to(torch.float32)
    f_w = w.to(torch.float32) * w_scale
    f_out = torch.nn.functional.linear(f_x, f_w.transpose(1, 0))
    return f_out.to(xtype)

#torch.fx.wrap('torch.ops._C.static_scaled_int8_quant')
#torch.fx.wrap('torch.ops._C.cutlass_scaled_mm_dq')

def scale(x, x_scale):
    out = torch.empty_like(x, dtype=torch.int8)
    torch.ops._C.static_scaled_int8_quant(out, x, x_scale)
    return out

def replace3(x, w, w_scale, x_type):
    x_scale = torch.rand((1, 1), device=x.device, dtype=torch.float32)  # temp
    x_q, _ = custom_ops.scaled_int8_quant(x, x_scale)
    return custom_ops.cutlass_scaled_mm_dq(x_q, w, x_scale, w_scale, x_type)

def pattern3(x, w, w_scale, xtype):
    f_x = x.to(torch.float32)
    f_w = w.to(torch.float32) * w_scale
    f_out = torch.nn.functional.linear(f_x, f_w.transpose(1, 0))
    return f_out.to(xtype)

def replace4(x_scale):
    def replace_sub(x, w, w_scale, x_type):
        x_q, _ = custom_ops.scaled_int8_quant(x, x_scale)
        return custom_ops.cutlass_scaled_mm_dq(x_q, w, x_scale, w_scale, x_type)
    return replace_sub


class replacement_class(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.x_scale = torch.empty((1, 1), dtype=torch.float32,
                                   device='cuda')  # temp hack

    def forward(self, x, w, w_scale, x_type):
        x_q, _ = custom_ops.scaled_int8_quant(x, self.x_scale)
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

torch.fx.wrap(torch.ops._C.static_scaled_int8_quant)
#torch.fx.wrap('_C.cutlass_scaled_mm_dq')

class tracer:
    def __init__(self):
        self.mod = None

    def __call__(self, gm: torch.fx.GraphModule,
                 example_inputs: List[torch.Tensor]) -> Callable:
        self.mod = copy.copy(gm)
        return gm.forward


def symbolic_trace(
        fn: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        list_args: Optional[List[Any]] = None
):
    if True:
        t = tracer()
        cf = torch.compile(fn, backend=t)
        cf(*list_args)
        return t.mod.graph
    elif False:
        t = torch.fx.Tracer(autowrap_modules=(torch.math, torch.ops._C, torch.ops),
                            autowrap_functions=('vllm._custom_ops.scaled_int8_quant','vllm._custom_ops.cutlass_scaled_mm_dq',
                                                '_custom_ops.scaled_int8_quant','_custom_ops.cutlass_scaled_mm_dq',
                                                'custom_ops.scaled_int8_quant','custom_ops.cutlass_scaled_mm_dq',
                                                'scaled_int8_quant','cutlass_scaled_mm_dq',
                                                '_C.static_scaled_int8_quant','_C.cutlass_scaled_mm_dq'))
        return t.trace(fn, concrete_args)
    elif False:
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._functorch.eager_transforms import functionalize
        g = make_fx(functionalize(fn))(*list_args).graph
        #g.eliminate_dead_code()
        return g
    elif False:
        from torch.fx.experimental.meta_tracer import symbolic_trace
        return symbolic_trace(fn, {}, concrete_args).graph
    else:
        return torch.fx.symbolic_trace(fn, concrete_args).graph

from pytorch_symbolic import Input, SymbolicModel

# registering dynamic shapes
# See: https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html


def rewrite_quantized_gemms(
        mod: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor]) -> torch.fx.GraphModule:

    #x = torch.empty((16,16), dtype=torch.float16)
    #x_scale = torch.empty((1,1), dtype=torch.float32, device='cuda')

    #w = torch.empty((x.size(0),16), dtype=torch.int8, device='cuda')
    #w_scale = torch.empty((w.size(1),), dtype=torch.float32, device='cuda')

    x_type = torch.float16

    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch._dynamo.source import SyntheticLocalSource, EphemeralSource

    if True:
        m, n, k = 512, 512, 512
    else:
        senv = ShapeEnv()
        src = EphemeralSource("junk")
        m = senv.create_unspecified_symbol(senv.create_unbacked_symint(), src)
        n = senv.create_unspecified_symbol(senv.create_unbacked_symint(), src)
        k = senv.create_unspecified_symbol(senv.create_unbacked_symint(), src)

    if True:
        x = torch.empty((m, k), device="cuda", dtype=torch.float16)
        x_scale = torch.empty((1, 1), device="cuda", dtype=torch.float32)
        w = torch.empty((n, k), device="cuda", dtype=torch.int8).transpose(1, 0)
        w_scale = torch.empty((1, n), device="cuda", dtype=torch.float32)
    else:
        x = Input((m, k), dtype=torch.float16)
        x_scale = Input((1, 1), dtype=torch.float32)
        w = Input((k, n), dtype=torch.int8)
        w_scale = Input((1, n), dtype=torch.float32)

    pattern_graph = symbolic_trace(pattern3, {'x_scale':x_scale}, [x, w, w_scale, x_type])
    print(f"Pattern graph:\n{graph_print_tabular(pattern_graph)}\n")

    #replace_graph = symbolic_trace(replace3, {'x':x, 'w': w, 'x_scale':x_scale, 'w_scale':w_scale, 'x_type': x_type})
    #replace_graph = symbolic_trace(replace3, {'x':x, 'w': w, 'x_type': x_type})

    #fake_mode = torch._guards.detect_fake_mode()
    fake_mode = FakeTensorMode(allow_non_fake_inputs=False, shape_env=ShapeEnv())
    #with unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs", True):

    def mark(x):
        return x
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        return x

    if False:
        xsf = mark(fake_mode.from_tensor(x_scale))
        xf = mark(fake_mode.from_tensor(x))
        wf = mark(fake_mode.from_tensor(w))
        with fake_mode:
            replace_graph = symbolic_trace(replace4(xsf), {'x':xf, 'w': wf, 'x_type': x_type})
            #replace_graph = symbolic_trace(replace4(xsf), {'x_type': x_type})
    else:
        xsf = mark(x_scale)
        xf = mark(x)
        wf = mark(w)
        replace_graph = symbolic_trace(replace4(xsf), {'x':xf, 'w': wf, 'x_type': x_type}, [x, w, w_scale, x_type])
        #replace_graph = symbolic_trace(replace4(xsf), {'x_type': x_type})


    print(f"Replace graph:\n{graph_print_tabular(replace_graph)}\n")

    # See https://github.com/pytorch/pytorch/issues/93002
    # https://github.com/pytorch/pytorch/issues/120124
    #with torch._C.DisableTorchFunction():
    if False:
        replacement = replacement_class()
        replace_graph = symbolic_trace(replacement)

    rep_matches = replace_pattern(mod, pattern_graph, replace_graph)
    print(f"root MATCHES {rep_matches}")

    if len(rep_matches) > 0:
        logger.debug(
            f"Rewritten module {mod}:\n{graph_print_tabular(mod.graph)}")
        print(f"Rewritten module {mod}:\n{graph_print_tabular(mod.graph)}")
        return mod

    llama_mlp_pattern_graph = symbolic_trace(llama_mlp_pattern)
    matcher = SubgraphMatcher(llama_mlp_pattern_graph, ignore_literals=True)
    matches = matcher.match(mod.graph)
    print(f"LLAMA MATCHES {matches}")

    #    for name, subm in mod.named_modules():
    #        matches = matcher.match(subm.graph)
    #        print(f"sub {name} MATCHES {matches}")

    return mod
