import copy
import torch
import unittest.mock

from .utils import ModuleInputGenerator, graph_print_tabular, is_call, call_method_class, node_function_target

from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.subgraph_rewriter import replace_pattern, replace_pattern_with_filters
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
# Quantization pass
#
###############################################################################

QUANTIZE_OPS = [
    'torch.ops._C.static_scaled_int8_quant',
    'torch.ops._C.dynamic_scaled_int8_quant',
    'torch.ops._C.static_scaled_fp8_quant',
    'torch.ops._C.dynamic_scaled_fp8_quant',
]

PASS_THRU_OPS = [
]

def is_quantize(n: torch.fx.Node) -> bool:
    return is_call(n) and node_function_target(n) in QUANTIZE_OPS

def move_quantization(
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor]
) -> torch.fx.GraphModule:

    quantize_nodes = []
    for n in mod.graph.nodes:
        if is_quantize(n):
            quantize_nodes.append(n)

    # find quantized ranges
    # while !done
    #   merge quantized ranges

    return mod
