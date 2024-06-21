import torch

from .utils import graph_print_tabular, is_call, call_method_class, node_function_target

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set, Type, Union

from vllm.logger import init_logger

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
