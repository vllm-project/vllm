###############################################################################
#
# Operator register
#
###############################################################################

import torch
import types

from typing import Callable, Optional, Union, Dict, Set, Tuple
from vllm.logger import init_logger

logger = init_logger(__name__)


class OpSupport:
    def __init__(
        self,
        op_name: Union[str, Callable],
        is_fusable: bool = False,
        is_compute: bool = False
    ):
        self.op_name = operator_name(op_name)


# Set of supported operations. These will be partitioned into separate
# submodules by the backend.
SUPPORTED  : Dict[str, Optional[Set[str]]] = dict()

# Dictionary of fusable operations. The key is the operation name and
# the value indicates whether the operation is "compute" (e.g. gemm) or
# not.
FUSABLE = dict()


"""
Extract a string operator name from a Callable (or string).
"""
def operator_name(op: Union[str, Callable]) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(op, str):
        return (None, op)
    elif (isinstance(op, types.FunctionType) or
          isinstance(op, types.BuiltinFunctionType)):
        return (None, f"{op.__module__}.{op.__name__}")
    elif isinstance(op, types.BuiltinMethodType):
        return (op.__module__, op.__name__)
    elif isinstance(op, types.MethodDescriptorType):
        return (op.__objclass__, op.__name__)
    else:
        return (None, None)


"""
Register 'op' as an operation supported by the backend.
Can be used as a function decorator.
"""
def register_supported(op: Union[str, Callable]):
    class_name, op_name = operator_name(op)
    if op_name is None:
        raise RuntimeError(f"{op} has unsupported type.")
    logger.debug(f"register supported {class_name}/{op_name}")
    if class_name:
        if not op_name in SUPPORTED:
            SUPPORTED[op_name] = set()
        SUPPORTED[op_name].add(class_name)
    else:
        SUPPORTED[op_name] = None


"""
Register 'op' as an operation that can be fused with other fusable ops.
Can be used as a function decorator.
"""
def register_fusable(op: Union[str, Callable], is_compute: bool = False):
    class_name, op_name = operator_name(op)
    if op_name is None:
        raise RuntimeError(f"{op} has unsupported type.")
    assert op_name not in FUSABLE or FUSABLE[op_name] == is_compute
    logger.debug(f"register fusable {op_name}, is_compute {is_compute}")
    register_supported(op)

    # TODO: need to register classes for methods
    FUSABLE[op_name] = is_compute


"""
Register default supported operations.
"""
def register_defaults():
    logger.debug("REGISTER DEFAULTS")
    register_fusable(torch.Tensor.to)
    register_fusable(torch.Tensor.transpose)
    register_fusable('_operator.add')
    register_fusable('_operator.mul')
    register_fusable('_operator.getitem')
    #register_fusable('torch.empty')  # maybe TBD
    register_fusable('torch.relu')
    register_fusable('torch.nn.functional.silu')
    register_fusable('torch.ops.vllm.silu_and_mul')
    register_fusable('torch.matmul', True)
    register_fusable('torch._C._nn.linear', True)
    #register_fusable('torch.ops.vllm.cutlass_scaled_mm_dq', True)


register_defaults()
