###############################################################################
#
# Operator register
#
###############################################################################

import torch
import types

from typing import Callable, Optional, Union
from vllm.logger import init_logger

logger = init_logger(__name__)

# Set of supported operations. These will be partitioned into separate
# submodules by the backend.
SUPPORTED = set()

# Dictionary of fusable operations. The key is the operation name and
# the value indicates whether the operation is "compute" (e.g. gemm) or
# not.
FUSABLE = dict()

"""
Extract a string operator name from a Callable (or string).
"""
def operator_name(op: Union[str, Callable]) -> Optional[str]:
    if isinstance(op, str):
        return op
    elif (isinstance(op, types.FunctionType) or
          isinstance(op, types.BuiltinFunctionType) or
          isinstance(op, types.BuiltinMethodType)):
        return f"{op.__module__}.{op.__name__}"
    else:
        return None


"""
Register 'op' as an operation supported by the backend.
Can be used as a function decorator.
"""
def register_supported(op: Union[str, Callable]):
    op_name = operator_name(op)
    if op_name is None:
        raise RuntimeError(f"{op} has unsupported type.")
    logger.debug(f"register supported {op_name}")
    SUPPORTED.add(op)


"""
Register 'op' as an operation that can be fused with other fusable ops.
Can be used as a function decorator.
"""
def register_fusable(op: Union[str, Callable], is_compute: bool = False):
    op_name = operator_name(op)
    if op_name is None:
        raise RuntimeError(f"{op} has unsupported type.")
    assert op_name not in FUSABLE or FUSABLE[op_name] == is_compute
    logger.debug(f"register fusable {op_name}, is_compute {is_compute}")
    register_supported(op_name)
    FUSABLE[op_name] = is_compute


"""
Register default supported operations.
"""
def register_defaults():
    logger.debug("REGISTER DEFAULTS")
    register_fusable('_operator.add')
    register_fusable('_operator.mul')
    register_fusable('_operator.getitem')
    #register_fusable('torch.empty')  # maybe TBD
    register_fusable('torch.relu')
    register_fusable('torch.nn.functional.silu')
    register_fusable('torch.ops.vllm.silu_and_mul')
    register_fusable('torch.matmul', True)
    register_fusable('torch._C._nn.linear', True)


register_defaults()
