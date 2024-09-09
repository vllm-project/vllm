###############################################################################
#
# Operator register
#
###############################################################################

import types
from typing import Callable, Dict, Optional, Set, Tuple, Type, Union

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class OpSupport:
    """
    A class containing information about registered ops.
    - op_name    - operator name
    - is_fusable - can the operator be fused with other operators?
    - is_compute - is this a operation that does significant compute, e.g. gemm?
    - is_trivial - is this op "trivial"?  stacks of only trivial fusable ops
                   will not be fused.
    """

    def __init__(self,
                 op_name: Union[str, Callable],
                 is_fusable: bool = False,
                 is_compute: bool = False,
                 is_trivial: bool = False):
        self.op_name = operator_name(op_name)
        self.is_fusable = is_fusable
        self.is_compute = is_compute
        self.is_trivial = is_trivial


# Set of supported operations. These will be partitioned into separate
# submodules by the backend.
SUPPORTED: Dict[str, Set[str]] = dict()

# Dictionary of fusable operations. The key is the operation name and
# the value is an OpSupport instance.
FUSABLE: Dict[str, OpSupport] = dict()


def operator_name(
    op: Union[str,
              Callable]) -> Tuple[Optional[Union[str, Type]], Optional[str]]:
    """
    Extract a string operator name from a Callable (or string).
    """
    if isinstance(op, str):
        return (None, op)
    elif isinstance(op, (types.FunctionType, types.BuiltinFunctionType)):
        return (None, f"{op.__module__}.{op.__name__}")
    elif isinstance(op, types.BuiltinMethodType):
        return (op.__module__, op.__name__)
    elif isinstance(op, types.MethodDescriptorType):
        return (op.__objclass__, op.__name__)
    else:
        return (None, None)


def register_supported(op: Union[str, Callable]):
    """
    Register 'op' as an operation supported by the backend.
    Can be used as a function decorator.
    """
    class_name, op_name = operator_name(op)
    if op_name is None:
        raise RuntimeError(f"{op} has unsupported type.")
    logger.debug("register supported %s/%s", str(class_name), op_name)
    if op_name not in SUPPORTED:
        SUPPORTED[op_name] = set()
    if class_name:
        SUPPORTED[op_name].add(str(class_name))


def register_fusable(op: Union[str, Callable],
                     is_compute: bool = False,
                     is_trivial: bool = False):
    """
    Register 'op' as an operation that can be fused with other fusable ops.
    Can be used as a function decorator.
    """
    class_name, op_name = operator_name(op)
    if op_name is None:
        raise RuntimeError(f"{op} has unsupported type.")
    assert op_name not in FUSABLE
    logger.debug("register fusable %s, is_compute %s, is_trivial %s", op_name,
                 is_compute, is_trivial)
    register_supported(op)

    # TODO: need to register classes for methods
    FUSABLE[op_name] = OpSupport(op_name, True, is_compute, is_trivial)


def register_defaults():
    """
    Register default supported operations.
    """
    logger.debug("REGISTER DEFAULTS")
    # Note: methods need to be supported via function object and not name.
    register_fusable(torch.Tensor.to)
    register_fusable(torch.Tensor.size, is_trivial=True)
    register_fusable(torch.Tensor.transpose, is_trivial=True)
    register_fusable(torch.Tensor.numel, is_trivial=True)
    register_fusable('_operator.add')
    register_fusable('_operator.mul')
    register_fusable('_operator.floordiv')
    register_fusable('_operator.getitem', is_trivial=True)
    register_fusable('torch.empty', is_trivial=True)
    register_fusable('torch.empty_like', is_trivial=True)
    register_fusable('torch.relu')
    register_fusable('torch.narrow', is_trivial=True)
    register_fusable('torch.nn.functional.silu')
    register_fusable('torch.matmul', is_compute=True)
    register_fusable('torch.ops._C.silu_and_mul')
    register_fusable('torch.ops._C.static_scaled_int8_quant')
    register_fusable('torch.ops._C.static_scaled_fp8_quant')
    register_fusable('torch.ops._C.dynamic_scaled_int8_quant')
    register_fusable('torch.ops._C.dynamic_scaled_fp8_quant')
    register_fusable('torch.ops._C.fused_add_rms_norm', is_compute=True)
    register_fusable('torch.ops._C.rms_norm', is_compute=True)
    register_fusable('torch._C._nn.linear', is_compute=True)
    register_fusable('torch.ops._C.cutlass_scaled_mm', is_compute=True)


register_defaults()
