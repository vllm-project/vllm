# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from torch import fx
from torch._ops import OpOverload, OpOverloadPacket

from vllm.ir.op import IrOp
from vllm.logger import init_logger

logger = init_logger(__name__)


def overload_or_default(op: OpOverload | OpOverloadPacket) -> OpOverload:
    if isinstance(op, OpOverloadPacket):
        return op.default
    assert isinstance(op, OpOverload), "Expected an OpOverload or OpOverloadPacket"
    return op


def get_ir_op(node: fx.Node) -> IrOp | None:
    if node.op != "call_function":
        return None

    if not isinstance(node.target, (OpOverload, OpOverloadPacket)):
        return None

    op_overload = overload_or_default(node.target)
    if op_overload.namespace != "vllm_ir":
        return None

    op_name = op_overload._opname
    if op_name not in IrOp.registry:
        logger.warning(
            "Unknown vLLM IR op %s, there's likely an issue with torch registration, "
            "or a torch custom op was registered in the vllm_ir namespace by mistake.",
            op_name,
        )
        return None

    ir_op = IrOp.registry[op_name]
    return ir_op
