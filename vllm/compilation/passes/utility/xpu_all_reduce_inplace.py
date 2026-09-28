# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make the XPU tensor-parallel all-reduce in-place where it is safe.

``vllm::all_reduce`` is out-of-place: the XPU communicator clones its input
and then calls ``torch.distributed.all_reduce`` on the copy. When the input
is a fresh (non-view, contiguous) intermediate whose only user is the
all-reduce, the copy is redundant. This pass rewrites

    y = vllm.all_reduce(x, group_name)

to

    y = auto_functionalized(vllm.xpu_all_reduce_, x=x, group_name=group_name)[1]

Inductor re-inplaces the mutation (x has no other user and is not a graph
input), so no copy remains, and the op calls the process group directly.
"""

import operator

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..fx_utils import is_func
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


def _xpu_all_reduce_inplace_available() -> bool:
    return hasattr(torch.ops.vllm, "xpu_all_reduce_")


def _is_fresh_tensor(node: fx.Node) -> bool:
    """Whether node is a call to an op that returns a new, non-aliasing,
    contiguous tensor (so mutating it cannot affect any other value)."""
    if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
        return False
    schema = node.target._schema
    if schema.is_mutable or len(schema.returns) != 1:
        return False
    if schema.returns[0].alias_info is not None:
        return False
    val = node.meta.get("val")
    return isinstance(val, torch.Tensor) and val.is_contiguous()


class XpuAllReduceInplacePass(VllmInductorPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.enabled = _xpu_all_reduce_inplace_available()
        if not self.enabled:
            logger.warning_once(
                "XpuAllReduceInplacePass disabled: vllm::xpu_all_reduce_ is not "
                "registered."
            )
        self.matched_count = 0

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        if not self.enabled:
            return
        for node in list(graph.nodes):
            if not is_func(node, torch.ops.vllm.all_reduce.default):
                continue
            x, group_name = node.args[0], node.args[1]
            if not isinstance(x, fx.Node) or len(x.users) != 1:
                continue
            if not _is_fresh_tensor(x):
                continue
            with graph.inserting_before(node):
                af = graph.call_function(
                    auto_functionalized,
                    (torch.ops.vllm.xpu_all_reduce_.default,),
                    {"x": x, "group_name": group_name},
                )
                out = graph.call_function(operator.getitem, (af, 1))
            af.meta["val"] = (None, x.meta["val"])
            out.meta["val"] = node.meta.get("val", x.meta["val"])
            node.replace_all_uses_with(out)
            graph.erase_node(node)
            self.matched_count += 1
        logger.debug(
            "XpuAllReduceInplacePass rewrote %d all-reduces", self.matched_count
        )

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, _is_fresh_tensor)
