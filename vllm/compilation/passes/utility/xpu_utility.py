# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU graph utility passes.

* `XpuAllReduceInplacePass`: make the TP all-reduce in-place where it is safe
  (compile ranges ending at <= 8 tokens only).
* `XpuGdnOutputAllocPass`: drop the zero-fill of the GDN core output buffer.
"""

import operator

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

from ..fx_utils import is_func
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

MAX_TOKEN_NUM = 8


def _xpu_all_reduce_inplace_available() -> bool:
    return hasattr(torch.ops.vllm, "xpu_all_reduce_")


def _returns_fresh_tensors(node: fx.Node) -> bool:
    """Whether node calls a non-mutating op none of whose results alias."""
    if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
        return False
    schema = node.target._schema
    return not schema.is_mutable and all(r.alias_info is None for r in schema.returns)


def _is_fresh_tensor(node: fx.Node) -> bool:
    """Whether node is a new, non-aliasing, contiguous tensor (a single-result
    op or one result of a multi-result op), so mutating it cannot affect any
    other value."""
    producer = node
    if node.op == "call_function" and node.target is operator.getitem:
        producer = node.args[0]
        if not isinstance(producer, fx.Node):
            return False
    elif len(getattr(getattr(node.target, "_schema", None), "returns", ())) != 1:
        return False
    if not _returns_fresh_tensors(producer):
        return False
    val = node.meta.get("val")
    return isinstance(val, torch.Tensor) and val.is_contiguous()


class XpuAllReduceInplacePass(VllmInductorPass):
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

    Only applied for compile ranges ending at <= 8 tokens (decode), where the
    copy and the Python wrapper are a noticeable part of the step. Larger ranges
    keep ``vllm::all_reduce``: in-place XCCL all-reduces on the large prefill /
    profiling intermediates leave XCCL holding extra device memory, which comes
    out of the KV cache.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.enabled = _xpu_all_reduce_inplace_available()
        if not self.enabled:
            logger.warning_once(
                "XpuAllReduceInplacePass disabled: vllm::xpu_all_reduce_ is not "
                "registered."
            )
        self.matched_count = 0

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.enabled and compile_range.end <= MAX_TOKEN_NUM

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
        return (
            VllmInductorPass.hash_source(self, _is_fresh_tensor, _returns_fresh_tensors)
            + f"|{MAX_TOKEN_NUM}"
        )


def _is_zero_fill(node: object) -> bool:
    return (
        isinstance(node, fx.Node)
        and is_func(node, torch.ops.aten.full.default)
        and len(node.args) == 2
        and node.args[1] == 0
    )


class XpuGdnOutputAllocPass(VllmInductorPass):
    """Drop the zero-fill of the XPU GDN core output buffer.

    QwenGatedDeltaNetAttention.forward_xpu allocates ``core_attn_out`` with
    ``torch.zeros`` before ``vllm::gdn_attention_core_xpu`` fills it. The XPU op
    defines every row of that buffer itself (the kernel writes all active
    tokens and the op zeroes any rows it does not write, including the whole
    buffer when there is no attention metadata), so the zero-fill is redundant.
    This pass turns the ``aten.full(shape, 0)`` feeding the op's
    ``core_attn_out`` (and nothing else) into ``aten.empty``, saving one
    elementwise kernel per GDN layer.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.enabled = hasattr(torch.ops.vllm, "gdn_attention_core_xpu")
        self.matched_count = 0

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        if not self.enabled:
            return
        target = torch.ops.vllm.gdn_attention_core_xpu.default
        for node in list(graph.nodes):
            if not (is_func(node, auto_functionalized) and node.args[0] is target):
                continue
            buf = node.kwargs.get("core_attn_out")
            if not _is_zero_fill(buf) or len(buf.users) != 1:
                continue
            with graph.inserting_before(buf):
                empty = graph.call_function(
                    torch.ops.aten.empty.memory_format,
                    (buf.args[0],),
                    dict(buf.kwargs),
                )
            empty.meta = dict(buf.meta)
            buf.replace_all_uses_with(empty)
            graph.erase_node(buf)
            self.matched_count += 1
        logger.debug("XpuGdnOutputAllocPass replaced %d zero-fills", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, _is_zero_fill)
