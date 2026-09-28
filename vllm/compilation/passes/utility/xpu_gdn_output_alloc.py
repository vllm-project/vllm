# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..fx_utils import is_func
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


def _is_zero_fill(node: object) -> bool:
    return (
        isinstance(node, fx.Node)
        and is_func(node, torch.ops.aten.full.default)
        and len(node.args) == 2
        and node.args[1] == 0
    )


class XpuGdnOutputAllocPass(VllmInductorPass):
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
