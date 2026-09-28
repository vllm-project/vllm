# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run two XPU fp8 linears that share their input as one op.

Rewrites exactly two ``_xpu_C.fp8_gemm_w8a16(a, B_i, scale_i, None)``
calls on the same activation ``a`` (e.g. the GDN in_proj_qkvz and
in_proj_ba projections) into one ``_xpu_C.fp8_gemm_w8a16_pair`` call, which
runs a single GEMV launch for decode-sized inputs and two oneDNN GEMMs
otherwise. Only applied for compile ranges whose end is <= 8 tokens.
"""

import operator

import torch
from torch import fx

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

from ..fx_utils import is_func
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

MAX_TOKEN_NUM = 8


def _pair_op_available() -> bool:
    return hasattr(torch.ops, "_xpu_C") and hasattr(
        torch.ops._xpu_C, "fp8_gemm_w8a16_pair"
    )


def _is_per_tensor_fp8_gemm(node: fx.Node) -> bool:
    if not is_func(node, torch.ops._xpu_C.fp8_gemm_w8a16.default):
        return False
    args = list(node.args) + [None] * (4 - len(node.args))
    _, weight, scale, bias = args[:4]
    if bias is not None or not isinstance(scale, fx.Node):
        return False
    w, s = weight.meta.get("val"), scale.meta.get("val")
    return (
        isinstance(w, torch.Tensor)
        and w.dtype == torch.float8_e4m3fn
        and isinstance(s, torch.Tensor)
        and s.numel() == 1
    )


class XpuFp8GemmPairFusionPass(VllmInductorPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.enabled = _pair_op_available()
        self.matched_count = 0

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.enabled and compile_range.end <= MAX_TOKEN_NUM

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        by_input: dict[fx.Node, list[fx.Node]] = {}
        for node in graph.nodes:
            if _is_per_tensor_fp8_gemm(node) and isinstance(node.args[0], fx.Node):
                by_input.setdefault(node.args[0], []).append(node)
        for a, gemms in by_input.items():
            if len(gemms) != 2:
                continue
            g1, g2 = gemms
            # Insert after both weights/scales are available, before the
            # first use of either result (graph order is topological).
            with graph.inserting_before(g1):
                pair = graph.call_function(
                    torch.ops._xpu_C.fp8_gemm_w8a16_pair.default,
                    (a, g1.args[1], g1.args[2], g2.args[1], g2.args[2]),
                )
                out1 = graph.call_function(operator.getitem, (pair, 0))
                out2 = graph.call_function(operator.getitem, (pair, 1))
            if any(
                isinstance(arg, fx.Node) and not _before(arg, g1)
                for arg in (g2.args[1], g2.args[2])
            ):
                # g2's weight is defined after g1; keep the original pair.
                graph.erase_node(out2)
                graph.erase_node(out1)
                graph.erase_node(pair)
                continue
            pair.meta["val"] = (g1.meta["val"], g2.meta["val"])
            out1.meta["val"], out2.meta["val"] = g1.meta["val"], g2.meta["val"]
            g1.replace_all_uses_with(out1)
            g2.replace_all_uses_with(out2)
            graph.erase_node(g1)
            graph.erase_node(g2)
            self.matched_count += 1
        logger.debug("XpuFp8GemmPairFusionPass fused %d pairs", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, _is_per_tensor_fp8_gemm)


def _before(a: fx.Node, b: fx.Node) -> bool:
    """Whether node a precedes node b in the graph."""
    node = a
    while node is not None and node.op != "root":
        if node is b:
            return True
        node = node.next
    return False
