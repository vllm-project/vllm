# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Coalesce duplicate ``split_with_sizes`` nodes that operate on the same
input tensor with the same split sizes.

On certain hardware/dtype combinations (e.g. B200 + FP8) the Inductor
graph may contain multiple ``split_with_sizes`` calls on the same tensor
that CSE fails to merge. This pass detects and replaces the duplicates
so that downstream pattern-matching passes (e.g. QK-Norm+RoPE fusion)
see a single split node with all users attached.

See also:
  - vLLM  #33295  (original issue)
  - PyTorch #174472 (upstream CSE gap)
"""

import operator

import torch
from torch import fx

from vllm.logger import init_logger

from ..fx_utils import is_func
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class SplitCoalescingPass(VllmInductorPass):
    """Replace duplicate ``split_with_sizes`` nodes with a single canonical
    node when they share the same input tensor and split sizes."""

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        count = 0

        # Map from input tensor node -> list of split nodes seen so far.
        split_nodes: dict[fx.Node, list[fx.Node]] = {}

        for node in graph.nodes:
            if not is_func(node, torch.ops.aten.split_with_sizes.default):
                continue
            if not all(is_func(user, operator.getitem) for user in node.users):
                continue

            arg_node, split_sizes = node.args[:2]

            if arg_node not in split_nodes:
                split_nodes[arg_node] = [node]
                continue

            # Find existing node with same split_sizes
            canonical = next(
                (
                    n
                    for n in split_nodes[arg_node]
                    if list(n.args[1]) == list(split_sizes)
                ),
                None,
            )
            if canonical is not None:
                node.replace_all_uses_with(canonical)
                graph.erase_node(node)
                count += 1
            else:
                split_nodes[arg_node].append(node)

        logger.debug("Coalesced %d duplicate split_with_sizes nodes", count)
