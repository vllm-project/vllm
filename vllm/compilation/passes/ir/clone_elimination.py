# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._ops import OpOverload

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..fx_utils import is_func
from ..inductor_pass import get_pass_context
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


def user_writes_to_node(user: fx.Node, node: fx.Node) -> bool:
    if user.op == "output":
        return False

    if is_func(user, auto_functionalized):
        # While autofunc writes to the node,
        # this is a follow-up use we're not interested in.
        # It is also guaranteed to be the final use,
        # as auto_functionalized returns the tensor back for follow-up use.
        return False

    assert isinstance(user.target, OpOverload), (
        f"{node=} {user=} {user.op=} {user.target=}"
    )
    schema = user.target._schema
    assert len(user.args) <= len(schema.arguments)
    for i, arg in enumerate(user.args):
        # Only interested in writes to node
        if arg is not node:
            continue

        # If not a write, next arg could be
        if not schema.arguments[i].is_write:
            continue

        # Short-circuit, we know it's a write
        return True

    # No writes found
    return False


class UnsafeCloneEliminationPass(VllmInductorPass):
    """
    This pass removes clone nodes that are no longer needed after vLLM IR lowering.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        count = 0
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
        pass_context = get_pass_context()
        donated_input_ids = pass_context.donated_input_ids
        logger.debug("Donated input ids: %s", donated_input_ids)

        for node in graph.nodes:
            if not is_func(node, torch.ops.aten.clone.default):
                continue

            original_node = node.args[0]
            assert isinstance(original_node, fx.Node)

            # Clone needs to be preserved if node is getting written to and
            # the old value is used again.
            # This could only happen if an inplace implementation was lowered.
            # Then node (the clone) will have one write.
            # TODO(luka) hopefully this can be removed once we lower functional graphs.
            write_idxs = [
                node_to_idx[u] for u in node.users if user_writes_to_node(u, node)
            ]
            assert len(write_idxs) in (0, 1)
            if write_idxs:
                # Check if a user of original_node occurs after a write
                write_idx = write_idxs[0]
                if any(
                    node_to_idx[orig_user] > write_idx
                    for orig_user in original_node.users
                ):
                    logger.debug(
                        "Clone removal not possible, "
                        "original_node=%s used after mutation on node=%s",
                        original_node,
                        node,
                    )
                    continue

                # Check if a node is a (non-donated) graph input
                if (
                    original_node.op == "placeholder"
                    and node_to_idx[original_node] not in donated_input_ids
                ):
                    logger.debug(
                        "Graph input %s not donated, cannot eliminate its clone",
                        original_node,
                    )
                    continue

            logger.debug(
                "Node %s is a redundant clone node of %s, removing it",
                node,
                original_node,
            )
            node.replace_all_uses_with(original_node)
            graph.erase_node(node)
            count += 1

        logger.debug("CloneCleanupPass removed %d clone nodes", count)
