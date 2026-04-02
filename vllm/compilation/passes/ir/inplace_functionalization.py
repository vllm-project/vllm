# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict

from torch import fx
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..inductor_pass import get_pass_context
from ..vllm_inductor_pass import VllmInductorPass
from .lowering_pass import get_ir_op
from .utils import overload_or_default

logger = init_logger(__name__)


class VllmIRInplaceFunctionalizationPass(VllmInductorPass):
    """
    This pass functionalizes maybe_inplace vLLM IR ops to the default overload.
    The maybe_inplace overloads have the same signature as the default overload
    so the pass simply replaces the called overload.
    That makes the graph properly functional.

    This pass operates pre-AOTAutograd,
    so it must handle non-normalized and non-functional IR.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)
        self.patterns = PatternMatcherPass(self.pass_name)
        self.functionalized_ops: dict[str, int] = defaultdict(lambda: 0)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        # clear at the beginning instead of end, so that tests can inspect
        self.functionalized_ops.clear()
        assert graph.owning_module is not None
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}

        # Pass donated input via vLLM's pass context
        pass_context = get_pass_context()
        pass_context.donated_input_ids = set[int]()

        for node in graph.nodes:
            if (ir_op := get_ir_op(node)) is None:
                continue

            op_overload = overload_or_default(node.target)
            overload_name = op_overload._overloadname
            if overload_name != "maybe_inplace":
                assert overload_name == "default", (
                    f"Found overload {overload_name} for op {ir_op.name}, "
                    f"expected maybe_inplace or default"
                )
                continue

            # must have maybe_inplace overload and allow_inplace
            assert ir_op.allow_inplace and ir_op.maybe_inplace is not None

            # Check that activation inputs are not used after this op
            for arg_idx in ir_op.activation_indices:
                arg = node.args[arg_idx]
                assert isinstance(arg, fx.Node), "Activation inputs must be fx.Node"
                for user in arg.users:
                    if node_to_idx[user] > node_to_idx[node]:
                        raise ValueError(
                            f"Input {arg} to maybe_inplace node {node} "
                            f"is used again after the node. "
                            f"This is not allowed; activation inputs to maybe_inplace "
                            f"ops are donated to the op, meaning their memory may be "
                            f"recycled for outputs.\n\n"
                            f"To preserve the inputs, use the default overload or "
                            f"clone them manually beforehand."
                        )

                if arg.op == "placeholder":
                    # Graph input that maybe_inplace might modify.
                    # Mark it so downstream passes know it's donated.
                    # TODO(luka) store in placeholder node meta once supported
                    pass_context.donated_input_ids.add(node_to_idx[arg])

            # Same signature, just replace the overload that's called.
            node.target = ir_op.torch_op
            self.functionalized_ops[ir_op.name] += 1

        count = sum(self.functionalized_ops.values())
        ops = ",".join(self.functionalized_ops.keys())
        logger.debug("Donated input IDs: %s", pass_context.donated_input_ids)
        logger.debug(
            "%s functionalized %d vLLM IR nodes for op(s) %s",
            self.pass_name,
            count,
            ops,
        )
