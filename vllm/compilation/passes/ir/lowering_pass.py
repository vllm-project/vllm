# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable

from torch import fx
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._ops import OpOverload, OpOverloadPacket

from vllm.config import VllmConfig
from vllm.ir.op import IrOp
from vllm.logger import init_logger
from vllm.logging_utils import lazy

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


def get_default_overload(op: OpOverload | OpOverloadPacket) -> OpOverload:
    if isinstance(op, OpOverloadPacket):
        return op.default
    assert isinstance(op, OpOverload), "Expected an OpOverload or OpOverloadPacket"
    return op


def get_ir_op(node: fx.Node) -> IrOp | None:
    if node.op != "call_function":
        return None

    if not isinstance(node.target, (OpOverload, OpOverloadPacket)):
        return None

    op_overload = get_default_overload(node.target)
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


class VllmIRLoweringPass(VllmInductorPass):
    """
    This pass lowers vLLM IR ops to their implementations the priority list.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)
        self.patterns = PatternMatcherPass(self.pass_name)
        self.selected_impls: dict[str, dict[str, str]] = defaultdict(lambda: {})
        self.ops = [ir_op.torch_op for ir_op in IrOp.registry.values()]

        # Look for any call_function node where the target is a vLLM IR op.
        # Then, lower_matched_op will select, trace, and insert the implementation.
        register_graph_pattern(
            CallFunctionVarArgs(self.ops),
            pass_dict=self.patterns,
        )(self.lower_matched_op)

    def lower_matched_op(self, match: Match, *args, **kwargs):
        # TODO(luka) I think args and kwargs are for the match, but just use the node?

        assert len(match.nodes) == 1, "Expected single node match"
        node = match.nodes[0]
        ir_op = get_ir_op(node)
        assert ir_op is not None, "Expected vLLM IR op"

        bound_args, bound_kwargs = ir_op.apply_arg_defaults(*node.args, **node.kwargs)
        assert not bound_kwargs  # I think there should never be kwargs here

        # Select and record the implementation, using fake args
        fake_args = fx.map_arg(bound_args, lambda arg: arg.meta["val"])
        ir_op_impl = ir_op.dispatch(*fake_args)
        self.selected_impls[ir_op.name][node.name] = ir_op_impl.provider

        # replace_by_example wants node args, not the fake tensors
        # TODO(luka): Use aot_export_module to get functionalized graph
        # TODO(luka): Cache the fx_replacement to avoid re-tracing the same impl
        match.replace_by_example(ir_op_impl.impl_fn, bound_args)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        count = self.patterns.apply(graph)
        logger.debug("VllmIRLoweringPass lowered %d vLLM IR nodes", count)

        # TODO write self.selected_impls to depyf/tlparse dir
        def count_items(impls: Iterable[str]) -> dict[str, int]:
            counts: dict[str, int] = defaultdict(lambda: 0)
            for impl in impls:
                counts[impl] += 1
            return counts

        def print_count(counts: dict[str, int]) -> str:
            # e.g., "impl1*3,impl2"
            impl_count = lambda i, c: f"{i}" if c == 1 else f"{i}*{c}"
            return ",".join(impl_count(impl, count) for impl, count in counts.items())

        logger.debug(
            "Selected implementations: %s",
            lazy(
                lambda: ", ".join(
                    f"{op}={print_count(count_items(impls_by_node.values()))}"
                    for op, impls_by_node in self.selected_impls.items()
                )
            ),
        )

        failed_nodes: list[fx.Node] = []
        failed_ops: set[str] = set()
        # Check no vllm_ir nodes were left in the graph
        for node in graph.nodes:
            if (ir_op := get_ir_op(node)) is None:
                continue

            failed_nodes.append(node)
            failed_ops.add(ir_op.name)

        if failed_nodes or failed_ops:
            logger.warning("Failed to lower vLLM IR ops: %s", ",".join(failed_ops))
            logger.warning("Full node list: %s", failed_nodes)

        self.selected_impls.clear()
