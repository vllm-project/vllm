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

from vllm.config import VllmConfig
from vllm.ir.op import IrOp
from vllm.logger import init_logger
from vllm.logging_utils import lazy

from ..vllm_inductor_pass import VllmInductorPass
from .utils import get_ir_op

logger = init_logger(__name__)


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
        ir_op = get_ir_op(node)  # TODO is node.target always an overload?
        assert ir_op is not None, "Expected vLLM IR op"
        assert not node.kwargs  # I think there should never be kwargs here

        # Select and record the implementation, using fake args
        fake_args = fx.map_arg(node.args, lambda arg: arg.meta["val"])
        ir_op_impl = ir_op.dispatch(*fake_args)
        self.selected_impls[ir_op.name][node.name] = ir_op_impl.provider

        # replace_by_example wants node args, not the fake tensors
        # use func_impl_fn to properly handle in-place implementations
        # TODO(luka): Use aot_export_module to get functionalized graph
        # TODO(luka): Cache the fx_replacement to avoid re-tracing the same impl

        # Defaults not present on node.args but required for replacement tracing
        bound_args = ir_op._py_signature.bind(*node.args)
        bound_args.apply_defaults()
        match.replace_by_example(
            ir_op_impl.func_impl_fn, bound_args.args, run_functional_passes=False
        )

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        # clear at the beginning instead of end, so that tests can inspect
        self.selected_impls.clear()

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
            logger.warning("Full node list: %s", ",".join(failed_nodes))

    def uuid(self) -> str:
        """
        IR op priority & impl sources affect lowering pass output,
        so we include them in the cache key.
        """
        priorities = {name: op.get_priority() for name, op in IrOp.registry.items()}
        priorities_str = ";".join(
            f"{name}={','.join(p)}" for name, p in priorities.items()
        )

        impl_uuids_str = ";".join(
            f"{name}={
                ','.join(IrOp.registry[name].impls[provider].uuid() for provider in p)
            }"
            for name, p in priorities.items()
        )

        return f"{super().uuid()}|{priorities_str}|{impl_uuids_str}"
