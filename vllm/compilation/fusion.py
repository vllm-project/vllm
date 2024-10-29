import operator
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch._inductor.pattern_matcher as pm
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import CompilationConfig
from vllm.logger import init_logger

from .vllm_inductor_pass import VllmInductorPass, is_func

logger = init_logger(__name__)


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp8(*args, **kwargs):
    fp8 = torch.float8_e4m3fn
    return torch.empty(*args, **kwargs, dtype=fp8, device="cuda")


def empty_fp32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float32, device="cuda")


# Utilities for post-processing multi-output matches


# Returns the first auto_functionalized node with the given op (if it exists)
def find_auto_fn_maybe(nodes: Iterable[torch.fx.Node],
                       op) -> Optional[torch.fx.Node]:
    for node in nodes:
        if is_func(node, auto_functionalized) and node.args[0] == op:  # noqa
            return node
    return None


# Returns the first auto_functionalized node with the given op
def find_auto_fn(nodes: Iterable[torch.fx.Node], op) -> torch.fx.Node:
    node = find_auto_fn_maybe(nodes, op)
    assert node is not None, f"Could not find {op} in nodes {nodes}"
    return node


# Returns the getitem node that extracts the idx-th element from node
# (if it exists)
def find_getitem_maybe(node: torch.fx.Node,
                       idx: int) -> Optional[torch.fx.Node]:
    for user in node.users:
        if is_func(user, operator.getitem) and user.args[1] == idx:
            return user
    return None


# Returns the getitem node that extracts the idx-th element from node
def find_getitem(node: torch.fx.Node, idx: int) -> torch.fx.Node:
    ret = find_getitem_maybe(node, idx)
    assert ret is not None, f"Could not find getitem {idx} in node {node}"
    return ret


class MultiOutputMatch:
    pass


class RMSNormQuantPattern:

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def register(self, pm_pass: PatternMatcherPass):
        # Cannot use methods, as the self argument affects tracing
        def pattern(result: torch.Tensor, result_rms: torch.Tensor,
                    input: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at1 = auto_functionalized(torch.ops._C.rms_norm.default,
                                      result=result_rms,
                                      input=input,
                                      weight=weight,
                                      epsilon=self.epsilon)
            at2 = auto_functionalized(
                torch.ops._C.static_scaled_fp8_quant.default,
                result=result,
                input=at1[1],
                scale=scale)

            # result
            return at2[1]

        def replacement(result: torch.Tensor, result_rms: torch.Tensor,
                        input: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(
                torch.ops._C.rms_norm_static_fp8_quant.default,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon)

            # result
            return at[1]

        inputs = [
            empty_fp8(5, 4),  # result
            empty_bf16(5, 4),  # result_rms
            empty_bf16(5, 4),  # input
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1)  # scale
        ]

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only,
                                pm_pass)


class FusedAddRMSNormQuantPattern:

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def register(self, pm_pass: PatternMatcherPass,
                 record_match: Callable[[MultiOutputMatch], bool]):

        def pattern(result: torch.Tensor, input: torch.Tensor,
                    residual: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at = auto_functionalized(torch.ops._C.fused_add_rms_norm.default,
                                     input=input,
                                     residual=residual,
                                     weight=weight,
                                     epsilon=self.epsilon)
            at1 = auto_functionalized(
                torch.ops._C.static_scaled_fp8_quant.default,
                result=result,
                input=at[1],
                scale=scale)

            # result, residual
            return at1[1], at[2]

        def replacement(result: torch.Tensor, input: torch.Tensor,
                        residual: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(
                torch.ops._C.fused_add_rms_norm_static_fp8_quant.default,
                result=result,
                input=input,
                residual=residual,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon)

            # result, residual
            return at[1], at[2]

        inputs = [
            empty_fp8(5, 4),  # result
            empty_bf16(5, 4),  # input
            empty_bf16(5, 4),  # residual
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1)  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=lambda m: record_match(self.Match(m, self)))

    class Match(MultiOutputMatch):

        def __init__(self, match: pm.Match,
                     pattern: 'FusedAddRMSNormQuantPattern'):
            self.match = match
            self.pattern = pattern

        def process(self, graph: torch.fx.Graph):
            # Find the nodes in the match that we need to rebind
            rms_node = find_auto_fn(self.match.nodes,
                                    torch.ops._C.fused_add_rms_norm.default)
            quant_node = find_auto_fn(
                self.match.nodes, torch.ops._C.static_scaled_fp8_quant.default)

            assert len(rms_node.users) == 2
            assert len(quant_node.users) == 1

            # First, insert a new auto_functionalized node for the fused op,
            # as well as getitem nodes to extract the result and residual.
            # The auto_functionalized node returns a tuple of
            # (None, result, residual) - None is the function return value.
            # The resulting graph looks like this:
            # at = auto_functionalized(torch.ops._C.fused_add_rms_norm_static_fp8_quant.default, ...)  # noqa
            # result_node_new = at[1]
            # residual_node_new = at[2]
            with self.inserting_after_match(graph):
                kwargs = self.match.kwargs.copy()

                # Scalars cannot be inputs to the pattern
                kwargs["epsilon"] = rms_node.kwargs["epsilon"]

                fused_node = graph.call_function(
                    auto_functionalized,
                    (torch.ops._C.fused_add_rms_norm_static_fp8_quant.default,
                     ),
                    kwargs=kwargs)

                getitem_nodes = self.insert_getitems(graph, fused_node, (1, 2))
                result_node_new, residual_node_new = getitem_nodes

            # Next, rebind the users of nodes in the match to use the new nodes.
            # Find the getitem nodes and replace their uses with the new nodes.
            # The old nodes will be removed by DCE at the end of the pass.
            find_getitem(rms_node, 2).replace_all_uses_with(residual_node_new)
            find_getitem(quant_node, 1).replace_all_uses_with(result_node_new)

            # Finally, fix meta["val"] for de-functionalization
            rms_tup, quant_tup = rms_node.meta["val"], quant_node.meta["val"]
            # Result of fused node is (None, result, residual)
            fused_node.meta["val"] = (None, quant_tup[1], rms_tup[2])

        def inserting_after_match(self, graph: torch.fx.Graph):
            """
            TODO comment

            :param graph:
            :return:
            """
            # match.nodes is not guaranteed to be sorted.
            # Find the last node in the match.
            for last_node_in_match in reversed(graph.nodes):
                if last_node_in_match in self.match.nodes:
                    break
            else:
                raise ValueError("No nodes in graph")

            return graph.inserting_after(last_node_in_match)

        def insert_getitems(self, graph: torch.fx.Graph,
                            tuple_node: torch.fx.Node, indices: Tuple[int,
                                                                      ...]):
            """
            TODO comment

            :param graph:
            :param tuple_node:
            :param indices:
            :return:
            """
            with graph.inserting_after(tuple_node):
                return [
                    graph.call_function(operator.getitem, (tuple_node, idx))
                    for idx in indices
                ]


class FusionPass(VllmInductorPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.
    It also manually processes multi-output matches, as those are broken in
    the torch pattern matcher.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    _instance: 'Optional[FusionPass]' = None

    @classmethod
    def instance(cls, config: CompilationConfig.PassConfig):
        """
        Get the singleton instance of the FusionPass.
        If the instance exists, the config is updated but
        initialization is not repeated.
        """
        if cls._instance is None:
            cls._instance = FusionPass(config)
        else:
            cls._instance.config = config
        return cls._instance

    def __init__(self, config: CompilationConfig.PassConfig):
        assert self.__class__._instance is None, \
            "FusionPass singleton instance already exists"
        super().__init__(config)

        self.matches: List[MultiOutputMatch] = []
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="fusion_pass")

        for epsilon in [1e-5]:  # TODO figure out how to do multiple epsilons
            # Fuse rms_norm + static_scaled_fp8_quant into
            # rms_norm_static_fp8_quant
            RMSNormQuantPattern(epsilon).register(self.patterns)

            # Fuse fused_add_rms_norm + static_scaled_fp8_quant into
            # fused_add_rms_norm_static_fp8_quant
            # Because pattern has 2 outputs, we need to manually process
            # the match (see process_matches)
            FusedAddRMSNormQuantPattern(epsilon).register(
                self.patterns, self.record_match)

    def record_match(self, match: MultiOutputMatch) -> bool:
        # Hijack the extra_check to record the match and
        # save it for post-processing.
        self.matches.append(match)

        # Return False to prevent automatic replacement.
        return False

    def process_matches(self, graph: torch.fx.Graph):
        """
        Manually process multi-output matches and replace them with fused nodes.
        This is necessary because the automatic replacement for multi-output
        matches is broken: https://github.com/pytorch/pytorch/issues/137280
        """
        for match in self.matches:
            match.process(graph)

        # Finally, remove matched nodes
        graph.eliminate_dead_code()
        assert all(node not in graph.nodes for match in self.matches
                   for node in match.match.nodes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_fusion")

        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_pattern_match")

        # Manually process multi-output matches (and run DCE)
        self.process_matches(graph)
        logger.debug("Post-processed %s matches", len(self.matches))
        self.dump_graph(graph, "after_fusion")
        self.matches.clear()
        self.end_and_log()
