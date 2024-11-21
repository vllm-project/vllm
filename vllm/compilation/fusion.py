import abc
import operator
from abc import abstractmethod
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


class MultiOutputMatch(abc.ABC):
    """
    This class provides utilities to process multi-output matches and
    manually insert replacements.

    This is necessary because the automatic replacement for multi-output
    matches is broken: https://github.com/pytorch/pytorch/issues/137280
    """

    def __init__(self, match: pm.Match):
        self.match = match

    @abstractmethod
    def process(self):
        """
        Process a multi-output match and manually insert the replacement.

        This method should:
        1. Insert the replacement nodes after the last node in the match.
        2. Rebind the users of nodes in the match to use the new nodes.
        3. Set meta["val"] for de-functionalization.

        The result of an auto-functionalized node is a tuple of tensors.
        The first element is the return value of the function, usually None.
        The remaining elements are the mutated args of the function.

        All auto-functionalized nodes must contain a proper meta["val"],
        as it is used by de-functionalization. meta["val"] has to contain the
        value of the node (tuple of tensors) that would be returned by the
        functionalized node during tracing.

        Existing nodes in the graph all have this property set, but we have
        to set it manually for new nodes we insert.

        Example:
        # op schema: foo(a: Tensor!, b: Tensor, c: Tensor!) -> None
        at = auto_functionalized(torch.ops._C.foo.default, a, b, c)
        # at.meta["val"] = (None, a, c)
        """
        raise NotImplementedError

    @property
    def nodes(self) -> List[torch.fx.Node]:
        return self.match.nodes

    @property
    def graph(self) -> torch.fx.Graph:
        return self.match.graph

    def find_auto_fn(self, op) -> torch.fx.Node:
        """
        Find the first auto_functionalized node with the given op in the match.
        """
        return find_auto_fn(self.nodes, op)

    def inserting_after_match(self):
        """
        Insert nodes after the last node in the match.
        This is done to avoid use-before-definition errors after inserting
        replacement nodes.
        """

        # match.nodes is not guaranteed to be sorted.
        # Find the last node in the match.
        for last_node_in_match in reversed(self.graph.nodes):
            if last_node_in_match in self.match.nodes:
                break
        else:
            raise ValueError("No nodes in graph")

        return self.graph.inserting_after(last_node_in_match)

    def insert_getitems(self, tuple_node: torch.fx.Node,
                        indices: Tuple[int, ...]) -> Tuple[torch.fx.Node, ...]:
        """
        Insert operator.getitem nodes to extract elements from a tuple node.

        :param tuple_node: The tuple node to extract elements from.
        :param indices: The indices of the elements to extract.
        :return: Tuple of the new getitem nodes, corresponding to the indices.
        """
        with self.graph.inserting_after(tuple_node):
            return tuple(
                self.graph.call_function(operator.getitem, (tuple_node, idx))
                for idx in indices)

    def insert_auto_fn(self, op, kwargs):
        """
        Insert an auto_functionalized node with the given op and kwargs.
        """
        return self.graph.call_function(auto_functionalized, (op, ),
                                        kwargs=kwargs)


RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_STATIC_FP8_OP = torch.ops._C.static_scaled_fp8_quant.default
QUANT_DYNAMIC_FP8_OP = torch.ops._C.dynamic_scaled_fp8_quant.default


class RMSNormQuantPattern:

    def __init__(self, epsilon: float):
        self.epsilon = epsilon


class RMSNormStaticFP8QuantPattern(RMSNormQuantPattern):

    def register(self, pm_pass: PatternMatcherPass):
        # Cannot use methods, as the self argument affects tracing
        def pattern(result: torch.Tensor, result_rms: torch.Tensor,
                    input: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at1 = auto_functionalized(RMS_OP,
                                      result=result_rms,
                                      input=input,
                                      weight=weight,
                                      epsilon=self.epsilon)
            at2 = auto_functionalized(QUANT_STATIC_FP8_OP,
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


class FusedAddRMSNormStaticFP8QuantPattern(RMSNormQuantPattern):

    def register(self, pm_pass: PatternMatcherPass,
                 record_match: Callable[[MultiOutputMatch], bool]):

        def pattern(result: torch.Tensor, input: torch.Tensor,
                    residual: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at = auto_functionalized(RMS_ADD_OP,
                                     input=input,
                                     residual=residual,
                                     weight=weight,
                                     epsilon=self.epsilon)
            at1 = auto_functionalized(QUANT_STATIC_FP8_OP,
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
            extra_check=lambda m: record_match(self.Match(m)))

    class Match(MultiOutputMatch):

        def process(self):
            # Find the nodes in the match that we need to rebind
            rms_node = self.find_auto_fn(RMS_ADD_OP)
            quant_node = self.find_auto_fn(QUANT_STATIC_FP8_OP)

            assert len(rms_node.users) == 2
            assert len(quant_node.users) == 1

            # First, insert a new auto_functionalized node for the fused op,
            # as well as getitem nodes to extract the result and residual.
            # The auto_fn node returns a tuple of (None, result, residual).
            #
            # The resulting graph looks like this:
            # at = auto_functionalized(torch.ops._C.fused_add_rms_norm_static_fp8_quant.default, ...)  # noqa
            # result_node_new = at[1]
            # residual_node_new = at[2]
            with self.inserting_after_match():
                kwargs = self.match.kwargs.copy()

                # Scalars cannot be inputs to the pattern
                kwargs["epsilon"] = rms_node.kwargs["epsilon"]

                fused_node = self.insert_auto_fn(
                    torch.ops._C.fused_add_rms_norm_static_fp8_quant.default,
                    kwargs)

                getitem_nodes = self.insert_getitems(fused_node, (1, 2))
                result_node_new, residual_node_new = getitem_nodes

            # Rebind the users of match getitem nodes to use the new nodes.
            # The old nodes will be removed by DCE at the end of the pass.
            find_getitem(rms_node, 2).replace_all_uses_with(residual_node_new)
            find_getitem(quant_node, 1).replace_all_uses_with(result_node_new)

            # Finally, fix meta["val"] for de-functionalization.
            # See MultiOutputMatch.process for more details.
            rms_tup, quant_tup = rms_node.meta["val"], quant_node.meta["val"]
            # Result of fused node is (None, result, residual)
            fused_node.meta["val"] = (None, quant_tup[1], rms_tup[2])


class RMSNormDynamicFP8QuantPattern(RMSNormQuantPattern):

    def register(self, pm_pass: PatternMatcherPass,
                 record_match: Callable[[MultiOutputMatch], bool]):

        def pattern(result: torch.Tensor, result_rms: torch.Tensor,
                    input: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at1 = auto_functionalized(RMS_OP,
                                      result=result_rms,
                                      input=input,
                                      weight=weight,
                                      epsilon=self.epsilon)
            at2 = auto_functionalized(QUANT_DYNAMIC_FP8_OP,
                                      result=result,
                                      input=at1[1],
                                      scale=scale)

            # result, scale
            return at2[1], at2[2]

        def replacement(result: torch.Tensor, result_rms: torch.Tensor,
                        input: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(
                torch.ops._C.rms_norm_dynamic_per_token_quant.default,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=None)

            # result, scale
            return at[1], at[2]

        inputs = [
            empty_fp8(5, 4),  # result
            empty_bf16(5, 4),  # result_rms
            empty_bf16(5, 4),  # input
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1)  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=lambda m: record_match(self.Match(m)))

    class Match(MultiOutputMatch):

        def process(self):
            # Find the nodes in the match that we need to rebind
            rms_node = self.find_auto_fn(RMS_OP)
            quant_node = self.find_auto_fn(QUANT_DYNAMIC_FP8_OP)

            assert len(rms_node.users) == 1
            assert len(quant_node.users) == 2

            # First, insert a new auto_functionalized node for the fused op,
            # as well as getitem nodes to extract the result and scale.
            # The auto_fn node returns a tuple of (None, result, scale).
            #
            # The resulting graph looks like this:
            # at = auto_functionalized(torch.ops._C.rms_norm_dynamic_per_token_quant.default, ...)  # noqa
            # result_node_new = at[1]
            # scale_node_new = at[2]
            with self.inserting_after_match():
                kwargs = self.match.kwargs.copy()

                # Scalars cannot be inputs to the pattern
                kwargs["epsilon"] = rms_node.kwargs["epsilon"]
                kwargs["scale_ub"] = None  # not used but required
                kwargs["residual"] = None  # not used but required
                del kwargs["result_rms"]  # not used in the fused op

                fused_node = self.insert_auto_fn(
                    torch.ops._C.rms_norm_dynamic_per_token_quant.default,
                    kwargs=kwargs)

                getitem_nodes = self.insert_getitems(fused_node, (1, 2))
                result_node_new, scale_node_new = getitem_nodes

            # Rebind the users of match getitem nodes to use the new nodes.
            # The old nodes will be removed by DCE at the end of the pass.
            find_getitem(quant_node, 1).replace_all_uses_with(result_node_new)
            find_getitem(quant_node, 2).replace_all_uses_with(scale_node_new)

            # Finally, fix meta["val"] for de-functionalization.
            # See MultiOutputMatch.process for more details.
            # Result of fused node is (None, result, scale)
            fused_node.meta["val"] = quant_node.meta["val"]


class FusedAddRMSNormDynamicFP8QuantPattern(RMSNormQuantPattern):

    def register(self, pm_pass: PatternMatcherPass,
                 record_match: Callable[[MultiOutputMatch], bool]):

        def pattern(result: torch.Tensor, input: torch.Tensor,
                    residual: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor):
            at = auto_functionalized(RMS_ADD_OP,
                                     input=input,
                                     residual=residual,
                                     weight=weight,
                                     epsilon=self.epsilon)
            at1 = auto_functionalized(QUANT_DYNAMIC_FP8_OP,
                                      result=result,
                                      input=at[1],
                                      scale=scale)

            # result, residual, scale
            return at1[1], at[2], at1[2]

        def replacement(result: torch.Tensor, input: torch.Tensor,
                        residual: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            at = auto_functionalized(
                torch.ops._C.rms_norm_dynamic_per_token_quant.default,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=residual)

            # result, residual, scale
            return at[1], at[3], at[2]

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
            extra_check=lambda m: record_match(self.Match(m)))

    class Match(MultiOutputMatch):

        def process(self):
            # Find the nodes in the match that we need to rebind
            rms_node = self.find_auto_fn(RMS_ADD_OP)
            quant_node = self.find_auto_fn(QUANT_DYNAMIC_FP8_OP)

            assert len(rms_node.users) == 2
            assert len(quant_node.users) == 2

            # First, insert a new auto_functionalized node for the fused op,
            # as well as getitem nodes to extract result, scale, and residual.
            # The auto_fn node returns a tuple (None, result, scale, residual).
            #
            # The resulting graph looks like this:
            # at = auto_functionalized(torch.ops._C.rms_norm_dynamic_per_token_quant.default, ...)  # noqa
            # result_node_new = at[1]
            # scale_node_new = at[2]
            # residual_node_new = at[3]
            with self.inserting_after_match():
                kwargs = self.match.kwargs.copy()

                # Scalars cannot be inputs to the pattern
                kwargs["epsilon"] = rms_node.kwargs["epsilon"]
                kwargs["scale_ub"] = None  # not used but required

                fused_node = self.insert_auto_fn(
                    torch.ops._C.rms_norm_dynamic_per_token_quant.default,
                    kwargs=kwargs)

                getitem_ns = self.insert_getitems(fused_node, (1, 2, 3))
                result_node_new, scale_node_new, residual_node_new = getitem_ns

            # Rebind the users of match getitem nodes to use the new nodes.
            # The old nodes will be removed by DCE at the end of the pass.
            find_getitem(rms_node, 2).replace_all_uses_with(residual_node_new)
            find_getitem(quant_node, 1).replace_all_uses_with(result_node_new)
            find_getitem(quant_node, 2).replace_all_uses_with(scale_node_new)

            # Finally, fix meta["val"] for de-functionalization.
            # See MultiOutputMatch.process for more details.
            rms_tup, quant_tup = rms_node.meta["val"], quant_node.meta["val"]
            # Result of fused node is (None, result, scale, residual)
            fused_node.meta["val"] = (None, quant_tup[1], quant_tup[2],
                                      rms_tup[2])


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

        for epsilon in [1e-5, 1e-6]:
            # Fuse rms_norm + static_scaled_fp8_quant into
            # rms_norm_static_fp8_quant
            RMSNormStaticFP8QuantPattern(epsilon).register(self.patterns)

            # Fuse fused_add_rms_norm + static_scaled_fp8_quant into
            # fused_add_rms_norm_static_fp8_quant
            # Because pattern has 2 outputs, we need to manually process
            # the match (see process_matches)
            FusedAddRMSNormStaticFP8QuantPattern(epsilon).register(
                self.patterns, self.record_match)

            # Fuse rms_norm + dynamic_scaled_fp8_quant into
            # rms_norm_dynamic_per_token_quant
            RMSNormDynamicFP8QuantPattern(epsilon).register(
                self.patterns, self.record_match)

            # Fuse fused_add_rms_norm + dynamic_scaled_fp8_quant into
            # rms_norm_dynamic_per_token_quant
            FusedAddRMSNormDynamicFP8QuantPattern(epsilon).register(
                self.patterns, self.record_match)

            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

    def record_match(self, match: MultiOutputMatch) -> bool:
        # Hijack the extra_check to record the match and
        # save it for post-processing.
        self.matches.append(match)

        # Return False to prevent automatic replacement.
        return False

    def process_matches(self, graph: torch.fx.Graph):
        """
        Manually process multi-output matches and replace them with fused nodes.
        See MultiOutputMatch for more details.
        """
        for match in self.matches:
            match.process()

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
