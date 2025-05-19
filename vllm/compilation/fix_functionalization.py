# SPDX-License-Identifier: Apache-2.0

import operator
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.logger import init_logger

from .fx_utils import is_func
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class FixFunctionalizationPass(VllmInductorPass):
    """
    This pass defunctionalizes certain nodes to avoid redundant tensor copies.
    After this pass, DCE (dead-code elimination) should never be run,
    as de-functionalized nodes may appear as dead code.

    To add new nodes to defunctionalize, add to the if-elif chain in __call__.
    """

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_fix_functionalization")

        self.nodes_to_remove: list[torch.fx.Node] = []
        count = 0
        for node in graph.nodes:
            if not is_func(node, auto_functionalized):
                continue  # Avoid deep if-elif nesting

            kwargs = node.kwargs
            at_target = node.args[0]

            if at_target == torch.ops._C.rotary_embedding.default:
                query = kwargs['query']
                mm_node = query.args[0].args[0]

                # rotary_embedding is a special case: the two mutating inputs
                # are query and key, which are slices of mm_node.
                # While functionalized, results at[1] and at[2] are scattered
                # back into mm_node. After de-functionalization, we can just
                # use mm_node directly.
                for idx, user in self.getitem_users(node).items():
                    for user_of_getitem in user.users:
                        if is_func(user_of_getitem,
                                   torch.ops.aten.slice_scatter.default):
                            user_of_getitem.replace_all_uses_with(mm_node)
                            self._remove(user_of_getitem)
                    self._remove(user)

                self.insert_defunctionalized(graph, node)
                self._remove(node)

            # rms_norm replacements avoid the most copies for LLaMa.
            elif at_target == torch.ops._C.fused_add_rms_norm.default:
                mutated_args = {1: 'input', 2: 'residual'}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target == torch.ops._C.fused_add_rms_norm_static_fp8_quant.default:  # noqa: E501
                mutated_args = {1: 'result', 2: 'residual'}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target == torch.ops._C.rms_norm_dynamic_per_token_quant.default:  # noqa: E501
                mutated_args = {1: 'result', 2: 'scale', 3: 'residual'}
                self.defunctionalize(graph, node, mutated_args)
            elif at_target in [
                    torch.ops._C.rms_norm.default,
                    torch.ops._C.rms_norm_static_fp8_quant.default,
            ]:
                mutated_args = {1: 'result'}
                self.defunctionalize(graph, node, mutated_args)
            # For some reason we need to specify the args for both
            # silu_and_mul and silu_and_mul_quant. The kwargs
            # pathway gets the wrong answer.
            elif at_target == torch.ops._C.silu_and_mul.default:
                mutated_args = {1: 'result'}
                self.defunctionalize(graph,
                                     node,
                                     mutated_args,
                                     args=('result', 'input'))
            elif at_target == torch.ops._C.silu_and_mul_quant.default:
                mutated_args = {1: 'result'}
                self.defunctionalize(graph,
                                     node,
                                     mutated_args,
                                     args=('result', 'input', 'scale'))
            else:
                continue  # skip the count

            count += 1

        self.dump_graph(graph, "before_fix_functionalization_cleanup")

        # Remove the nodes all at once
        count_removed = len(self.nodes_to_remove)
        for node in self.nodes_to_remove:
            graph.erase_node(node)

        logger.debug("De-functionalized %s nodes, removed %s nodes", count,
                     count_removed)
        self.dump_graph(graph, "after_fix_functionalization")
        self.end_and_log()

    def _remove(self, node_or_nodes: Union[torch.fx.Node,
                                           Iterable[torch.fx.Node]]):
        """
        Stage a node (or nodes) for removal at the end of the pass.
        """
        if isinstance(node_or_nodes, torch.fx.Node):
            self.nodes_to_remove.append(node_or_nodes)
        else:
            self.nodes_to_remove.extend(node_or_nodes)

    def defunctionalize(self,
                        graph: torch.fx.Graph,
                        node: torch.fx.Node,
                        mutated_args: dict[int, Union[torch.fx.Node, str]],
                        args: Optional[tuple[Union[torch.fx.Node, str],
                                             ...]] = None):
        """
        De-functionalize a node by replacing it with a call to the original.
        It also replaces the getitem users with the mutated arguments.
        See replace_users_with_mutated_args and insert_defunctionalized.
        """
        self.replace_users_with_mutated_args(node, mutated_args)
        self.insert_defunctionalized(graph, node, args=args)
        self._remove(node)

    def replace_users_with_mutated_args(self, node: torch.fx.Node,
                                        mutated_args: dict[int,
                                                           Union[torch.fx.Node,
                                                                 str]]):
        """
        Replace all getitem users of the auto-functionalized node with the
        mutated arguments.
        :param node: The auto-functionalized node
        :param mutated_args: The mutated arguments, indexed by getitem index.
        If the value of an arg is a string, `node.kwargs[arg]` is used.
        """
        for idx, user in self.getitem_users(node).items():
            arg = mutated_args[idx]
            arg = node.kwargs[arg] if isinstance(arg, str) else arg
            user.replace_all_uses_with(arg)
            self._remove(user)

    def getitem_users(self, node: torch.fx.Node) -> dict[int, torch.fx.Node]:
        """
        Returns the operator.getitem users of the auto-functionalized node,
        indexed by the index they are getting.
        """
        users = {}
        for user in node.users:
            if is_func(user, operator.getitem):
                idx = user.args[1]
                users[idx] = user
        return users

    def insert_defunctionalized(self,
                                graph: torch.fx.Graph,
                                node: torch.fx.Node,
                                args: Optional[tuple[Union[torch.fx.Node, str],
                                                     ...]] = None):
        """
        Insert a new defunctionalized node into the graph before node.
        If one of the kwargs is 'out', provide args directly,
        as node.kwargs cannot be used.
        See https://github.com/pytorch/pytorch/blob/a00faf440888ffb724bad413f329a49e2b6388e7/torch/_inductor/lowering.py#L351

        :param graph: Graph to insert the defunctionalized node into
        :param node: The auto-functionalized node to defunctionalize
        :param args: If we cannot use kwargs, specify args directly.
        If an arg is a string, `node.kwargs[arg]` is used.
        """  # noqa: E501
        assert is_func(node, auto_functionalized), \
            f"node must be auto-functionalized, is {node} instead"

        # Create a new call to the original function
        with graph.inserting_before(node):
            function = node.args[0]
            if args is None:
                graph.call_function(function, kwargs=node.kwargs)
            else:
                # Args passed as strings refer to items in node.kwargs
                args = tuple(node.kwargs[arg] if isinstance(arg, str) else arg
                             for arg in args)
                graph.call_function(function, args=args)
