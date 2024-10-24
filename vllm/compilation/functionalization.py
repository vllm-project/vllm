import operator
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.compilation.inductor_pass import InductorPass, is_func


class FixFunctionalizationPass(InductorPass):
    """
    This pass de-functionalizes certain nodes to avoid redundant copies.
    After this pass, DCE (dead-code elimination) should never be run,
    as de-functionalized nodes may appear as dead code.
    """

    def __call__(self, graph: torch.fx.Graph):
        self.dump_graph(graph, "before_fix_functionalization")

        self.nodes_to_remove: List[torch.fx.Node] = []
        for node in graph.nodes:
            if not is_func(node, auto_functionalized):
                continue  # Avoid deep if-elif nesting

            kwargs = node.kwargs
            at_target = node.args[0]

            if at_target == torch.ops._C.rotary_embedding.default:
                self.insert_defunctionalized(graph, node)

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
                self._remove(node)

            elif at_target in [
                    torch.ops._C.fused_add_rms_norm.default,
                    torch.ops._C.fused_add_rms_norm_static_fp8_quant.default
            ]:
                # This replacement avoids the most copies for LLaMa.
                self.insert_defunctionalized(graph, node)

                # Fused quantized uses separate result instead of input
                quantized = at_target != torch.ops._C.fused_add_rms_norm.default
                out_key = 'result' if quantized else 'input'
                out, residual = kwargs[out_key], kwargs['residual']

                getitem_users = self.getitem_users(node)
                getitem_users[1].replace_all_uses_with(out)
                if 2 in getitem_users:
                    # residual not used for the last fused_add_rms_norm
                    getitem_users[2].replace_all_uses_with(residual)
                self._remove(getitem_users.values())
                self._remove(node)
            elif at_target in [
                    torch.ops._C.rms_norm.default,
                    torch.ops._C.rms_norm_static_fp8_quant.default
            ]:
                result = kwargs['result']
                self.insert_defunctionalized(graph, node)

                getitem_users = self.getitem_users(node)
                getitem_users[1].replace_all_uses_with(result)
                self._remove(getitem_users.values())
                self._remove(node)

            elif at_target == torch.ops._C.silu_and_mul.default:
                input, out = kwargs['input'], kwargs['out']
                # Because we have an 'out', cannot use kwargs.
                self.insert_defunctionalized(graph, node, args=(out, input))

                getitem_users = self.getitem_users(node)
                getitem_users[1].replace_all_uses_with(out)
                self._remove(getitem_users.values())
                self._remove(node)

        self.dump_graph(graph, "before_fix_functionalization_cleanup")

        # Remove the nodes all at once
        for node in self.nodes_to_remove:
            graph.erase_node(node)

        self.dump_graph(graph, "after_fix_functionalization")

    def _remove(self, node_or_nodes: Union[torch.fx.Node,
                                           Iterable[torch.fx.Node]]):
        if isinstance(node_or_nodes, torch.fx.Node):
            self.nodes_to_remove.append(node_or_nodes)
        else:
            self.nodes_to_remove.extend(node_or_nodes)

    def getitem_users(self, node: torch.fx.Node) -> Dict[int, torch.fx.Node]:
        """
        TODO should this be called getitem_nodes instead?
        Returns the users of the auto-functionalized node,
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
                                args: Optional[Tuple[torch.fx.Node,
                                                     ...]] = None):
        """
        Insert a new defunctionalized node into the graph before node.
        If one of the kwargs is 'out', use args instead.
        See https://github.com/pytorch/pytorch/blob/a00faf440888ffb724bad413f329a49e2b6388e7/torch/_inductor/lowering.py#L351

        :param graph: Graph to insert the defunctionalized node into
        :param node: The auto-functionalized node to defunctionalize
        :param args: If we cannot use kwargs, specify args directly
        """  # noqa: E501
        assert is_func(node, auto_functionalized), \
            f"node must be auto-functionalized, is {node} instead"

        # Create a new call to the original function
        with graph.inserting_before(node):
            function = node.args[0]
            if args is None:
                graph.call_function(function, kwargs=node.kwargs)
            else:
                graph.call_function(function, args=args)
