###############################################################################
#
# Operator fusion pass
#
###############################################################################

import torch

from .code_cache import CodeCache
from .fused_op_generator import FusedOpGenerator, FusionFail
from .register import FUSABLE
from .utils import extract_node_type, ModuleInputGenerator, FlowGraph, node_function_target, graph_print_tabular, SubGraph

from torch.fx.passes.split_module import split_module
from torch.fx.passes.shape_prop import ShapeProp
from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set
from vllm.logger import init_logger

logger = init_logger(__name__)

"""
Fuse all the nodes in the given module into a single function call.
"""
def fuse_graph_nodes(
    cc: CodeCache,
    fgen: FusedOpGenerator,
    sub: SubGraph,
):
    outputs = sub.outputs
    inputs = sub.inputs

    if len(outputs) != 1:
        raise FusionFail("only single output supported currently.")

    # Collect all kwargs for fused ops and all the nodes that
    # will need to be fused (and erased) later.
    first = sub.first_in_graph()
    nodes_to_fuse = []
    kwargs = dict()
    for n in sub.nodes:
        if n.op != 'call_function':
            continue

        if n.kwargs is not None and len(n.kwargs) > 0:
            kwargs[n.name] = n.kwargs

        nodes_to_fuse.append(n)

    if kwargs is not None and len(kwargs) > 0:
        raise FusionFail(f"kwargs for fused ops not supported. {kwargs}")

    # Lookup or create the fused operation.
    try:
        fn_key = fgen.make_fused_op(inputs, outputs, nodes_to_fuse, kwargs)

        def generate() -> Optional[Callable]:
            fn_dict = fgen.build_ops()
            assert fn_key in fn_dict
            return fn_dict[fn_key]

        fn = cc.lookup_or_create(fn_key, generate)

    except FusionFail as ff:
        logger.info(f"fusion failed '{ff}' for subgraph.")
        return

    if fn is None:
        logger.info(f"fusion failed previously for subgraph.")
        return

    logger.debug(f"fused fn = {fn}")

    #
    # Update the graph
    # 1. insert the call_function for the fused op
    # 2. insert new output node(s)
    # 3. delete old call_function and output nodes.
    #

    sub.module.graph.inserting_after(first)

    # Note: we do not update the meta info for cf here.  It should
    # not be required after transformation anyway.
    cf = sub.module.graph.call_function(fn, args=tuple(inputs), kwargs=kwargs)
    logger.debug(f"fused op: {cf.format_node()}")

    # Note: assumes single output
    outputs[0].replace_all_uses_with(cf, propagate_meta=True)

    sub.erase()
    sub.build([cf])  # not necessary but nice for debugging


"""
Determine whether or not node is a fusable operations.
TODO: Smarter filter for 'getitem'.
"""
def is_fusable(node: torch.fx.Node) -> bool:
    if node.op != 'call_function':
        return False

    op_name = node_function_target(node)
    return op_name in FUSABLE and not FUSABLE[op_name]


"""
Determine whether or not node is a fusable compute operation, e.g. gemm.
"""
def is_compute(node: torch.fx.Node) -> bool:
    if node.op != 'call_function':
        return False

    op_name = node_function_target(node)
    return op_name in FUSABLE and FUSABLE[op_name]


def is_getitem(a: torch.fx.Node) -> bool:
    if a.op != 'call_function':
        return False
    return node_function_target(a) == '_operator.getitem'


"""
Are nodes a and b fusable together?
This function assumes 'b' is a direct successor of 'a'.
"""
def is_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    return is_fusable(a) and is_fusable(b)


"""
Are nodes 'a' and 'b' fusable together and is 'a' optionally a compute op?
This function assumes 'b' is a direct successor of 'a'.
"""
def is_compute_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    return (is_fusable(a) or is_compute(a)) and is_fusable(b)


"""
Determine if any kwargs associated with 'node' are supported.
"""
def supported_kwargs(node: torch.fx.Node, allow_const_kwargs: bool = False) -> bool:
    if allow_const_kwargs:
        for arg in node.kwargs.values():
            if not isinstance(arg, torch.fx.node.BaseArgumentTypes):
                return False
        return True
    else:
        return node.kwargs is None or len(node.kwargs) == 0


"""
1. create Partition objects from sequences of fusable nodes
2. use fuse_partitions to recreate the graph torch._inductor.fx_passes.group_batch_fusion
"""
def pointwise_fusion(
    cc: CodeCache,
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    fuse_inputs: bool = False,
    fuse_with_compute=True
) -> torch.fx.GraphModule:
    # find all groups of nodes that can be fused and assign to
    # unique partition id, i.e. map_node

    fg = FlowGraph(mod)

    ShapeProp(mod).propagate(*example_inputs)

    node_map = dict()
    partition = 0

    def map_node(n: torch.fx.Node) -> int:
        return node_map[n]

    # assumption, graph.nodes are in topo order
    mod.graph.lint()

    logger.debug("start fusion")

    # create partition groups
    # run in reverse order so predecesors of non-unary ops will appear
    # in the same partition.
    for n in reversed(mod.graph.nodes):
        logger.debug(f"CONSIDER {n}")

        if n.op != 'call_function':
            logger.debug(f"  REJECT {n} not call")
            node_map[n] = 0
            continue

        # TODO: handle get_attr ops
        # should probably be lifted/put in partition 0 but not prevent fusion

        pred = is_fusable_pair if not fuse_with_compute else is_compute_fusable_pair

        fusable = [pred(s, n) for s in fg.predecessors(n)]
        if not all(fusable):
            if not n in node_map:
                logger.debug(f"  REJECT {n} no fusable preds and not in map: {fusable}, {fg.predecessors(n)}")
                node_map[n] = 0
            continue

        # don't support anything with kwargs for now
        if not supported_kwargs(n):
            logger.debug(f"  REJECT {n} unsupported kwargs")
            node_map[n] = 0
            continue

        if n not in node_map:
            partition = partition + 1
            node_map[n] = partition

        for s in fg.predecessors(n):
            node_map[s] = node_map[n]

    logger.debug(f"node_map = {node_map}")

    def same_partition(nodes: Set[torch.fx.Node]) -> bool:
        if len(nodes) > 0:
            part = node_map[next(iter(nodes))]
            return all([node_map[n] == part for n in nodes])
        return False

    def only_pointwise(partition: int) -> bool:
        nodes = [n for n, p in node_map.items() if p == partition]
        return all([is_fusable(n) and not is_compute(n) for n in nodes])


    if fuse_with_compute:
        for n in mod.graph.nodes:
            if n.op != 'call_function':
                continue

            if fuse_inputs:
                nodes = fg.predecessors(n)
            else:
                nodes = fg.successors(n)

            if not is_compute(n):
                continue

            if not same_partition(nodes):
                #logger.debug(f"REJECT {n} not all neighbors in same partition {nodes}")
                continue

            fuse_part = next(iter(nodes))

            if only_pointwise(fuse_part):
                node_map[n] = node_map[fuse_part]

    logger.debug(f"final node_map = {node_map}")

    assert(all([n in node_map for n in mod.graph.nodes]))

    logger.debug(f"pre-fusion split mod:\n{graph_print_tabular(mod.graph, 'part', map_node)}")

    subgraphs = dict()
    for n, p in node_map.items():
        if p > 0:
            if not p in subgraphs:
                subgraphs[p] = []
            subgraphs[p].append(n)

    for p, nodes in subgraphs.items():
        sub = SubGraph(mod, subgraphs[p])
        logger.debug(f"Fusing sub-module:\n{sub.tabular()}")
        fuse_graph_nodes(cc, fgen, sub)
        logger.debug(f"Post fusion sub-module:\n{sub.tabular()}")

    logger.debug(f"Post fusion module:\n{graph_print_tabular(mod.graph)}")

    return mod
