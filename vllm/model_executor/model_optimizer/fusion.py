###############################################################################
#
# Operator fusion pass
#
###############################################################################

import operator
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Set

import torch
from torch.fx.passes.shape_prop import ShapeProp

from vllm.logger import init_logger

from .code_cache import CodeCache
from .fused_op_generator import FusionFail
from .naive_fused_op_generator import NaiveFusedOpGenerator
from .register import FUSABLE
from .utils import (FlowGraph, SubGraph, extract_constant_vals,
                    generate_const_name, is_simple_call,
                    lazy_graph_print_tabular, mangle_name,
                    node_function_target, simplify_mangled_name)

logger = init_logger(__name__)


def extract_named_constants(named_args: OrderedDict[str,
                                                    torch.fx.node.Argument],
                            sub_graph: SubGraph):
    # iterate over each node in the subgraph
    for n in sub_graph.nodes:
        count = 1
        # iterate over each of that nodes arguments
        for arg in n.args:
            const_vals = extract_constant_vals(arg)
            for const_val in const_vals:
                const_name = generate_const_name(n, count)
                count += 1
                named_args[const_name] = const_val


def fuse_graph_nodes(cc: CodeCache, sub: SubGraph):
    """
    Fuse all the nodes in the given sub-graph into a single function call.
    """
    outputs = sub.outputs
    inputs = sub.inputs

    named_args: OrderedDict[str, torch.fx.node.Argument] = OrderedDict()

    # Add each subgraph input along with its name to "named_args"
    for input in inputs:
        named_args[input.name] = input

    # Extract any constants from inside the subgraph and generate arguments
    # to represent them.
    # extract_named_constants(named_args, sub)

    sub.topo_sort()

    # Collect all the nodes that will need to be fused (and erased) later.
    nodes_to_fuse = []
    kwargs: Dict[torch.fx.Node, Dict[str, torch.fx.Argument]] = dict()

    for n in sub.nodes:
        # XXXXXX remove debugging code
        if False and n.name == 'output_160':
            print(f"sorted {sub.nodes}\n{sub.tabular()}")

        if not is_simple_call(n):
            continue

        nodes_to_fuse.append(n)

    # Lookup or create the fused operation.
    try:
        fn_key = simplify_mangled_name(f"{mangle_name(nodes_to_fuse)}_fused")

        def generate() -> Optional[Callable]:
            fgen = NaiveFusedOpGenerator()
            return fgen.make_fused_op(fn_key, named_args, outputs,
                                      nodes_to_fuse, kwargs)

        fn = cc.lookup_or_create(fn_key, generate)

    except FusionFail as ff:
        logger.info("fusion failed '%s' for subgraph.", ff)
        return

    if fn is None:
        logger.debug("fusion failed previously for subgraph.")
        return

    logger.debug("fused fn = %s", fn)

    #
    # Update the graph
    # 1. insert the call_function for the fused op
    # 2. insert new output node(s)
    # 3. delete old call_function and output nodes.
    #

    insert_point = sub.last_input()
    sub.module.graph.inserting_after(insert_point)

    # Note: we do not update the meta info for cf here.  It should
    # not be required after transformation anyway.
    cf = sub.module.graph.call_function(fn,
                                        args=tuple(named_args.values()),
                                        kwargs=kwargs)
    logger.debug("fused op: %s, num_outputs=%s", cf.format_node(),
                 len(outputs))

    new_sub = [cf]

    # Note: assumes single output
    if len(outputs) == 1:
        outputs[0].replace_all_uses_with(cf, propagate_meta=True)
    else:
        cf_idx = cf
        for i, output in enumerate(outputs):
            sub.module.graph.inserting_after(cf_idx)
            cf_idx = sub.module.graph.call_function(operator.getitem,
                                                    args=(cf, i))
            output.replace_all_uses_with(cf_idx, propagate_meta=True)
            new_sub.append(cf_idx)

    # Erase all the nodes in the subgraph
    sub.erase()

    # Extra build() not necessary but nice for debugging.
    sub.build(new_sub)


def is_fusable(node: torch.fx.Node) -> bool:
    """
    Determine whether or not node is a fusable operations.
    TODO: Smarter filter for 'getitem'.
    """
    if not is_simple_call(node):
        return False

    op_name = node_function_target(node)
    if node.op == 'call_function':
        return op_name in FUSABLE and not FUSABLE[op_name].is_compute
    else:
        # TODO: check class type
        # class_type = call_method_class(node)
        return op_name in FUSABLE and not FUSABLE[op_name].is_compute


def is_compute(node: torch.fx.Node) -> bool:
    """
    Determine whether or not node is a fusable compute operation, e.g. gemm.
    """
    if not is_simple_call(node):
        return False

    op_name = node_function_target(node)
    if node.op == 'call_function':
        return op_name in FUSABLE and FUSABLE[op_name].is_compute
    else:
        # TODO: check class type
        # class_type = call_method_class(node)
        return op_name in FUSABLE and FUSABLE[op_name].is_compute


def is_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    """
    Are nodes a and b fusable together?
    This function assumes 'b' is a direct successor of 'a'.
    """
    return is_fusable(a) and is_fusable(b)


def is_compute_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    """
    Are nodes 'a' and 'b' fusable together and is 'a' optionally a compute op?
    This function assumes 'b' is a direct successor of 'a'.
    """
    return (is_fusable(a) or is_compute(a)) and is_fusable(b)


def supported_kwargs(node: torch.fx.Node,
                     only_const_kwargs: bool = True) -> bool:
    """
    Determine if any kwargs associated with 'node' are supported.
    """
    if only_const_kwargs:
        for arg in node.kwargs.values():
            if not isinstance(arg, torch.fx.node.BaseArgumentTypes):
                return False
        return True
    else:
        return node.kwargs is None or len(node.kwargs) == 0


def dump_partitions(node_map: Dict[torch.fx.Node, int]) -> str:
    parts = dict()
    for n, p in node_map.items():
        if p not in parts:
            parts[p] = set([n])
        else:
            parts[p].add(n)

    part_str = "{\n"
    for p, ns in parts.items():
        part_str = part_str + f"  {p}: {str(ns)}\n"
    part_str = part_str + "\n}"
    return part_str


def non_trivial_op(n: torch.fx.Node) -> bool:
    if not is_simple_call(n):
        return False
    trg = node_function_target(n)
    return not FUSABLE[trg].is_trivial if trg in FUSABLE else False


def is_get_attr(a: torch.fx.Node) -> bool:
    return a.op == 'get_attr'


def validate_path(path: List[torch.fx.Node], fuse_part_id: int,
                  node_map: Dict[torch.fx.Node, int]) -> bool:
    saw_non_part = False
    for p in path:
        curr_id = node_map[p] if p in node_map else 0
        if curr_id == fuse_part_id and saw_non_part:
            return False
        if curr_id != fuse_part_id:
            saw_non_part = True
    return True


def valid_subgraph_partition(fg: FlowGraph, part_g: SubGraph,
                             node_map: Dict[torch.fx.Node,
                                            int], fuse_part_id) -> bool:

    first = [part_g.first_in_subgraph()]
    last = [part_g.last_in_subgraph()]
    paths = fg.paths_to_nodes(first, last)

    for path in paths:
        if not validate_path(path, fuse_part_id, node_map):
            logger.debug("Found bad path: %s", str(path))
            return False

    return True


# UNUSED, probably delete
def valid_partition(fg: FlowGraph, n: torch.fx.Node,
                    node_map: Dict[torch.fx.Node,
                                   int], num_compute, fuse_part_id) -> bool:
    #fuse_part_id = node_map[n]
    preds = fg.predecessors(n)

    tmp_node_map = node_map.copy()
    tmp_node_map[n] = fuse_part_id

    for s in preds:
        if s.op != 'placeholder' and (not is_compute(s)
                                      or num_compute[fuse_part_id] == 0):
            tmp_node_map[s] = fuse_part_id

    part_g = SubGraph(
        fg, [n for n, p in tmp_node_map.items() if p == fuse_part_id])

    return valid_subgraph_partition(fg, part_g, tmp_node_map, fuse_part_id)


def pointwise_fusion(cc: CodeCache,
                     mod: torch.fx.GraphModule,
                     example_inputs: List[torch.Tensor],
                     fuse_inputs: bool = False,
                     fuse_with_compute=True) -> torch.fx.GraphModule:
    # find all groups of nodes that can be fused and assign to
    # unique partition id, i.e. node_map

    fg = FlowGraph(mod)

    logger.debug("FlowGraph:\n%s", fg.dump())
    #print(f"FlowGraph:\n{fg.dump()}")
    #print(f"MODULE\n{mod}")
    #print(f"FlowGraph (dot):\n{fg.to_dot('orig')}")

    ShapeProp(mod).propagate(*example_inputs)

    # Map of nodes to partitions.  Partition 0 represents unfusable operations.
    node_map: Dict[torch.fx.Node, int] = dict()

    # TODO: make UNFUSED_PARTITION = 0 constant
    partition = 0

    # assumption, graph.nodes are in topo order
    mod.graph.lint()

    logger.debug("Start fusion")

    num_compute: Dict[int, int] = dict()
    num_compute[0] = 0

    num_inputs: Dict[int, int] = dict()

    def set_partition(n: torch.fx.Node, part_id: int):
        node_map[n] = part_id

        if part_id not in num_compute:
            num_compute[part_id] = 0

        if part_id not in num_inputs:
            num_inputs[part_id] = 0

        if is_compute(n):
            num_compute[part_id] = num_compute[part_id] + 1

        # conservative count of fused op input arguments.
        num_inputs[part_id] = num_inputs[part_id] + len(n.args)

        logger.debug("SET_PARTITION[%s] = %d (%s), %d", n, part_id,
                     is_compute(n), num_inputs[part_id])

    # create partition groups
    # run in reverse order so predecessors of non-unary ops will appear
    # in the same partition.
    for n in reversed(mod.graph.nodes):
        logger.debug("CONSIDER %s", n)

        if not is_simple_call(n):
            logger.debug("  REJECT %s: not call", n)
            node_map[n] = 0
            continue

        pred = (is_fusable_pair
                if not fuse_with_compute else is_compute_fusable_pair)

        fusable = [
            pred(s, n) for s in fg.predecessors(n)
            if s.op != 'placeholder' and not is_get_attr(s)
        ]
        if not all(fusable):
            logger.debug("  REJECT %s[%d]: not all preds fusable: %s, %s", n,
                         node_map[n] if n in node_map else 0, fusable,
                         fg.predecessors(n))

            if n not in node_map:
                node_map[n] = 0
            continue

        # don't support anything with kwargs for now
        if not supported_kwargs(n):
            logger.debug("  REJECT %s: unsupported kwargs", n)
            node_map[n] = 0
            continue

        if n not in node_map:
            partition = partition + 1
            set_partition(n, partition)

        fuse_part_id = node_map[n]

        # Pytorch only supports functions with up to 64 inputs.
        if fuse_part_id in num_inputs and num_inputs[fuse_part_id] >= 63:
            logger.debug("  REJECT %s: too many inputs in partition %d", n,
                         fuse_part_id)
            continue

        # Add predecessors to partition
        for s in fg.predecessors(n):
            if s.op != 'placeholder' and (not is_compute(s)
                                          or num_compute[fuse_part_id] == 0):
                set_partition(s, fuse_part_id)

        logger.debug("ACCEPT %s: partition %s", n, fuse_part_id)

    logger.debug("partitions = %s", dump_partitions(node_map))

    def same_partition(nodes: Set[torch.fx.Node]) -> bool:
        if len(nodes) > 0:
            part = node_map[next(iter(nodes))]
            return all([node_map[n] == part for n in nodes])
        return False

    def only_pointwise(partition: int) -> bool:
        nodes = [n for n, p in node_map.items() if p == partition]
        return all([is_fusable(n) and not is_compute(n) for n in nodes])

    #
    # Fuse pointwise op stacks with compute nodes. The fuse_inputs flag controls
    # prologue or epilogue fusion.
    #
    if fuse_with_compute:
        assert all([(num <= 1 or part == 0)
                    for part, num in num_compute.items()])

        for n in mod.graph.nodes:
            if not is_simple_call(n):
                continue

            logger.debug("COMPUTE CONSIDER %s", n)

            nodes = fg.predecessors(n) if fuse_inputs else fg.successors(n)

            if not is_compute(n):
                logger.debug("COMPUTE REJECT %s: not compute", n)
                continue

            if not same_partition(nodes):
                logger.debug(
                    "COMPUTE REJECT %s: not all neighbors in same partition %s",
                    n, str(nodes))
                continue

            fuse_part = next(iter(nodes))

            if not only_pointwise(fuse_part):
                logger.debug(
                    "COMPUTE REJECT %s: not only_pointwise in users %s", n,
                    str(fuse_part))
                continue

            fuse_part_id = node_map[fuse_part]

            if (fuse_part_id != 0 and fuse_part_id in num_compute
                    and num_compute[fuse_part_id] >= 1):
                logger.debug("COMPUTE REJECT %s: already a compute node in %s",
                             n, str(node_map[fuse_part]))
                continue

            logger.debug("COMPUTE ACCEPT %s: partition %s, nodes=%s", n,
                         fuse_part_id, str(nodes))

            set_partition(n, fuse_part_id)

    logger.debug("final partitions = %s", dump_partitions(node_map))

    # Make sure all nodes have been assigned a partition.
    assert all([n in node_map for n in mod.graph.nodes])

    logger.debug("pre-fusion split mod:")
    logger.debug(
        lazy_graph_print_tabular(mod.graph, 'part', lambda x: node_map[x]))

    # Create subgraph for every fusable partition.
    subgraphs: Dict[int, List[torch.fx.Node]] = dict()
    for n, p in node_map.items():
        if p > 0:
            if p not in subgraphs:
                subgraphs[p] = []
            subgraphs[p].append(n)

    to_delete = []
    for p in subgraphs:
        sub = SubGraph(fg, subgraphs[p])  # TODO: only do this once
        #subgraphs[p] = sub

        if len([n for n in sub.nodes if non_trivial_op(n)]) <= 1:
            logger.debug("\nReject empty/singleton/trivial subgraph:\n%s",
                         sub.tabular())
            to_delete.append(p)
            continue

        if not valid_subgraph_partition(fg, sub, node_map,
                                        node_map[sub.nodes[0]]):
            logger.debug("\nREJECT %d:\n%s\n", node_map[sub.nodes[0]],
                         sub.tabular())
            to_delete.append(p)
            continue

    for p in to_delete:
        del subgraphs[p]

    logger.debug("Found %s fusable subgraphs.", len(subgraphs))

    #print(f"FlowGraph (sub):\n{FlowGraph(mod).to_dot('hilite', subgraphs.values())}") # noqa: E501

    # Attempt to fuse each subgraph.
    for p in subgraphs:
        sub = SubGraph(fg, subgraphs[p])
        logger.debug("\nFusing sub-module (last_input=%s):\n%s",
                     sub.last_input(), sub.tabular())
        #print(f"\nFusing sub-module (last_input={sub.last_input()}):\n{sub.tabular()}")
        #print(f"fusing subgraph: {sub.to_dot(str(p))}")
        fuse_graph_nodes(cc, sub)
        logger.debug("Post fusion sub-module:\n%s", sub.tabular())

    logger.debug("Post fusion module:")
    logger.debug(lazy_graph_print_tabular(mod.graph))

    #print(f"Final FlowGraph (dot):\n{FlowGraph(mod).to_dot('final', [[n] for n in mod.graph.nodes if cc.has_entry(n.name)])}") # noqa: E501

    return mod
