###############################################################################
#
# Utils
#
###############################################################################

import torch

try:
    from tabulate import tabulate  # type: ignore
    have_tabulate = True
except ImportError:
    have_tabulate = False

from collections import OrderedDict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.tools_common import get_node_target

from vllm.logger import init_logger

logger = init_logger(__name__)

EllipsisType = type(...)
NoneType = type(None)


def lazy_string(fn: Callable):
    """
    Lazily convert the result of fn() to a string.
    """

    class lazy:

        def __init__(self, fn: Callable):
            self.fn = fn

        def __str__(self):
            return fn()

    return lazy(lambda: fn())


def trunc(x, lim=30) -> Any:
    """
    Convert x into a string while making sure that it stays under 'lim'
    characters.  If x is a tuple/list/dict, the elements will be
    truncated to 'lim' characters.
    """
    if isinstance(x, tuple):
        return tuple(map(lambda v: trunc(v, lim), x))
    elif isinstance(x, list):
        return [trunc(y, lim) for y in x]
    elif isinstance(x, dict):
        return {trunc(k, lim): trunc(v, lim) for k, v in x.items()}
    xs = str(x)
    return xs if len(xs) <= lim else xs[:lim]


def graph_print_tabular(g: torch.fx.Graph,
                        col: Optional[str] = None,
                        col_get: Optional[Callable] = None) -> str:
    """
    Similar to torch.fx.Graph.print_tabular except it returns a string and
    allows the addition of extra columns.
    """
    if not have_tabulate:
        return str(g)

    assert (col and col_get) or (not col and not col_get)

    headers = ['opcode', 'name', 'target', 'args', 'kwargs']

    if col_get:
        assert col
        headers.append(str(col))
        node_specs = [[
            n.op,
            trunc(n.name),
            trunc(n.target), n.args, n.kwargs,
            col_get(n)
        ] for n in g.nodes]
    else:
        node_specs = [[n.op,
                       trunc(n.name),
                       trunc(n.target), n.args, n.kwargs] for n in g.nodes]

    return tabulate(node_specs, headers=headers)


def lazy_graph_print_tabular(g: torch.fx.Graph,
                             col: Optional[str] = None,
                             col_get: Optional[Callable] = None):
    """
    Lazily print a tabular graph. This defers calling graph_print_tabular
    until the resulting object is converted to a string.  Useful for logging.
    """
    return lazy_string(lambda: graph_print_tabular(g, col, col_get))


def lazy_module_print_readable(gm: torch.fx.GraphModule,
                               print_outputs: bool = True):
    """
    Lazily print a readable graph module. This defers calling print_readable
    until the resulting object is converted to a string.  Useful for logging.
    """
    return lazy_string(lambda: gm.print_readable(print_outputs))


def is_call(node: torch.fx.Node) -> bool:
    """
    Is the given node a call of some kind?
    """
    return (node.op == 'call_function' or node.op == 'call_method'
            or node.op == 'call_module')


def is_simple_call(node: torch.fx.Node) -> bool:
    """
    Is the given node a call of some kind?
    """
    return (node.op == 'call_function' or node.op == 'call_method')


def node_function_target(node: torch.fx.Node) -> str:
    """
    Get the name of the function being called in a 'call_function' op.
    """
    assert is_call(node)
    mod = None if node.op != 'call_module' else node.graph.owning_module
    return get_node_target(mod, node)


def call_method_class(node: torch.fx.Node):
    """
    Find class for method called by node.
    """
    assert node.op == 'call_method'
    ex_val = node.args[0].meta.get('example_value')
    assert ex_val is not None
    return type(ex_val)


def extract_node_type(n: torch.fx.Node):
    """
    Get the data type (float, int, fp16, etc.)  of the tensor associated with
    the given node.
    """
    if 'tensor_meta' in n.meta:
        return n.meta['tensor_meta'].dtype
    else:
        return None


def argument_type_str(arg: torch.fx.node.Argument,
                      include_constants: bool = False):
    """
    Return a string representation of the type of the given argument.  This is
    used for name mangling.
    """
    if isinstance(arg, torch.fx.Node):
        ty = extract_node_type(arg)
        return str(ty) if ty else arg.meta.get('type').__name__
    elif isinstance(arg, torch.Tensor):
        return str(arg.dtype)
    elif isinstance(arg, torch.dtype):
        return str(arg)
    elif isinstance(arg, (str, int, float, bool)):
        if include_constants:
            arg_name = str(arg).replace('-', '_').replace('.', '_')
            return f"{type(arg).__name__}_{arg_name}"
        else:
            return type(arg).__name__
    elif isinstance(arg, (EllipsisType, NoneType)):
        return str(arg)
    elif isinstance(arg, tuple):
        return "T_" + "_".join(
            [argument_type_str(a, include_constants) for a in arg])
    elif isinstance(arg, slice):
        return f"S_{arg.start}_{arg.stop}_{arg.step}"
    elif isinstance(arg, torch.device):
        return f"D_{arg.type}_{arg.index}"
    else:
        raise RuntimeError(f"unsupported argument type {arg}")


def mangle_name(nodes: List[torch.fx.Node], rep: str = "_P_") -> str:
    """
    Generate a mangled name from a list of call_function nodes.
    The mangled name includes the names of all the operators and their types.
    """
    name = ""
    sep = ""
    for n in nodes:
        fn = node_function_target(n)
        types = [
            argument_type_str(arg).replace("torch.", "")
            for arg in n.args
        ]
        ktypes = [
            f"K_{name}_{argument_type_str(kwarg, True).replace('torch.', '')}"
            for name, kwarg in n.kwargs.items()
        ]
        name = name + sep + f"{fn}_{'_'.join(types)}{'_'.join(ktypes)}"
        sep = "_"

    return name.replace(".", rep)


class ModuleInputGenerator(FakeTensorProp):
    """
    Generate example inputs for all submodules in the given GraphModule.
    After running propagate the 'module_args' property will hold a map of
    module name to list of example inputs.

    For now, if a particular submodule is called from more than one location,
    'module_args' will contain 'None'.  This could be made smarter but is not
    necessary for now since we currently only care about single use submodules.

    TODO: can this be combined with ShapeProp somehow?
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        mode: Optional[FakeTensorMode] = None,
    ):
        super().__init__(module, mode)
        self.module_args: Dict[torch.fx.node.Target,
                               Tuple[Optional[Tuple[torch.fx.node.Argument,
                                                    ...]],
                                     Optional[Dict[str, Any]]]] = {}

    def call_module(self, target: torch.fx.node.Target,
                    args: Tuple[torch.fx.node.Argument,
                                ...], kwargs: Dict[str, Any]) -> Any:
        # Problem here with multiple call sites and different args,
        # for now set to None if there are multiple callers.
        # Could check for "compatible" inputs and allow.
        if target in self.module_args:
            self.module_args[target] = (None, None)
        else:
            self.module_args[target] = (args, kwargs)

        return super().call_module(target, args, kwargs)


def get_function_schema(n: torch.fx.Node) -> Optional[torch._C.FunctionSchema]:
    """
    Find a function schema (if any) matching the callsite in 'n'.
    """
    assert is_call(n)

    sigs, schemas = torch.fx.operator_schemas.get_signature_for_torch_op(
        n.target, return_schemas=True)

    if schemas is None:
        return None

    matched_schemas = []
    for candidate_signature, schema in zip(sigs, schemas):
        try:
            candidate_signature.bind(*n.args, **n.kwargs)
            matched_schemas.append((candidate_signature, schema))
        except TypeError:
            continue

    if len(matched_schemas) == 0:
        # Did not match any schema. Cannot check for mutation
        return None

    if len(matched_schemas) != 1:
        logger.debug("ambiguous sig failure: %s, %s", n.format_node(),
                     matched_schemas)
        return None

    _, s = matched_schemas[0]

    return s


def mutable_function_args(n: torch.fx.Node) -> List[Union[int, str]]:
    """
    Return a list of all the mutable argument indices for the callsite
    in 'n'.
    """
    mutable_arg_indices: List[Union[int, str]] = []

    if not is_call(n):
        return mutable_arg_indices

    s = get_function_schema(n)
    if not s or not s.is_mutable:
        return mutable_arg_indices

    num_outputs = len([a for a in s.arguments if a.is_out])

    for i, a in enumerate(s.arguments):
        if a.alias_info and a.alias_info.is_write:
            if not a.kwarg_only:
                mutable_arg_indices.append(i - num_outputs)
            else:
                mutable_arg_indices.append(a.name)

    return mutable_arg_indices


def nth_arg_or_kwarg(n: torch.fx.Node, arg: Union[int, str]):
    """
    Return the nth argument (or kwarg) of the given callsite.
    """
    # TODO: see node.normalized_arguments
    if isinstance(arg, int):
        if arg >= len(n.args):
            return list(n.kwargs.values())[arg - len(n.args)]
        else:
            return n.args[arg]
    else:
        return n.kwargs[arg]


def dump_inputs_users(
    nodes: List[torch.fx.Node],
    all_input_nodes: OrderedDict[torch.fx.Node, List[torch.fx.Node]],
    all_node_users: OrderedDict[torch.fx.Node, OrderedDict[torch.fx.Node,
                                                           None]]
) -> str:
    """
    Pretty print inputs/users info for a set of nodes and tag where
    they differ from node.all_input_nodes and node.users.
    """
    if not have_tabulate:
        return "dump_inputs_users: tabulate not installed"

    headers = ['name', 'inputs', 'users']

    entries = [[
        trunc(n.name),
        (f"{trunc(all_input_nodes[n])}"
         f"{'***' if n.all_input_nodes != all_input_nodes[n] else ''}"),
        (f"{trunc(all_node_users[n])}"
         f"{'***' if n.users != all_node_users[n] else ''}"),
    ] for n in nodes]

    return tabulate(entries, headers=headers)


def gather_all_input_nodes(
    nodes: List[torch.fx.Node],
    do_renames: bool = True
) -> Tuple[OrderedDict[torch.fx.Node, List[torch.fx.Node]], OrderedDict[
        torch.fx.Node, OrderedDict[torch.fx.Node, None]]]:
    """
    Collect all def/use information for each node in 'nodes'.  This is different
    than node.all_input_nodes and node.users since it handles in-place
    functions.  It is assumed that any mutable input in an in-place function is
    also an output.

    Run a pass over the graph (graph must be topo sorted)
    1. keep map of renames
    2. if call node has a mutable input
       - the call node must record itself as user of the mutable input (modulo
         renames)
       - the output of that call node will be the "rename" of the mutable input
         until the next use of the original name.
         (there could be multiple mutable nodes with the same rename)
       - only one rename for a node should be active at one point.
    3. Use rename maps to track proper input nodes
    4. make this return inputs + users for all nodes

    Note: this will include Node kwargs
    """
    all_input_nodes: OrderedDict[torch.fx.Node,
                                 List[torch.fx.Node]] = OrderedDict()
    all_node_users: OrderedDict[torch.fx.Node,
                                OrderedDict[torch.fx.Node,
                                            None]] = OrderedDict()
    renames: Dict[torch.fx.Node, torch.fx.Node] = dict()

    def process_arg(n: torch.fx.Node, arg: torch.fx.node.Argument):
        if isinstance(arg, tuple):
            for sub_arg in arg:
                process_arg(n, sub_arg)
            return

        if not isinstance(arg, torch.fx.Node):
            return

        renames[arg] = n

    def rename_inputs(n: torch.fx.Node):
        for i, inp in enumerate(all_input_nodes[n]):
            all_input_nodes[n][i] = renames[inp] if inp in renames else inp

    def rename_users(n: torch.fx.Node):
        for user in all_node_users[n]:
            if user in renames:
                del all_node_users[n][user]
                all_node_users[n][user] = None

    #
    # populate maps up front
    #
    for n in nodes:
        all_input_nodes[n] = n.all_input_nodes
        all_node_users[n] = OrderedDict(
            sorted(n.users.items(), key=lambda kv: kv[0].name))

    #
    # process each node
    #
    for n in nodes:
        if do_renames:
            rename_inputs(n)
            rename_users(n)

        for i in mutable_function_args(n):
            arg = nth_arg_or_kwarg(n, i)
            process_arg(n, arg)

    # Update users here
    for n in nodes:
        for inp in all_input_nodes[n]:
            if n not in all_node_users[inp]:
                all_node_users[inp][n] = None

    logger.debug(dump_inputs_users(nodes, all_input_nodes, all_node_users))

    return all_input_nodes, all_node_users


class FlowGraph:
    """
    The FlowGraph is a dataflow graph for a fx.GraphModule.
    The nodes are fx.Nodes and the edges represent the producers (inputs) and
    consumers (outputs) of each operation.

    The FlowGraph is invalidated if the underlying GraphModule is modified.
    It can be regenerated at any time by calling the `build` method.

    TODO: turn getitems into "reader views"?
    """

    def __init__(self, gm: torch.fx.GraphModule):
        self.module = gm
        self.build()

    def add_edge(self, src: torch.fx.GraphModule, dst: torch.fx.GraphModule):
        if src not in self.succs:
            self.succs[src] = set()
        if dst not in self.preds:
            self.preds[dst] = set()

        self.succs[src].add(dst)
        self.preds[dst].add(src)

    def build(self):
        """
        Construct the FlowGraph.
        """
        self.succs = dict()
        self.preds = dict()
        self.outputs = [n for n in self.module.graph.nodes if n.op == 'output']
        self.inputs = [
            n for n in self.module.graph.nodes if n.op == 'placeholder'
        ]
        visited = set()
        q = self.outputs

        self.all_renamed_input_nodes, self.all_renamed_node_users = (
            gather_all_input_nodes(self.module.graph.nodes, True))

        self.all_input_nodes, self.all_node_users = (gather_all_input_nodes(
            self.module.graph.nodes, False))

        while len(q) > 0:
            n = q.pop()
            if n in visited:
                continue

            visited.add(n)
            for input in self.all_renamed_input_nodes[n]:
                self.add_edge(input, n)
                q.append(input)

    def successors(self, n: torch.fx.Node) -> Set[torch.fx.Node]:
        return self.succs[n] if n in self.succs else set()

    def predecessors(self, n: torch.fx.Node) -> Set[torch.fx.Node]:
        return self.preds[n] if n in self.preds else set()

    def visit(self, fn: Callable):
        q = self.inputs
        visited = set()
        while len(q) > 0:
            n = q.pop()
            if n in visited:
                continue
            visited.add(n)
            fn(n)
            q = list(self.successors(n)) + q

    def dump(self) -> str:
        res = ""
        oc = '{'
        cc = '}'
        for n in self.module.graph.nodes:
            res = res + (f"{n} ({n.op}) {oc}\n  "
                         f"preds={str(self.predecessors(n))}\n  "
                         f"succs={str(self.successors(n))}\n{cc}\n")
        return res


class SubGraph:
    """
    A class representing a set of nodes somewhat like a virtual GraphModule.
    """

    def __init__(self, fg: FlowGraph, nodes: List[torch.fx.Node]):
        self.module = fg.module
        self.inputs: List[torch.fx.Node] = []
        self.outputs: List[torch.fx.Node] = []
        self.nodes: List[torch.fx.Node] = []
        self.all_input_nodes = fg.all_input_nodes
        self.all_node_users = fg.all_node_users
        self.all_renamed_input_nodes = fg.all_renamed_input_nodes
        self.all_renamed_node_users = fg.all_renamed_node_users
        self.build(nodes)

    def in_subgraph(self, n: torch.fx.Node) -> bool:
        return n in self.nodes

    def _collect_inputs_outputs(
            self) -> Tuple[List[torch.fx.Node], List[torch.fx.Node]]:
        inputs = []
        outputs = []

        all_input_nodes, all_node_users = (self.all_input_nodes,
                                           self.all_node_users)

        for n in self.nodes:
            new_inputs = [
                inp for inp in all_input_nodes[n] if not self.in_subgraph(inp)
            ]
            for inp in new_inputs:
                if inp not in inputs:
                    inputs.append(inp)

            if any([
                    user
                    for user in all_node_users[n] if not self.in_subgraph(user)
            ]) and n not in outputs:
                outputs.append(n)

        return inputs, outputs

    def topo_sort(self):
        order = []
        in_degree = dict()
        worklist: deque = deque()

        all_input_nodes, all_node_users = (self.all_renamed_input_nodes,
                                           self.all_renamed_node_users)

        for n in sorted(self.nodes, key=lambda n: n.name):
            count = len(
                [inp for inp in all_input_nodes[n] if self.in_subgraph(inp)])
            in_degree[n] = count
            if count == 0:
                worklist.append(n)

        while len(worklist) > 0:
            n = worklist.popleft()
            order.append(n)

            for u in all_node_users[n]:
                if not self.in_subgraph(u):
                    continue
                in_degree[u] = in_degree[u] - 1
                if in_degree[u] == 0:
                    worklist.append(u)

        # Check for cycles (should not be any).
        assert len(order) == len(
            self.nodes), f"cycle found: ({order}) != ({self.nodes})"

        self.nodes = order

    def build(self, nodes: List[torch.fx.Node]):
        """
        Construct the SubGraph
        """
        self.nodes = nodes
        self.topo_sort()
        self.inputs, self.outputs = self._collect_inputs_outputs()

    def first_in_subgraph(self):
        first = None
        for n in self.module.graph.nodes:
            if not first and n in self.nodes:
                first = n
                break
        return first

    def last_input(self):
        # there has to be a smarter way to do this
        first = next(iter(self.module.graph.nodes))
        candidates = set(self.inputs)

        for n in self.inputs:
            p = n.prev
            if len(candidates) == 1:
                break

            while p != first.prev:
                if p in self.inputs:
                    candidates.remove(p)
                    break
                p = p.prev

        assert len(candidates) == 1

        return candidates.pop()

    def _refresh_def_use(self):
        self.all_renamed_input_nodes, self.all_renamed_node_users = (
            gather_all_input_nodes(self.module.graph.nodes, True))

        self.all_input_nodes, self.all_node_users = gather_all_input_nodes(
            self.module.graph.nodes, False)

    def erase(self):
        """
        Erase all the nodes in the SubGraph.
        """
        for n in reversed(self.nodes):
            self.module.graph.erase_node(n)

        # TODO: be smarter with updating just for deleted/new nodes
        self._refresh_def_use()

    def tabular(self,
                col: Optional[str] = None,
                col_get: Optional[Callable] = None) -> str:
        """
        Print a SubGraph in tabular form.
        """
        if not have_tabulate:
            return "SubGraph::tabular: tabulate not installed"

        assert (col and col_get) or (not col and not col_get)

        headers = ['opcode', 'name', 'target', 'args', 'kwargs']

        if col_get:
            assert col
            headers.append(str(col))

        def mcol_get(x):
            if col_get:
                return [trunc(col_get(x))]
            else:
                return []

        node_specs = [[
            'placeholder*',
            trunc(n.name),
            trunc(n.target),
            tuple(),
            dict(), *mcol_get(n)
        ] for n in self.inputs]

        node_specs = node_specs + [[
            n.op,
            trunc(n.name),
            trunc(n.target),
            trunc(n.args), n.kwargs, *mcol_get(n)
        ] for n in self.nodes]

        node_specs = node_specs + [[
            'output*',
            'output*',
            'output*',
            (trunc(n), ),
            dict(),
            *mcol_get(n),
        ] for n in self.outputs]

        return tabulate(node_specs, headers=headers)

def contains_constant(arg: torch.fx.node.Argument):
    # TODO: tuple isn't right unless its composed of constants????
    if isinstance(arg, tuple):
        return True
    elif isinstance(arg, (str, int, float, bool)):
        return True
    else:
        return False

# This takes in an fx node that has at least one const argument. 
# Count should be incremented by the caller. Once for each const argument
def generate_const_name(node: torch.fx.node, count):
    return f"{node.name}_const_{count}"

def extract_constant_vals(arg: torch.fx.node.Argument):
    if isinstance(arg, tuple):
        val = ()
        for elem in arg:
            val += extract_constant_vals(elem)
        return val
    if isinstance(arg, slice):
        if arg.stop:
            assert arg.start == None
            return (arg.stop,)
        else:
            assert arg.start is not None
            assert arg.stop == None
            return (arg.start,)
    elif isinstance(arg, (str, int, float, bool)):
        return (arg,)
    else:
        return ()