###############################################################################
#
# Utils
#
###############################################################################

import collections
import copy
import functools
import torch
import torch.utils.cpp_extension
import types

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.tools_common import get_node_target
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set, Union

from vllm.logger import init_logger

logger = init_logger(__name__)


def trunc(x) -> str:
    lim = 30
    if isinstance(x, tuple):
        return tuple(map(trunc, x))
    elif isinstance(x, list):
        return [trunc(y) for y in x]
    elif isinstance(x, dict):
        return {trunc(k): trunc(v) for k,v in x.items()}
    xs = str(x)
    return xs if len(xs) <= lim else xs[:lim]


"""
Similar to torch.fx.Graph.print_tabular except it returns a string and
allows the addition of extra columns.
"""
def graph_print_tabular(g: torch.fx.Graph,
                        col: Optional[str] = None,
                        col_get: Optional[Callable] = None) -> str:
    try:
        from tabulate import tabulate
    except ImportError:
        print("`print_tabular` relies on the library `tabulate`, "
              "which could not be found on this machine. Run `pip "
              "install tabulate` to install the library.")
        raise

    assert (col and col_get) or (not col and not col_get)

    headers = ['opcode', 'name', 'target', 'args', 'kwargs']

    if col_get:
        headers.append(col)
        node_specs = [[n.op, trunc(n.name), trunc(n.target), n.args, n.kwargs,
                       col_get(n)] for n in g.nodes]
    else:
        node_specs = [[n.op, trunc(n.name), trunc(n.target), n.args, n.kwargs]
                      for n in g.nodes]

    return tabulate(node_specs, headers=headers)


def is_call(node: torch.fx.Node) -> bool:
    return node.op == 'call_function' or node.op == 'call_method'


"""
Get the name of the function being called in a 'call_function' op.
"""
def node_function_target(node: torch.fx.Node) -> str:
    assert is_call(node)
    return get_node_target(None, node)


"""
Return a string representation of the type of the given argument.  This is
used for name mangling.
"""
def argument_type_str(arg: torch.fx.node.Argument, include_constants: bool = False):
    if isinstance(arg, torch.fx.Node):
        ty = extract_node_type(arg)
        return str(ty) if ty else arg.meta.get('type').__name__
    elif isinstance(arg, torch.Tensor):
        return str(arg.dtype)
    elif isinstance(arg, torch.dtype):
        return str(arg)
    elif (isinstance(arg, str) or isinstance(arg, int)
          or isinstance(arg, float) or isinstance(arg, bool)):
        if include_constants:
            return f"{type(arg).__name__}_{str(arg).replace('-','_').replace('.','_')}"
        else:
            return type(arg).__name__
    elif (isinstance(arg, types.EllipsisType)
          or isinstance(arg, types.NoneType)):
        return str(arg)
    elif isinstance(arg, tuple):
        return "T_" + "_".join([argument_type_str(a) for a in arg])
    elif isinstance(arg, slice):
        return f"S_{arg.start}_{arg.stop}_{arg.step}"
    else:
        raise RuntimeError(f"unsupported argument type {arg}")


"""
Get the data type (float, int, fp16, etc.)  of the tensor associated with
the given node.
"""
def extract_node_type(n: torch.fx.Node):
    if 'tensor_meta' in n.meta:
        return n.meta['tensor_meta'].dtype
    else:
        return None

# Find class for method called by node.
def call_method_class(node: torch.fx.Node):  # -> Type:
    assert node.op == 'call_method'
    ex_val = node.args[0].meta.get('example_value')
    assert ex_val is not None
    return type(ex_val)


"""
Compose two functions.
"""
def compose2(f: Callable, g: Callable) -> Callable:
    return lambda *a, **kw: g(f(*a, **kw))


"""
Compose a list of functions.
"""
def compose(*fs: List[Callable]) -> Callable:
    return functools.reduce(compose2, fs)


"""
Generate a mangled name from a list of call_function nodes.
The mangled name includes the names of all the operators and their types.
"""
def mangle_name(nodes: List[torch.fx.Node], rep: str = "_P_") -> str:
    name = ""
    sep = ""
    for n in nodes:
        fn = node_function_target(n)
        types = [
            argument_type_str(arg, True).replace("torch.", "") for arg in n.args
        ]
        name = name + sep + f"{fn}_{'_'.join(types)}"
        sep = "_"

    return name.replace(".", rep)


"""
Generate example inputs for all submodules in the given GraphModule.
After running propagate the 'module_args' property will hold a map of
module name to list of example inputs.

For now, if a particular submodule is called from more than one location, 'module_args'
will contain 'None'.  This could be made smarter but is not necessary for now
since we currently only care about single use submodules.

TODO: can this be combined with ShapeProp somehow?
"""
class ModuleInputGenerator(torch.fx.passes.fake_tensor_prop.FakeTensorProp):

    def __init__(
        self,
        module: torch.fx.GraphModule,
        mode: Optional[FakeTensorMode] = None,
    ):
        super().__init__(module, mode)
        self.module_args = {}

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


def mutable_function_args(n: torch.fx.Node) -> List[Union[int, str]]:
    mutable_arg_indices = []

    if n.op != 'call_function':
        return mutable_arg_indices

    sigs, schemas = torch.fx.operator_schemas.get_signature_for_torch_op(
        n.target, return_schemas=True)

    if schemas is None or not any([s.is_mutable for s in schemas]):
        return mutable_arg_indices

    matched_schemas = []
    for candidate_signature, schema in zip(sigs, schemas):
        try:
            candidate_signature.bind(*n.args, **n.kwargs)
            matched_schemas.append((candidate_signature, schema))
        except TypeError as e:
            continue

    if len(matched_schemas) == 0:
        # Did not match any schema. Cannot check for mutation
        return mutable_arg_indices

    # What to do here?
    if len(matched_schemas) != 1:
        raise Exception("ambiguous sig failure")

    _, s = matched_schemas[0]
    if not s.is_mutable:
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
    if isinstance(arg, int):
        if arg >= len(n.args):
            return list(n.kwargs.values())[arg - len(n.args)]  #????
        else:
            return n.args[arg]
    else:
        return n.kwargs[arg]


def dump_inputs_users(
    nodes: List[torch.fx.Node],
    all_input_nodes: Dict[torch.fx.Node, List[torch.fx.Node]],
    all_node_users: Dict[torch.fx.Node, Dict[torch.fx.Node, None]]
) -> str:
    try:
        from tabulate import tabulate
    except ImportError:
        print("`print_tabular` relies on the library `tabulate`, "
              "which could not be found on this machine. Run `pip "
              "install tabulate` to install the library.")
        raise

    headers = ['name', 'inputs', 'users']

    entries = [
        [
            trunc(n.name),
            f"{trunc(all_input_nodes[n])}{'***' if n.all_input_nodes != all_input_nodes[n] else ''}",
            f"{trunc(all_node_users[n])}{'***' if n.users != all_node_users[n] else ''}",
        ] for n in nodes
    ]

    return tabulate(entries, headers=headers)


# Run a pass over the graph (graph must be topo sorted)
# 1. keep map of renames
# 2. if call node has a mutable input
#    - the call node must record itself as user of the mutable input (modulo renames)
#    - the output of that call node will be the "rename" of the mutable input until the next use of the original name
#      (there could be multiple mutable nodes with the same rename)
#    - only one rename for a node should be active at one point.
# 3. Use rename maps to track proper input nodes
#
# 4. make this return inputs + users for all nodes
#
# Note: this will include Node kwargs
def gather_all_input_nodes(
    nodes: List[torch.fx.Node],
    do_renames: bool = True
) -> Tuple[Dict[torch.fx.Node, List[torch.fx.Node]], Dict[torch.fx.Node, Dict[torch.fx.Node, None]]]:
    all_input_nodes : Dict[torch.fx.Node, List[torch.fx.Node]] = dict()
    all_node_users : Dict[torch.fx.Node, Dict[torch.fx.Node, None]] = dict()
    renames : Dict[torch.fx.Node, torch.fx.Node] = dict()

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
        for user in all_node_users[n].keys():
            if user in renames:
                all_node_users[n].erase(user)
                all_node_users[n][user] = None

    #
    # populate maps up front
    #
    for n in nodes:
        all_input_nodes[n] = n.all_input_nodes
        all_node_users[n] = copy.copy(n.users)

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


"""
The FlowGraph is a dataflow graph for a fx.GraphModule.
The nodes are fx.Nodes and the edges represent the producers (inputs) and
consumers (outputs) of each operation.

The FlowGraph is invalidated if the underlying GraphModule is modified.
It can be regenerated at any time by calling the `build` method.

TODO: turn getitems into "reader views"?

TODO: might be able to use Node.all_input_nodes + Node.users instead (doesn't work with inplace)
"""
class FlowGraph:

    def __init__(self, gm: torch.fx.GraphModule):
        self.module = gm
        self.build()

    def add_edge(self, src: torch.fx.GraphModule, dst: torch.fx.GraphModule):
        if not src in self.succs:
            self.succs[src] = set()
        if not dst in self.preds:
            self.preds[dst] = set()

        self.succs[src].add(dst)
        self.preds[dst].add(src)

    """
    Construct the FlowGraph.
    """
    def build(self):
        self.succs = dict()
        self.preds = dict()
        self.outputs = [n for n in self.module.graph.nodes if n.op == 'output']
        self.inputs = [
            n for n in self.module.graph.nodes if n.op == 'placeholder'
        ]
        visited = set()
        q = self.outputs

        #print(f"Graph:\n{graph_print_tabular(self.module.graph,'users',lambda n: list(n.users.keys()))}")

        self.all_renamed_input_nodes, self.all_renamed_node_users = gather_all_input_nodes(self.module.graph.nodes, True)
        self.all_input_nodes, self.all_node_users = gather_all_input_nodes(self.module.graph.nodes, False)

        while len(q) > 0:
            n = q.pop()
            if n in visited:
                continue

            visited.add(n)
            for input in self.all_renamed_input_nodes[n]:
                self.add_edge(input, n)
                q.append(input)

    """
    The underlying GraphModule inputs.
    """
    def inputs(self) -> List[torch.fx.Node]:
        return self.inputs

    """
    The underlying GraphModule outputs.
    """
    def outputs(self) -> List[torch.fx.Node]:
        return self.outputs

    def successors(self, n: torch.fx.Node) -> Set[torch.fx.Node]:
        return self.succs[n] if n in self.succs else set()

    def predecessors(self, n: torch.fx.Node) -> Set[torch.fx.Node]:
        return self.preds[n] if n in self.preds else set()

    def topo_sort(self):
        # TBD
        return

        order = []
        in_degree = dict()
        worklist: collections.deque = collections.deque()

        self.all_renamed_input_nodes, self.all_renamed_node_users = gather_all_input_nodes(self.module.graph.nodes, True)
        self.all_input_nodes, self.all_node_users = gather_all_input_nodes(self.module.graph.nodes, False)

        new_g = torch.fx.Graph(self.module, self.module.graph._tracer_cls, self.module.graph._tracer_extras)
        new_g._codegen = self.module.graph._codegen
        env: Dict[torch.fx.Node, torch.fx.Node] = {}

        for n in self.module.graph.nodes:
            count = len(self.all_renamed_input_nodes[n])
            in_degree[n] = count
            if count == 0:
                worklist.append(n)
        while len(worklist) > 0:
            n = worklist.popleft()
            env[n] = new_g.node_copy(n) #, lambda x: env[x])

            for u in self.all_renamed_node_users[n]:
                in_degree[u] = in_degree[u] - 1
                if in_degree[u] == 0:
                    worklist.append(u)

        assert len(new_g.nodes) == len(self.module.graph.nodes), f"cycle found: ({new_g.nodes}) != ({self.module.graph.nodes})"

        self.module.graph = new_g

        print(f"topo'd graph\n {graph_print_tabular(new_g)}")

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
            res = res + f"{n} ({n.op}) {oc}\n  preds={str(self.predecessors(n))}\n  succs={str(self.successors(n))}\n{cc}\n"
        return res


class SubGraph:

    def __init__(self,
                 fg: FlowGraph,
                 nodes: Optional[List[torch.fx.Node]] = None):
        self.module = fg.module
        self.inputs = []
        self.outputs = []
        self.all_input_nodes = fg.all_input_nodes
        self.all_node_users = fg.all_node_users
        self.all_renamed_input_nodes = fg.all_renamed_input_nodes
        self.all_renamed_node_users = fg.all_renamed_node_users
        self.build(nodes)

    def in_subgraph(self, n: torch.fx.Node) -> bool:
        return n in self.nodes

    def collect_inputs_outputs(
            self) -> Tuple[List[torch.fx.Node], List[torch.fx.Node]]:
        inputs = []
        outputs = []

        all_input_nodes, all_node_users = self.all_input_nodes, self.all_node_users

        for n in self.nodes:
            new_inputs = [
                inp for inp in all_input_nodes[n] if not self.in_subgraph(inp)
            ]
            for inp in new_inputs:
                if inp not in inputs:
                    inputs.append(inp)

            if any([user for user in all_node_users[n] if not self.in_subgraph(user)
                    ]) and n not in outputs:
                outputs.append(n)

        return inputs, outputs

    def topo_sort(self):
        order = []
        in_degree = dict()
        worklist: collections.deque = collections.deque()

        all_input_nodes, all_node_users = self.all_renamed_input_nodes, self.all_renamed_node_users

        for n in self.nodes:
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
        assert len(order) == len(self.nodes), f"cycle found: ({order}) != ({self.nodes})"

        self.nodes = order

    def build(self, nodes: Optional[List[torch.fx.Node]]):
        self.nodes = nodes
        self.topo_sort()
        self.inputs, self.outputs = self.collect_inputs_outputs()

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

    def erase(self):
        for n in reversed(self.nodes):
            self.module.graph.erase_node(n)

        # TODO: make a function to get/recompute these on demand
        # TODO: be smarter with updating just for deleted/new nodes
        self.all_renamed_input_nodes, self.all_renamed_node_users = gather_all_input_nodes(self.module.graph.nodes, True)
        self.all_input_nodes, self.all_node_users = gather_all_input_nodes(self.module.graph.nodes, False)

        #try:
        #    for n in reversed(self.nodes):
        #        self.module.graph.erase_node(n)
        #except RuntimeError as ex:
        #    print(f"\n{ex}: failed to delete: {list(reversed(self.nodes))}")
        #    print(f"{self.tabular('users', lambda n: list(n.users.keys()))}")

    def tabular(self,
                col: Optional[str] = None,
                col_get: Optional[Callable] = None) -> str:
        try:
            from tabulate import tabulate
        except ImportError:
            print("`print_tabular` relies on the library `tabulate`, "
                  "which could not be found on this machine. Run `pip "
                  "install tabulate` to install the library.")
            raise

        assert (col and col_get) or (not col and not col_get)

        headers = ['opcode', 'name', 'target', 'args', 'kwargs']

        if col_get:
            headers.append(col)

        def mcol_get(x):
            if col_get:
                return [trunc(col_get(x))]
            else:
                return []

        node_specs = [['placeholder*', trunc(n.name), trunc(n.target),
                       tuple(),
                       dict(), *mcol_get(n)] for n in self.inputs]

        node_specs = node_specs + [[n.op, trunc(n.name), trunc(n.target), trunc(n.args), n.kwargs, *mcol_get(n)]
                                   for n in self.nodes]

        node_specs = node_specs + [[
            'output*', 'output*', 'output*',
            (trunc(n), ), dict(), *mcol_get(n),
        ] for n in self.outputs]

        return tabulate(node_specs, headers=headers)


"""
Given a list of cpp and cuda source files, build and load a pytorch extension
module with the given name.  Loaded ops will appear in the torch.ops.{lib_name}
namespace.
"""
def build_extension(lib_name: str,
                    sources: List[str],
                    opt: str = '-O2',
                    verbose: bool = False):
    vllm_root = Path(__file__).parent.parent.parent
    torch.utils.cpp_extension.load(
        name=lib_name,
        sources=sources,
        extra_cflags=[
            opt, f'-DLIBRARY_NAME={lib_name}', f'-I{vllm_root}/csrc'
        ],
        # Note: this is a total hack to get naive C++ fused ops working.
        extra_ldflags=[
            f'{vllm_root}/vllm/_C.abi3.so'
        ],
        verbose=verbose,
        is_python_module=False,
    )


# Turn into functionalization?
def add_uses_for_mutable_inputs(g: torch.fx.Graph):
    uses = dict()
    for n in g.nodes:
        for a in n.args:
            if isinstance(a, torch.fx.Node) and a in uses:
                defs = uses[a]
                for d in defs:
                    print(f"ADD USE {d}, {n}")
                    d.users[n] = None
            elif isinstance(a, tuple) and any(
                [isinstance(aa, torch.fx.Node) and aa in uses for aa in a]):
                for aa in a:
                    defs = uses[aa]
                    for d in defs:
                        print(f"ADD USE {d}, {n}")
                        d.users[n] = None

        if n.op != 'call_function':
            continue
        sigs, schemas = torch.fx.operator_schemas.get_signature_for_torch_op(
            n.target, return_schemas=True)
        if schemas is None or not any([s.is_mutable for s in schemas]):
            continue

        matched_schemas = []
        for candidate_signature, schema in zip(sigs, schemas):
            try:
                candidate_signature.bind(*n.args, **n.kwargs)
                matched_schemas.append((candidate_signature, schema))
            except TypeError as e:
                continue

        if len(matched_schemas) == 0:
            # Did not match any schema. Cannot check for mutation
            continue
        elif len(matched_schemas) == 1:
            _, s = matched_schemas[0]
            if s.is_mutable:
                print(f"MUTABLE SIG {s}")
                for i, a in enumerate(s.arguments):
                    if a.alias_info and a.alias_info.is_write:
                        nth_arg = n.args[i]
                        if isinstance(nth_arg, torch.fx.Node):
                            print(f"  ARG {a.name}, {a.alias_info}, {nth_arg}")
                            if not nth_arg in uses:
                                uses[nth_arg] = set([n])
                            else:
                                uses[nth_arg].add(n)


# Find all in-place functions in the graph and tag them as "impure" so
# fx eliminate_dead_code() and other utilities will not delete them.
def tag_side_effects(g: torch.fx.Graph):
    for n in g.nodes:
        if n.op != 'call_function':
            continue

        sigs, schemas = torch.fx.operator_schemas.get_signature_for_torch_op(
            n.target, return_schemas=True)
        if schemas is None or not any([s.is_mutable for s in schemas]):
            continue

        matched_schemas = []
        for candidate_signature, schema in zip(sigs, schemas):
            try:
                candidate_signature.bind(*n.args, **n.kwargs)
                matched_schemas.append((candidate_signature, schema))
            except TypeError as e:
                continue

        if len(matched_schemas) == 0:
            # Did not match any schema. Cannot check for mutation
            continue
        elif len(matched_schemas) == 1:
            _, s = matched_schemas[0]
            if s.is_mutable:
                torch.fx.node.has_side_effect(n.target)
                logger.debug(
                    f"Found mutable or inplace signature {n.target}: {s}")
