###############################################################################
#
# Utils
#
###############################################################################

import collections
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

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

"""
Similar to torch.fx.Graph.print_tabular except it returns a string and
allows the addition of extra columns.
"""
def graph_print_tabular(
    g: torch.fx.Graph,
    col: Optional[str] = None,
    col_get: Optional[Callable] = None
) -> str:
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
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs, col_get(n)]
                      for n in g.nodes]
    else:
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs]
                      for n in g.nodes]

    return tabulate(node_specs, headers=headers)


"""
Get the name of the function being called in a 'call_function' op.
"""
def node_function_target(node: torch.fx.Node) -> str:
    assert node.op == 'call_function'
    return get_node_target(None, node)


"""
Return a string representation of the type of the given argument.  This is
used for name mangling.
"""
def argument_type_str(arg: torch.fx.node.Argument):
    if isinstance(arg, torch.fx.Node):
        return str(extract_node_type(arg))
    elif isinstance(arg, torch.Tensor):
        return str(arg.dtype)
    elif isinstance(arg, torch.dtype):
        return str(arg)
    elif (isinstance(arg, str) or
          isinstance(arg, int) or
          isinstance(arg, float) or
          isinstance(arg, bool)):
        return type(arg).__name__
    elif (isinstance(arg, types.EllipsisType) or
          isinstance(arg, types.NoneType)):
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
        types = [argument_type_str(arg).replace("torch.","") for arg in n.args]
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

    def call_module(
            self,
            target: torch.fx.node.Target,
            args: Tuple[torch.fx.node.Argument, ...],
            kwargs: Dict[str, Any]
    ) -> Any:
        # Problem here with multiple call sites and different args,
        # for now set to None if there are multiple callers.
        # Could check for "compatible" inputs and allow.
        if target in self.module_args:
            self.module_args[target] = (None, None)
        else:
            self.module_args[target] = (args, kwargs)

        return super().call_module(target, args, kwargs)


"""
The FlowGraph is a dataflow graph for a fx.GraphModule.
The nodes are fx.Nodes and the edges represent the producers (inputs) and
consumers (outputs) of each operation.

The FlowGraph is invalidated if the underlying GraphModule is modified.
It can be regenerated at any time by calling the `build` method.

TODO: turn getitems into "reader views"?

TODO: might be able to use Node.all_input_nodes + Node.users instead
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
        self.inputs = [n for n in self.module.graph.nodes if n.op == 'placeholder']
        visited = set()
        q = self.outputs

        while len(q) > 0:
            n = q.pop()
            if n in visited:
                continue

            visited.add(n)
            for input in n.all_input_nodes:
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


class SubGraph:
    def __init__(self, gm: torch.fx.GraphModule, nodes: Optional[List[torch.fx.Node]] = None):
        self.module = gm
        self.build(nodes)

    def in_subgraph(self, n: torch.fx.Node) -> bool:
        return n in self.nodes

    def collect_inputs_outputs(self) -> Tuple[List[torch.fx.Node], List[torch.fx.Node]]:
        inputs = []
        outputs = []

        for n in self.nodes:
            new_inputs = [inp for inp in n.all_input_nodes if not self.in_subgraph(inp)]
            for inp in new_inputs:
                if inp not in inputs:
                    inputs.append(inp)

            if any([user for user in n.users if not self.in_subgraph(user)]) and n not in outputs:
                outputs.append(n)

        return inputs, outputs

    def topo_sort(self):
        order = []
        in_degree = dict()
        worklist: collections.deque = collections.deque()

        for n in self.nodes:
            count = len([inp for inp in n.all_input_nodes if self.in_subgraph(inp)])
            in_degree[n] = count
            if count == 0:
                worklist.append(n)

        while len(worklist) > 0:
            n = worklist.popleft()
            order.append(n)

            for u in n.users:
                if not self.in_subgraph(u):
                    continue
                in_degree[u] = in_degree[u] - 1
                if in_degree[u] == 0:
                    worklist.append(u)

        # Check for cycles (should not be any).
        assert len(order) == len(self.nodes)

        self.nodes = order

    def build(self, nodes: Optional[List[torch.fx.Node]]):
        self.nodes = nodes
        self.topo_sort()
        self.inputs, self.outputs = self.collect_inputs_outputs()

    def first_in_graph(self):
        first = None
        for n in self.module.graph.nodes:
            if not first and n.next in self.nodes:
                first = n
                break
        return first

    def erase(self):
        for n in reversed(self.nodes):
            self.module.graph.erase_node(n)

    def tabular(self) -> str:
        try:
            from tabulate import tabulate
        except ImportError:
            print("`print_tabular` relies on the library `tabulate`, "
                  "which could not be found on this machine. Run `pip "
                  "install tabulate` to install the library.")
            raise

        headers = ['opcode', 'name', 'target', 'args', 'kwargs']

        node_specs = [['placeholder*', n.name, n.target, tuple(), dict()]
                      for n in self.inputs]

        node_specs = node_specs + [[n.op, n.name, n.target, n.args, n.kwargs]
                      for n in self.nodes]

        node_specs = node_specs + [['output*', 'output*', 'output*', (n,), dict()]
                      for n in self.outputs]

        return tabulate(node_specs, headers=headers)


"""
Given a list of cpp and cuda source files, build and load a pytorch extension
module with the given name.  Loaded ops will appear in the torch.ops.{lib_name}
namespace.
"""
def build_extension(
    lib_name: str,
    sources: List[str],
    opt: str = '-O2',
    verbose: bool = False
):
    vllm_root = Path(__file__).parent.parent.parent
    torch.utils.cpp_extension.load(
        name=lib_name,
        sources=sources,
        extra_cflags=[opt, f'-DLIBRARY_NAME={lib_name}', f'-I{vllm_root}/csrc'],
        verbose=verbose,
        is_python_module=False,
    )
