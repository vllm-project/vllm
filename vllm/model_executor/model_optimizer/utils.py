###############################################################################
#
# Utils
#
###############################################################################

import copy
import torch

try:
    from tabulate import tabulate  # type: ignore
    have_tabulate = True
except ImportError:
    have_tabulate = False

from collections import OrderedDict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from .silu_mul_quant import silu_mul_quant_name
from .fused_rms_quant import rms_norm_quant_name, rms_norm_quant_name_2

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


def trunc(x, lim=40) -> Any:
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


def mangle_name(nodes: List[torch.fx.Node],
                rep: str = "_P_",
                include_constants=True) -> str:
    """
    Generate a mangled name from a list of call_function nodes.
    The mangled name includes the names of all the operators and their types.
    """
    name = ""
    sep = ""
    for n in nodes:
        fn = node_function_target(n)
        types = [
            argument_type_str(arg, include_constants).replace("torch.", "")
            for arg in n.args
        ]
        ktypes = [
            f"K_{name}_{argument_type_str(kwarg, True).replace('torch.', '')}"
            for name, kwarg in n.kwargs.items()
        ]
        name = name + sep + f"{fn}_{'_'.join(types)}{'_'.join(ktypes)}"
        sep = "_"

    return name.replace(".", rep)


SIMPLIFIED_NAMES: Dict[str, str] = dict()
SILU_MUL_QUANT_NAMES: List[str] = [
    # "torch_P_ops_P__C_P_cutlass_scaled_mm_float16_float8_e4m3fn_float8_e4m3fn_float32_float32_None_torch_P_empty_T_int_8192_int_14336K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_8192_int_14336K_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_silu_and_mul_float16_float16_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32",
    # "torch_P_empty_T_int_8192_int_28672K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_8192_int_14336K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_8192_int_14336K_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_cutlass_scaled_mm_float16_float8_e4m3fn_float8_e4m3fn_float32_float32_None_torch_P_ops_P__C_P_silu_and_mul_float16_float16_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32_fused",
    # "torch_P_empty_T_int_3_int_28672K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_3_int_14336K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_3_int_14336K_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_cutlass_scaled_mm_float16_float8_e4m3fn_float8_e4m3fn_float32_float32_None_torch_P_ops_P__C_P_silu_and_mul_float16_float16_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32_fused",
    # "torch_P_empty_T_int_256_int_28672K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_256_int_14336K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_256_int_14336K_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_cutlass_scaled_mm_float16_float8_e4m3fn_float8_e4m3fn_float32_float32_None_torch_P_ops_P__C_P_silu_and_mul_float16_float16_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32_fused",
    # "torch_P_empty_T_int_1_int_28672K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_1_int_14336K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_1_int_14336K_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_cutlass_scaled_mm_float16_float8_e4m3fn_float8_e4m3fn_float32_float32_None_torch_P_ops_P__C_P_silu_and_mul_float16_float16_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32_fused",
]

RMS_NORM_QUANT_NAMES: List[str] = [
    # "torch_P_empty_like_float16_torch_P_empty_T_int_8192_int_4096K_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_rms_norm_float16_float16_float16_float_1e_05_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32_fused",
    # "torch_P_empty_like_float16_torch_P_empty_T_int_1_int_4096K_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_rms_norm_float16_float16_float16_float_1e_05_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32_fused",
]
RMS_NORM_QUANT_2_NAMES: List[str] = [
    # "torch_P_empty_like_float16_torch_P_ops_P__C_P_rms_norm_float16_float16_float16_float_1e_05_size_float16_torch_P_empty_SizeK_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32_size_float8_e4m3fn_int_0_torch_P_empty_T_SymInt_int_6144K_dtype_float16_K_device_D_cuda_0_fused",
    # "torch_P_empty_like_float16_torch_P_ops_P__C_P_rms_norm_float16_float16_float16_float_1e_05_size_float16_torch_P_empty_SizeK_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32_size_float8_e4m3fn_int_0_torch_P_empty_T_int_int_6144K_dtype_float16_K_device_D_cuda_0_fused",
]


# This is not really necessary but helps debugging
def simplify_mangled_name(name: str) -> str:
    if name in SILU_MUL_QUANT_NAMES:
        simple_name = silu_mul_quant_name
        SIMPLIFIED_NAMES[name] = simple_name
    if name in RMS_NORM_QUANT_NAMES:
        simple_name = rms_norm_quant_name
        SIMPLIFIED_NAMES[name] = simple_name
    if name in RMS_NORM_QUANT_2_NAMES:
        simple_name = rms_norm_quant_name_2
        SIMPLIFIED_NAMES[name] = simple_name
    if name not in SIMPLIFIED_NAMES:
        print(f"NAME: {name}")
        simple_name = f"mangled_{len(SIMPLIFIED_NAMES)}"
        SIMPLIFIED_NAMES[name] = simple_name
    return SIMPLIFIED_NAMES[name]


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


def gather_all_input_nodes_old(
    mod: torch.fx.GraphModule,
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

    nodes: List[torch.fx.Node] = list(mod.graph.nodes)

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
        todel = []
        for user in all_node_users[n]:
            if user in renames:
                todel.append(user)

        for user in todel:
            del all_node_users[n][user]
            all_node_users[n][user] = None

    #
    # populate maps up front
    #
    for n in nodes:
        all_input_nodes[n] = n.all_input_nodes
        all_node_users[n] = OrderedDict(n.users)

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

    # This is a hack!!!!
    for n in nodes:
        for p in all_input_nodes[n]:
            if n not in all_node_users[p]:
                #print(f"{n} not in users of {p}!")
                all_node_users[p][n] = None
        for s in all_node_users[n]:
            if n not in all_input_nodes[s]:
                #print(f"{n} not in inputs of {s}!")
                all_input_nodes[s].append(n)

    # Make sure everything is sorted
    for n in nodes:
        all_input_nodes[n] = list(sorted(all_input_nodes[n]))
        all_node_users[n] = OrderedDict(
            sorted(all_node_users[n].items(), key=lambda kv: kv[0].name))

    return all_input_nodes, all_node_users


def gather_all_input_nodes(
    mod: torch.fx.GraphModule
) -> Tuple[OrderedDict[torch.fx.Node, List[torch.fx.Node]], OrderedDict[
        torch.fx.Node, OrderedDict[torch.fx.Node, None]], Dict[
            torch.fx.Node, Dict[torch.fx.Node, torch.fx.Node]]]:
    """
    Collect all def/use information for each node in 'nodes'.  This is different
    than node.all_input_nodes and node.users since it handles in-place
    functions.  It is assumed that any mutable input in an in-place function is
    also an output.

    Note: this will include Node kwargs
    """

    # Assure that the graph is toposorted
    #mod.graph.lint()

    all_input_nodes: OrderedDict[torch.fx.Node,
                                 List[torch.fx.Node]] = OrderedDict()
    all_node_users: OrderedDict[torch.fx.Node,
                                OrderedDict[torch.fx.Node,
                                            None]] = OrderedDict()
    renames: Dict[torch.fx.Node, torch.fx.Node] = dict()
    all_renames: Dict[torch.fx.Node, Dict[torch.fx.Node,
                                          torch.fx.Node]] = dict()

    def process_arg(arg: torch.fx.node.Argument, fn):
        if isinstance(arg, tuple):
            for sub_arg in arg:
                process_arg(sub_arg, fn)
            return

        if not isinstance(arg, torch.fx.Node):
            return

        fn(arg)

    def delete_rename(arg):
        if arg in renames:
            del renames[arg]

    def add_rename(arg):
        renames[arg] = n

    def add_renamed_user(n, arg):
        if arg in renames:
            renamed_arg = renames[arg]
            all_node_users[renamed_arg][n] = None
            if n not in all_renames:
                all_renames[n] = dict()
            all_renames[n][renamed_arg] = arg

    #
    # process each node
    #
    # just do outputs then make symmetric inputs
    # for nullary targets, tie to mutable args?  or add a mapping?
    for n in mod.graph.nodes:
        all_input_nodes[n] = list()
        all_node_users[n] = OrderedDict(n.users)
        mutable_args = mutable_function_args(n)

        # Add this node as a user for all mutable args.
        for arg in n.all_input_nodes:
            process_arg(arg, lambda arg: add_renamed_user(n, arg))

        # erase_renames(mutable_args)
        for marg in mutable_args:
            arg = nth_arg_or_kwarg(n, marg)
            process_arg(arg, delete_rename)

        # all mutable args need to be renamed as the lhs of current node
        for marg in mutable_args:
            arg = nth_arg_or_kwarg(n, marg)
            process_arg(arg, add_rename)

    # Make inputs symmetric with users
    for n in mod.graph.nodes:
        for s in all_node_users[n]:
            all_input_nodes[s].append(n)

    logger.debug(
        dump_inputs_users(mod.graph.nodes, all_input_nodes, all_node_users))
    #print(dump_inputs_users(mod.graph.nodes, all_input_nodes, all_node_users))

    # Make sure everything is sorted
    for n in mod.graph.nodes:
        all_input_nodes[n] = list(sorted(all_input_nodes[n]))
        all_node_users[n] = OrderedDict(
            sorted(all_node_users[n].items(), key=lambda kv: kv[0].name))

    return all_input_nodes, all_node_users, all_renames


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

    def build(self):
        """
        Construct the FlowGraph.
        """
        self.outputs = [n for n in self.module.graph.nodes if n.op == 'output']
        self.inputs = [
            n for n in self.module.graph.nodes if n.op == 'placeholder'
        ]

        self.all_renamed_input_nodes, self.all_renamed_node_users, self.renames = (
            gather_all_input_nodes(self.module.graph.nodes))

        #        self.renames = {}
        #        self.all_renamed_input_nodes, self.all_renamed_node_users = (
        #            gather_all_input_nodes_old(self.module.graph.nodes, True))
        #
        #        self.all_input_nodes, self.all_node_users = gather_all_input_nodes_old(
        #            self.module.graph.nodes, False)

        self.preds: Dict[torch.fx.Node, OrderedDict[torch.fx.Node, None]] = {}
        self.succs: Dict[torch.fx.Node, OrderedDict[torch.fx.Node, None]] = {}
        for n in self.module.graph.nodes:
            self.succs[n] = OrderedDict(self.all_renamed_node_users[n])
            self.preds[n] = OrderedDict.fromkeys(
                self.all_renamed_input_nodes[n])

    def successors(self, n: torch.fx.Node):  # -> Set[torch.fx.Node]:
        return self.succs[n].keys()

    def predecessors(self, n: torch.fx.Node):  # -> Set[torch.fx.Node]:
        return self.preds[n].keys()

    def dfs_visit(self,
                  fn: Callable,
                  start: Optional[List[torch.fx.Node]] = None):
        q: deque = deque(start if start else self.inputs)
        visited = set()
        while len(q) > 0:
            n = q.pop()
            if n in visited:
                continue
            visited.add(n)
            fn(n)
            q.extend(self.successors(n))

    def paths_to_nodes(self, starts: List[torch.fx.Node],
                       ends: List[torch.fx.Node]) -> List[List[torch.fx.Node]]:
        visited = set()
        path = []
        paths = []

        def dfs(n: torch.fx.Node):
            visited.add(n)
            path.append(n)

            if n in ends:
                paths.append(copy.copy(path))
            else:
                for p in self.successors(n):
                    if p not in visited:
                        dfs(p)

            path.pop()

        for n in starts:
            if n not in visited:
                dfs(n)

        return paths

    def to_dot(self,
               name: str = "my_graph",
               hilite: Optional[List[List[torch.fx.Node]]] = None,
               orig: bool = False) -> str:
        import pydot

        def make_label(n: torch.fx.Node) -> str:
            if is_simple_call(n):
                return f"{n.name} = {node_function_target(n)}(...)"
            else:
                return n.name

        palette = [
            "red", "green", "blue", "yellow", "orange", "purple", "pink",
            "gray"
        ]
        colors = []
        node_colors = dict()
        hilite = hilite if hilite else []
        for i, h in enumerate(hilite):
            for n in self.module.graph.nodes:
                if n in h:
                    node_colors[n] = i
            colors.append(palette[i % len(palette)])

        graph = pydot.Dot(name, graph_type="digraph")
        for n in self.module.graph.nodes:
            color = colors[node_colors[n]] if n in node_colors else "black"
            style = "filled" if n in node_colors else ""
            graph.add_node(
                pydot.Node(n.name,
                           shape="box",
                           color=color,
                           style=style,
                           label=make_label(n)))

        for n in self.module.graph.nodes:
            succs = self.successors(n) if not orig else n.users.keys()
            for s in succs:
                graph.add_edge(pydot.Edge(n.name, s.name))

        return str(graph)

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
        self.fg = fg
        self.module = fg.module
        self.inputs: List[torch.fx.Node] = []
        self.outputs: List[torch.fx.Node] = []
        self.nodes: List[torch.fx.Node] = []
        #self.all_input_nodes = fg.all_input_nodes # DELETE
        #self.all_node_users = fg.all_node_users   # DELETE
        self.all_renamed_input_nodes = fg.all_renamed_input_nodes
        self.all_renamed_node_users = fg.all_renamed_node_users
        self.renames = fg.renames
        self.build(nodes)

    def in_subgraph(self, n: torch.fx.Node) -> bool:
        return n in self.nodes

    def _collect_inputs_outputs(
            self) -> Tuple[List[torch.fx.Node], List[torch.fx.Node]]:
        inputs = []
        outputs = []

        # DELETE
        all_input_nodes, all_node_users = (self.all_renamed_input_nodes,
                                           self.all_renamed_node_users)

        #def rename(n, p):
        #    if n in self.renames and p in self.renames[n]:
        #        print(f"renames[{n}] = {self.renames[n]}, p = {p}")
        #        return self.renames[n][p]
        #    return p

        for n in self.nodes:
            new_inputs = [
                #rename(n, inp) for inp in self.fg.predecessors(n) if not self.in_subgraph(rename(n, inp))
                #inp for inp in self.fg.predecessors(n) if not self.in_subgraph(inp)
                inp for inp in n.all_input_nodes if not self.in_subgraph(inp)
            ]
            for inp in new_inputs:
                if inp not in inputs:
                    inputs.append(inp)

            if any([
                    #user for user in self.fg.successors(n) if not self.in_subgraph(n)
                    user
                    for user in all_node_users[n] if not self.in_subgraph(user)
            ]) and n not in outputs:
                #print(f"ADD OUTPUT {n} {n.users}")
                if len(n.users) > 0:
                    outputs.append(n)

        return inputs, outputs

    def topo_sort(self):
        order = []
        in_degree = dict()
        worklist: deque = deque()

        # XXXXXX remove debugging code
        debug = False and any([n.name == 'output_160' for n in self.nodes])

        all_input_nodes, all_node_users = (self.all_renamed_input_nodes,
                                           self.all_renamed_node_users)

        for n in sorted(self.nodes, key=lambda n: n.name):
            # This count is not right....
            # users/inputs not symmetric for nullary nodes
            count = len(
                [inp for inp in all_input_nodes[n] if self.in_subgraph(inp)])
            in_degree[n] = count
            if count == 0:
                worklist.append(n)

        if debug:
            print(f"count={count}")
            print(f"in_degree={in_degree}")
            print(f"worklist={worklist}")
            nl = "  \n"
            print(
                f"inputs={nl}{nl.join([str((x,y)) for x,y in all_input_nodes.items() if self.in_subgraph(x)])}"
            )
            print(
                f"outputs={nl}{nl.join([str((x,y)) for x,y in all_node_users.items() if self.in_subgraph(x)])}"
            )
            print(
                f"in_subgraph={[str((n.name, self.in_subgraph(n))) for n in self.nodes]}"
            )

        while len(worklist) > 0:
            n = worklist.popleft()
            order.append(n)
            if debug:
                print(f"n, worklist={n}, {worklist}, {order}, {in_degree}")

            for u in all_node_users[n]:
                if not self.in_subgraph(u):
                    continue
                in_degree[u] = in_degree[u] - 1
                if debug:
                    print(f"   dec {u} = {in_degree[u]}")
                if in_degree[u] == 0:
                    worklist.append(u)

        # Check for cycles (should not be any).
        assert len(order) == len(
            self.nodes), f"cycle found: ({order}) != ({self.nodes})"

        self.nodes = order

        if False and debug:
            print(f"order = {order}")
            exit(-1)

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

    def last_in_subgraph(self):
        last = None
        for n in reversed(self.module.graph.nodes):
            if not last and n in self.nodes:
                last = n
                break
        return last

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
        self.all_renamed_input_nodes, self.all_renamed_node_users, self.renames = (
            gather_all_input_nodes(self.module.graph.nodes))

#        self.all_renamed_input_nodes, self.all_renamed_node_users = (
#            gather_all_input_nodes_old(self.module.graph.nodes, True))
#
#        self.all_input_nodes, self.all_node_users = gather_all_input_nodes_old(
#            self.module.graph.nodes, False)

    def erase(self):
        """
        Erase all the nodes in the SubGraph.
        """
        for n in reversed(self.nodes):
            self.module.graph.erase_node(n)

        # TODO: be smarter with updating just for deleted/new nodes
        self._refresh_def_use()

    def to_dot(self, name: str = "") -> str:
        import textwrap

        import pydot

        graph = pydot.Dot(f"sub_graph{name}", graph_type="digraph")
        for n in self.nodes:
            graph.add_node(
                pydot.Node(n.name,
                           shape="box",
                           label=textwrap.fill(n.format_node(), width=30)))

        for n in self.nodes:
            for p in self.fg.predecessors(n):
                if not self.in_subgraph(p):
                    graph.add_edge(pydot.Edge(f"EXTERNAL_{p}", n.name))

            for s in self.fg.successors(n):
                if self.in_subgraph(s):
                    graph.add_edge(pydot.Edge(n.name, s.name))
                else:
                    graph.add_edge(pydot.Edge(n.name, f"EXTERNAL_{s}"))

        return str(graph)

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
            assert arg.start is None
            return (arg.stop, )
        else:
            assert arg.start is not None
            assert arg.stop is None
            return (arg.start, )
    elif isinstance(arg, (str, int, float, bool)):
        return (arg, )
    else:
        return ()


def arg_swap(arg: torch.fx.node.Argument,
             inputs: Dict[str, torch.fx.node.Argument]) -> str:
    try:
        # if this arg is in the list of external args and it's constant,
        # substitute it in. Otherwise just stringify the value
        # TODO: This is a little strange for duplicated values. It will
        # work, but will end up using the first instance of that value
        # in the argument list.
        key = list(inputs.keys())[list(inputs.values()).index(arg)]
        if "const" in key:
            return key
        else:
            return str(arg)
    except ValueError:
        return str(arg)
