import copy
import functools
import torch
import torch.fx as fx

from .ex_builder import build_extension
from torch._dynamo import register_backend, lookup_backend
from torch.fx.passes.operator_support import create_op_support
from torch.fx.passes.split_module import split_module
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.tools_common import get_node_target
from torch.fx.passes.shape_prop import ShapeProp
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx.experimental.schema_type_annotation import AnnotateTypesWithSchema

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set

import traceback

###############################################################################
#
# Utils
#
###############################################################################


def extract_type(arg: torch.fx.node.Argument):
    if isinstance(arg, torch.Tensor):
        return arg.dtype
    else:
        return None


def extract_node_type(n: torch.fx.Node):
    if 'tensor_meta' in n.meta:
        return n.meta['tensor_meta'].dtype
    return None


###############################################################################
#
# dataflow graph
#
###############################################################################

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

    def inputs(self) -> List[torch.fx.Node]:
        return self.inputs

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


###############################################################################
#
# Partitioning
#
###############################################################################

# TODO: make this smarter
def is_node_supported(submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
    return node.op == 'call_function' and (get_node_target(submodules, node) == '_operator.add' or
                                           get_node_target(submodules, node) == '_operator.mul' or
                                           get_node_target(submodules, node) == 'torch.matmul' or
                                           get_node_target(submodules, node) == 'torch.relu' or
                                           get_node_target(submodules, node) == 'torch.nn.functional.silu' or
                                           get_node_target(submodules, node) == 'torch._C._nn.linear' or
                                           get_node_target(submodules, node) == 'torch.ops.vllm.silu_and_mul')


# See: torch.fx.passes.infra.partitioner
def partition_graph(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Tuple[torch.fx.GraphModule, List[Partition]]:
    support = create_op_support(is_node_supported)
    p = CapabilityBasedPartitioner(
        gm,
        support,
        allows_single_node_partition=True, #False,
        non_compute_ops=None,
        allowed_single_node_partition_ops=None
    )
    parts = p.propose_partitions()
    return p.fuse_partitions(parts), parts


###############################################################################
#
# Quantization
#
###############################################################################

# torch._inductor.fx_passes.quantization
def add_quantization(mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # TODO fill this in later
    return mod


###############################################################################
#
# Fusion
#
###############################################################################

class FusedOpGenerator:
    N = 0

    def __init__(self):
        self.filename = "fused.cpp"
        self.callables = dict()
        self.reset_fused_op()
        self.N = FusedOpGenerator.N

    def reset_fused_op(self):
        self.fused_op = []
        self.fused_op.append(f'#include <torch/extension.h>')
        self.fused_op.append(f'#include <iostream>')
        self.fused_op.append('#define _torch_add(a, b) ((a) + (b))')
        self.fused_op.append('#define _torch_mul(a, b) ((a) * (b))')
        self.fused_op.append('#define TORCH_LIBRARY_EXPAND(name, mod) TORCH_LIBRARY(name, mod)')
        self.fused_op.append('#define TORCH_LIBRARY_IMPL_EXPAND(name, k, mod) TORCH_LIBRARY_IMPL(name, k, mod)')

    # This should take types into account. (what else?)
    def mangle(self, s: str, rep: str = '_P_') -> str:
        s = s.replace('.', rep)
        return s

    def rename(self, s: str) -> str:
        if s == '_operator.add':
            return '_torch_add'
        elif s == '_operator.mul':
            return '_torch_mul'
        else:
            return s

    #
    # Generate some (dumb) C++/CUDA code for a stack of fused ops.
    #
    # TODO:
    # - use cutlass
    # - include types in mangled names
    # - manage generated code (no duplicates)
    # - handle kwargs
    #
    # See https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?pli=1#heading=h.rmcmku6fe6ug
    #
    # Note: node.meta['tensor_meta'] will have shape and dtype fields
    #
    def make_fused_op(
        self,
        inputs: List[torch.fx.Node],
        outputs: List[torch.fx.Node],
        nodes: List[torch.fx.Node],
        # make this a list of Dict?
        kwargs: Dict[str, torch.fx.node.Argument]
    ) -> torch.fx.node.Target:
        fns = [n.target for n in nodes]
        print(f"MAKE_FUSED_OP {fns}")
        # See functools.partial for applying args/kwargs

        # assume unary output for now
        assert len(outputs) == 1

        submodules = dict(nodes[0].graph.owning_module.named_modules())

        fn_names = [self.rename(get_node_target(submodules, n)) for n in nodes]

        op = self.mangle("_".join(fn_names)) + '_fused'
        cxx_arg_sig = ''
        arg_sig = ''
        sep = ''
        input_type_map = {}
        for i, n in enumerate(inputs):
            cxx_arg_sig = cxx_arg_sig + sep + f"torch::Tensor const& {n}"
            arg_sig = arg_sig + sep + f"Tensor {n}"
            sep = ", "
            input_type_map[n.name] = extract_node_type(n)

        for n in nodes:
            input_type_map[n.name] = extract_node_type(n)

        def input_type(arg: torch.fx.node.Argument):
            if isinstance(arg, torch.fx.Node):
                return input_type_map[arg.name]
            else:
                return None

        oc = '{'
        cc = '}'

        self.fused_op.append(f'torch::Tensor {op}({cxx_arg_sig})')
        self.fused_op.append('{')
        self.fused_op.append('std::cout << "GOT HERE" << std::endl;')

        for n, fn in zip(nodes, fn_names):
            com_str = f"  // ({', '.join([str(input_type(inp)) for inp in n.args])}) -> {str(extract_node_type(n))}"
            call_str = f"  auto const& {self.mangle(n.name, '_')} = {self.mangle(fn, '::')}("
            sep =''
            for inp in n.args:
                call_str = call_str + sep + self.mangle(str(inp), '_')
                sep = ', '
            call_str = call_str + ');'
            self.fused_op.append(com_str)
            self.fused_op.append(call_str)
        self.fused_op.append(f"  // {str(extract_node_type(outputs[0]))}")
        self.fused_op.append(f"  return {self.mangle(outputs[0].args[0].name, '_')};")

        self.fused_op.append('}')
        self.fused_op.append(f'TORCH_LIBRARY_EXPAND(fused_ops{self.N}, m) {oc} m.def("{op}({arg_sig}) -> Tensor"); {cc}')
        self.fused_op.append(f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, CPU, m) {oc} m.impl("{op}", &{op}); {cc}')
        # TODO: make sure this does the "right thing"
        self.fused_op.append(f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, Meta, m) {oc} m.impl("{op}", &{op}); {cc}')

        self.callables[op] = f"torch.ops.fused_ops{self.N}.{op}"

        return op

    def build_ops(self) -> Dict[torch.fx.node.Target, Callable]:
        with open(self.filename, "w") as out:
            for l in self.fused_op:
                out.write(l)
                out.write('\n')

        build_extension(f"fused_ops{self.N}", self.filename)

        for k, v in self.callables.items():
            # there has to be a better way than eval?
            fn = eval(v)
            print(f'{self.callables[k]} = {fn}')
            self.callables[k] = fn

            # This works
            #a = torch.tensor([[1.0, -1.0],[2.0, 3.0]])
            #b = torch.tensor([[2.0, -2.0],[3.0, 4.0]])
            #c = fn(a, b)
            #print(f'FN ANSWER: {c}')

        print(f"CALLABLES {self.callables}")

        callables = self.callables

        self.reset_fused_op()
        self.callables = dict()

        # prevent multiple libraries with the same name
        FusedOpGenerator.N = FusedOpGenerator.N + 1
        self.N = FusedOpGenerator.N

        return callables


#
# Fuse all the nodes in a subgraph into a single node
#
def fuse_graph_nodes(
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule
) -> torch.fx.GraphModule:
    first = None

    outputs = [n for n in mod.graph.nodes if n.op == 'output']
    inputs = [n for n in mod.graph.nodes if n.op == 'placeholder']

    print(f"input_meta = {[n.meta for n in inputs]}")

    # for now
    assert len(outputs) == 1

    nodes_to_erase = []

    kwargs = None
    for n in mod.graph.nodes:
        if n.op == 'placeholder':
            first = n

        if n.op != 'call_function':
            continue

        if n.kwargs:
            if not kwargs:
                kwargs = n.kwargs
            else:
                # TODO: assert no duplicates
                kwargs = {**kwargs, **n.kwargs}

        nodes_to_erase.append(n)

    fn_key = fgen.make_fused_op(inputs, outputs, nodes_to_erase, kwargs)

    fn_dict = fgen.build_ops()

    assert fn_key in fn_dict

    fn = fn_dict[fn_key]
    #print(f"fused fn = {fn}, {type(fn)}, {isinstance(fn, torch.nn.Module)}, {str(fn)}")

    mod.graph.inserting_after(first)

    # TODO: no kwargs for now
    assert kwargs == None or len(kwargs) == 0

    cf = mod.graph.call_function(fn, args=tuple(inputs), kwargs=kwargs)

    # Note: we do not update the meta info for cf here.  It should
    # not be required after transformation anyway.

    # which way is best?  the else seems more general
    if False:
        outputs[0].prev.replace_all_uses_with(cf)
    else:
        mod.graph.inserting_after(cf)
        mod.graph.output(cf, type_expr=torch.Tensor)

        for o in outputs:
            print(f"ERASE {o}")
            mod.graph.erase_node(o)

    print(f"fuse mod {mod.print_readable(False)}")
    print(f"cf {cf.name} {cf.format_node()}")

    nodes_to_erase.reverse()
    for n in nodes_to_erase:
        print(f"ERASE {n}")
        mod.graph.erase_node(n)

    # TODO: see node.replace_all_uses_with(new_node)

    # Do this here or in caller?
    #mod.recompile()

    return mod


# TODO: add more stuff
def is_fusable(a: torch.fx.Node) -> bool:
    pointwise = ['_operator.add', '_operator.mul', 'torch.relu', 'torch.nn.functional.silu']
    if a.op == 'call_function':
        submodules = dict(a.graph.owning_module.named_modules())
        target = get_node_target(submodules, a)
        return target in pointwise
    return False


# TODO: add more stuff
def is_compute(a: torch.fx.Node) -> bool:
    if a.op != 'call_function':
        return false
    submodules = dict(a.graph.owning_module.named_modules())
    return (get_node_target(submodules, a) == 'torch.matmul' or
            get_node_target(submodules, a) == 'torch._C._nn.linear')


def is_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    return is_fusable(a) and is_fusable(b)

def is_compute_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    return (is_fusable(a) or is_compute(a)) and is_fusable(b)


def is_fused_subgraph(mod: torch.fx.GraphModule) -> bool:
    fg = FlowGraph(mod)
    saw_call = False
    for n in mod.graph.nodes:
        if n.op == 'call_module' or n.op == 'call_method':
            return False

        if n.op != 'call_function':
            continue

        pred = is_fusable_pair if saw_call else is_compute_fusable_pair
        saw_call = True

        if not all([pred(n, s) for s in fg.successors(n) if s.op == 'call_function']):
            return False

    return True


# 1. create Partition objects from sequences of fusable nodes
# 2. use fuse_partitions to recreate the graph
# torch._inductor.fx_passes.group_batch_fusion
def pointwise_fusion(
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    fuse_inputs: bool = False,
    fuse_with_compute=True
) -> torch.fx.GraphModule:
    # find all groups of nodes that can be fused and assign to
    # unique partition id, i.e. map_node

    fg = FlowGraph(mod)

    node_map = {}
    partition = 0

    def map_node(n: torch.fx.Node) -> int:
        return node_map[n]

    # assumption, graph.nodes are in topo order
    mod.graph.lint()

    print("start fusion")

    # create partition groups
    for n in mod.graph.nodes:
        if n.op != 'call_function':
            node_map[n] = 0
            continue

        if not all([is_fusable_pair(n, s) for s in fg.successors(n)]):
            if not n in node_map:
                node_map[n] = 0
            continue

        # don't support anything with kwargs for now
        if n.kwargs and len(n.kwargs) > 0:
            node_map[n] = 0
            continue

        if n not in node_map:
            partition = partition + 1
            node_map[n] = partition

        for s in fg.successors(n):
            node_map[s] = node_map[n]

    print(f"node_map = {node_map}")

    def same_partition(nodes: Set[torch.fx.Node]) -> bool:
        if len(nodes) > 0:
            part = next(iter(nodes))
            return all([node_map[n] == part for n in nodes])
        return False


    def only_pointwise(partition: int) -> bool:
        nodes = [n for n, p in node_map if p == parition]
        return all([is_fusable(n) and not is_compute(n) for n in nodes])

    if fuse_with_compute:
        for n in mod.graph.nodes:
            if n.op != 'call_function':
                continue

            if fuse_inputs:
                nodes = fg.predecessors(n)
            else:
                nodes = fg.successors(n)

            if not is_compute(n) or not same_partition(nodes):
                continue

            fuse_part = next(iter(nodes))

            if only_pointwise(fuse_part):
                node_map[n] = fuse_part

    print(f"final node_map = {node_map}")

    assert(all([n in node_map for n in mod.graph.nodes]))

    qualname_map=dict()

    print(f"mod {mod.print_readable(False)}")

    # create submodules for each fusable set of nodes
    new_mod = split_module(
        mod,
        mod,
        map_node,
        qualname_map,
        keep_original_order=False, #True
    )

    mig = ModuleInputGenerator(new_mod)
    mig.propagate(*example_inputs)

    # replace the fused submodules with new modules
    for cname, cm in new_mod.named_children():
        if is_fused_subgraph(cm):
            module_inputs = mig.module_args[cname][0]
            ShapeProp(cm).propagate(*module_inputs)

            print(f"FUSING GRAPH NODES {cname}")
            cm.graph.print_tabular()
            #graph_print_tabular(cm.graph)
            fuse_graph_nodes(fgen, cm)
            print(f"CM {cname}: {cm}")
            cm.recompile()

    print(f"new_mod {new_mod.print_readable(False)}")
    print(f"new mod {new_mod.graph.print_tabular()}")

    # Do this here or in caller?
    #new_mod.recompile()

    return new_mod


###############################################################################
#
# Backend
#
###############################################################################

# torch._inductor.fx_passes.joint_graph.joint_graph_passes
def optimize(
    fgen: FusedOpGenerator,
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> torch.fx.GraphModule:
    mod = add_quantization(mod)
    mod = pointwise_fusion(fgen, mod, example_inputs)
    # the inductor should(?) handle this.
    # mod = inline_submodules(mod)
    return mod


# names should be unique, so this is ok
def node_in_module(n: torch.fx.Node, m: torch.fx.GraphModule) -> bool:
    return n.name in [nn.name for nn in m.graph.nodes]


def module_in_partitions(parts: List[Partition], m: torch.fx.GraphModule) -> bool:
    for p in parts:
        if node_in_module(next(iter(p.nodes)), m):
            return True
    return False


def backend_compile(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    backend: str ='inductor'
) -> Callable:
    try:
        backend = lookup_backend(backend)
        print(f"attempting {backend}")
        backend_compiled = backend(gm, example_inputs)
        if backend_compiled is not None:
            print(f"{backend} COMPILED!")
            return backend_compiled
    except Exception as ex:
        print(f"EX '{ex}'")
        tb = ex.__traceback__
        print(f"EX TRACE")
        traceback.print_tb(tb)
        pass

    return gm.forward


# Combine with ShapeProp somehow?
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
        # TODO: problem here with multiple call sites and different args,
        # for now set to None if there are multiple callers.
        # Could check for "compatible" inputs and allow.
        if target in self.module_args:
            self.module_args[target] = (None, None)
        else:
            self.module_args[target] = (args, kwargs)

        # arg_types = [extract_type(arg) for arg in args]

        ret = super().call_module(target, args, kwargs)

	# print(f"arg_types = {arg_types}, ret = {extract_type(ret)}")

        return ret


# why doesn't this work?
def graph_print_tabular(g: torch.fx.Graph):
    try:
        from tabulate import tabulate
    except ImportError:
        print("`print_tabular` relies on the library `tabulate`, "
              "which could not be found on this machine. Run `pip "
              "install tabulate` to install the library.")
        raise

    node_specs = [[n.op, n.name, n.target, n.args, n.kwargs, n.meta]
                  for n in g.nodes]
    print(tabulate(node_specs,
                headers=['opcode', 'name', 'target', 'args', 'kwargs', 'meta']))


# See https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50
# maybe useful https://github.com/huggingface/optimum/blob/main/optimum/fx/optimization/transformations.py
# TODO: see if transforms can work here

class backend_class:
    def __init__(self, final='inductor'):
        self.final = final

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
        print(f"ORIGINAL {gm.graph}")
        #    fg = FlowGraph(gm)
        #    fg.visit(lambda n: print(n))

        # Must make a copy so that inductor backend doesn't choke.
        gm = copy.copy(gm)

        #gm = torch.fx.Tracer().trace(gm)
        #gm = torch.fx.symbolic_trace(gm)
        #gm = AnnotateTypesWithSchema(gm).transform()

        # TODO: see schema_type_annotation.py/AnnotateTypesWithSchema

        part_gm, parts = partition_graph(gm, example_inputs)

        #ShapeProp(part_gm).propagate(*example_inputs)

        print(f"BEFORE forward: {part_gm.forward}")

        print(f"part_gm: {part_gm}")
        #graph_print_tabular(part_gm.graph)
        print(f"parts: {parts}")
        print(f"children: {[(cname, cm.print_readable(False)) for cname, cm in part_gm.named_children()]}")

        # get the current FakeTensorMode (there should be one since we are in a backend)
        fake_mode = torch._guards.detect_fake_mode()

        # Is this ok?  probably should save/restore at least
        fake_mode.allow_non_fake_inputs = True

        #example_inputs = [fake_mode.from_tensor(input) for input in example_inputs]

        # There should be an existing fake_mode but double check
        if not fake_mode:
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        print(f"fake mode = {fake_mode}")

        # use FakeTensorProp-like class to get example inputs for submodules
        # static_shapes can be applied here
        mig = ModuleInputGenerator(part_gm, fake_mode)
        mig.propagate(*example_inputs)

        print(f"mod args = {mig.module_args}")

        # TODO: store this in the root module state dictionary so that code for
        # all sub-modules is shared?
        fgen = FusedOpGenerator()

        mods_to_compile = []

        for name, m in part_gm.named_modules():
            if module_in_partitions(parts, m):
                assert name in mig.module_args
                module_inputs = mig.module_args[name][0]

                # TODO: make this smarter
                if not module_inputs:
                    print(f"SKIPPING {name} FOR NOW (multiple callers): {m.print_readable(False)}")
                    continue

                print(f"OPTIMIZE! {name}: {m.print_readable(False)}")
                m = optimize(fgen, m, module_inputs)
                setattr(part_gm, name, m)

                print(f"POST OPTIMIZE! {name}: {m.print_readable(False)}")

                # TODO: don't really need to recompile if nothing happened.
                m.recompile()

                print(f"mod inputs {module_inputs}")
                #print(f"fake mode={torch._guards.detect_fake_mode(module_inputs)}")
                if self.final != None:
                    m.forward = backend_compile(m, module_inputs, backend=self.final)

        part_gm.recompile()
        print(f"FULL FINAL GRAPH: {part_gm.print_readable(False)}")
        #return backend_compile(part_gm, example_inputs)
        return part_gm.forward


def backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    return backend_class()(gm, example_inputs)


def make_backend(final: str = 'inductor') -> backend_class:
    return backend_class(final)
