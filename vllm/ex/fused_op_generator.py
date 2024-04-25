###############################################################################
#
# Fused operation generator
# Converts fx graph nodes into CUDA/C++ code
#
###############################################################################

import logging
import tempfile
import torch
import types

from .utils import extract_node_type, compose, build_extension, mangle_name, argument_type_str, node_function_target

from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set
from vllm.logger import init_logger, _default_handler

logger = init_logger(__name__)


"""
An exception used to indicate a failure in the fusion or fused op generation process.
Should be recoverable, i.e. can fall back to the non-fused version of graph.
"""
class FusionFail(Exception):
    pass


"""
The FusedOpGenerator is a class that is responsible for generating a fused CUDA/C++
operation for sequences of gx graph nodes.

Use of the class is broken up into two steps: 'make_fused_op' and 'build_ops'.

'make_fused_op' generates the C++/CUDA code for a list of fx graph nodes and
adds it to the current "library".  Multiple fused operations can be added to
the current "library" using 'make_fused_op'.  'make_fused_op' returns the
mangled name of the new fused operation.

In order to build the code for a "library", the 'build_ops' function is called.
'build_ops' invokes the compiler on all the code for the current "library".
This code will be associated with a torch library named 'fused_ops{N}' (where
N is the id provided by the FusedOpGenerator class). 'build_ops' returns a map
of mangled op name to Callable.  Each call to 'build_ops' will generate a new
torch library.

All generated code will appear in the 'torch.ops.fused_ops{N}' python namespace.

The reason this is broken up into two steps is that that the compilation costs
can be reduced by compiling all the operations in a single file rather than
multiple small files. Although currently, each operation is compiled individually
as it is fused.

In addition to generating the CUDA/C++ code, the FusedOpGenerator also needs
to register the schemas and meta functions for torch.compile support.
"""
class FusedOpGenerator:
    # A unique id to prevent multiple instances of the same torch library being created.
    N = 0

    def __init__(self):
        # base filename for generated code.
        self.filename = "fused_"
        self.reset_fused_op()
        self.N = FusedOpGenerator.N

    # Set up the pre-amble for each "library" and clear out the callables dict.
    def reset_fused_op(self):
        # 'callables' is a map of mangled function name to a tuple of:
        # (fully qualified op name, type schema, meta function)
        # The type schema and meta function fields are used for registering
        # with pytorch for torch.compile support.
        self.callables = dict()
        self.fused_op = []
        self.fused_op.append(f'#include <torch/extension.h>')
        self.fused_op.append(f'#include <iostream>')
        self.fused_op.append('#define _operator_add(a, b) ((a) + (b))')
        self.fused_op.append('#define _operator_mul(a, b) ((a) * (b))')
        self.fused_op.append('#define TORCH_LIBRARY_EXPAND(name, mod) TORCH_LIBRARY(name, mod)')
        self.fused_op.append('#define TORCH_LIBRARY_IMPL_EXPAND(name, k, mod) TORCH_LIBRARY_IMPL(name, k, mod)')
        self.fused_op.append('inline torch::Tensor to_tensor(py::object obj) { return THPVariable_Unpack(obj.release().ptr()); }')
        self.fused_op.append('namespace py = pybind11;')

    # 'mangle' a python name so it can be used with C++.
    def mangle(self, s: str, rep: str = '_P_') -> str:
        s = s.replace('.', rep)
        return s

    # Perform any renames on python symbols so they can be compiled with C++
    def rename(self, s: str) -> str:
        if s == 'torch._C._nn.linear':
            # Hack to map vllm ops to the standard torch linear op.
            return 'torch::nn::functional::linear'
        elif s.startswith('torch.ops.vllm.'):
            return s.replace('torch.ops.vllm.', '')
        else:
            return s.replace("_operator.", "_operator_")

    # Translate getitem arguments to C++/Python ABI
    def convert_getitem_arg(self, arg: torch.fx.node.Argument) -> str:
        if isinstance(arg, types.EllipsisType):
            return "py::ellipsis()"
        elif isinstance(arg, types.NoneType):
            return "std::nullopt" #"Py_None"
        elif isinstance(arg, int):
            return f"{arg}"
        elif isinstance(arg, slice):
            start = self.convert_getitem_arg(arg.start)
            stop = self.convert_getitem_arg(arg.stop)
            step = self.convert_getitem_arg(arg.step)
            return f"py::slice({start}, {stop}, {step})"
        else:
            raise FusionFail(f"unsupported getitem indexing arg: {arg}.")

    def convert_slice_args(self, args: Tuple[torch.fx.node.Argument]) -> str:
        idx = 1 if isinstance(args[0], types.EllipsisType) else 0

        assert isinstance(args[idx], slice)

        start = self.convert_getitem_arg(args[idx].start)
        stop = self.convert_getitem_arg(args[idx].stop)
        step = f", {self.convert_getitem_arg(args[idx].step)}" if args[idx].step is not None else ""
        return f"{idx}, {start}, {stop}{step}"

    # Detect simple 2d slices along a single dimension.
    def is_simple_slice(self, arg: torch.fx.node.Argument) -> str:
        if not isinstance(arg, tuple) or len(arg) != 2:
            return False
        if not ((isinstance(arg[0], types.EllipsisType) and isinstance(arg[1], slice)) or
                (isinstance(arg[1], types.EllipsisType) and isinstance(arg[0], slice))):
            return False
        return True

    def translate_getitem(self, n: torch.fx.Node) -> str:
        # Note: The default (non-simple slice) implementation causes extra copies of the
        # input to be made.
        call_str = ''
        tensor = n.args[0]
        idx = n.args[1]

        assert isinstance(idx, tuple)

        if self.is_simple_slice(idx):
            call_str = f"  auto {self.mangle(n.name, '_')} = {self.mangle(str(tensor), '_')}.slice({self.convert_slice_args(idx)});"
        else:
            call_str = f"  auto {self.mangle(n.name, '_')} = to_tensor("
            call_str = call_str + f"py::reinterpret_steal<py::object>(THPVariable_Wrap({self.mangle(str(tensor), '_')}))["
            call_str = call_str + f"py::make_tuple("

            sep = ""
            for idx_arg in idx:
                call_str = call_str + sep + self.convert_getitem_arg(idx_arg)
                sep = ", "

            call_str = call_str + ")]);"

        return call_str

    def last_uses(self, nodes: List[torch.fx.Node]) -> Dict[torch.fx.Node, List[torch.fx.Node]]:
        """
        Collect last uses locations for all variables in the set of nodes being fused.
        """
        node_to_last_use : Dict[torch.fx.Node, torch.fx.Node] = {}
        user_to_last_uses : Dict[torch.fx.Node, List[torch.fx.Node]] = {}

        def register_last_uses(n : torch.fx.Node, user : torch.fx.Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            torch.fx.node.map_arg(node.args, lambda n: register_last_uses(n, node))
            torch.fx.node.map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        return user_to_last_uses


    def delete_unused_values(self, user_to_last_uses, user : torch.fx.Node) -> str:
        """
        Delete values after their last use. This ensures that values that are
        not used in the remainder of the code are freed and the memory usage
        of the code is optimal.
        """
        if user.op == 'placeholder':
            return
        if user.op == 'output':
            return
        nodes_to_delete = user_to_last_uses.get(user, [])
        to_delete_str = ''
        sep = '  '
        for n in nodes_to_delete:
            to_delete_str = to_delete_str + sep + self.mangle(n.name, '_') + " = torch::Tensor();"
        return to_delete_str


    #
    # Generate naive C++/CUDA code for a stack of fused ops.
    #
    # TODO:
    # - use cutlass
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
        kwargs: Dict[str, Dict[str, torch.fx.node.Argument]]
    ) -> torch.fx.node.Target:
        fns = [n.target for n in nodes]
        logger.debug(f"make_fused_op: {fns}")

        # assume unary output for now
        assert len(outputs) == 1

        fn_names = [self.rename(node_function_target(n)) for n in nodes]

        user_to_last_uses = self.last_uses(inputs + nodes)

        op = f"{mangle_name(nodes)}_fused"

        cxx_arg_sig = ''
        sep = ''
        for i, n in enumerate(inputs):
            #cxx_arg_sig = cxx_arg_sig + sep + f"torch::Tensor const& {n}"
            cxx_arg_sig = cxx_arg_sig + sep + f"torch::Tensor& {n}"
            sep = ", "

        arg_sig = self.generate_op_schema(inputs, outputs, nodes, kwargs)

        oc = '{'
        cc = '}'

        self.fused_op.append(f'torch::Tensor {op}({cxx_arg_sig})')
        self.fused_op.append('{')
        self.fused_op.append('  pybind11::gil_scoped_acquire gil_lock;')

        # TODO: this debug logging/print is a hack, remove it later.
        if _default_handler.level == logging.DEBUG:
            self.fused_op.append(f'  std::cout << "Executing: {op}" << std::endl;')

        for n, fn in zip(nodes, fn_names):
            comment_str = f"  // ({', '.join([argument_type_str(inp) for inp in n.args])}) -> {str(extract_node_type(n))}"

            if fn == '_operator_getitem':
                call_str = self.translate_getitem(n)
                assert kwargs.get(n.name) is None or len(kwargs.get(n.name)) == 0
            else:
                call_str = f"  auto {self.mangle(n.name, '_')} = {self.mangle(fn, '::')}("
                sep =''
                for inp in n.args:
                    # bit of a hack for optional/empty tensor arguments
                    if inp is None:
                        call_str = call_str + sep + "torch::Tensor()"
                    elif isinstance(inp, tuple):
                        call_str = call_str + sep + "{" + ','.join([str(t) for t in inp]) + "}"
                    else:
                        call_str = call_str + sep + self.mangle(str(inp), '_')
                    sep = ', '
                n_kwargs = kwargs.get(n.name)
                # TODO
                if n_kwargs:
                    raise FusionFailed("kwargs nyi")
                call_str = call_str + ');'

            self.fused_op.append(comment_str)
            self.fused_op.append(call_str)
            self.fused_op.append(self.delete_unused_values(user_to_last_uses, n))

        self.fused_op.append(f"  // {str(extract_node_type(outputs[0]))}")
        self.fused_op.append(f"  return {self.mangle(outputs[0].name, '_')};")

        self.fused_op.append('}')
        self.fused_op.append(f'TORCH_LIBRARY_EXPAND(fused_ops{self.N}, m) {oc} m.def("{op}{arg_sig}"); {cc}')
        self.fused_op.append(f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, CPU, m) {oc} m.impl("{op}", &{op}); {cc}')
        self.fused_op.append(f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, CUDA, m) {oc} m.impl("{op}", &{op}); {cc}')
        # For now, generate the meta function via 'generate_meta_function' even though this version is probably
        # more robust.
        #self.fused_op.append(f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, Meta, m) {oc} m.impl("{op}", &{op}); {cc}')

        self.callables[op] = (
            f"torch.ops.fused_ops{self.N}.{op}",
            arg_sig,
            self.generate_meta_function(inputs, outputs, nodes, kwargs)
        )

        return op

    # The schema is (mostly) derivable from types annotations on input/output nodes.
    def generate_op_schema(
        self,
        inputs: List[torch.fx.Node],
        outputs: List[torch.fx.Node],
        nodes: List[torch.fx.Node],
        kwargs: Dict[str, Dict[str, torch.fx.node.Argument]]
    ):
        sep = f"("
        arg_sig = ""
        for i, n in enumerate(inputs):
            # TODO: the Tensor default here is sketchy
            arg_type = self.mangle(n.type.__name__ if n.type is not None else "Tensor", '::')
            arg_name = self.mangle(n.name, '_')
            arg_sig = arg_sig + sep + f"{arg_type} {arg_name}"
            sep = ", "
        arg_sig = arg_sig + ") -> "

        sep = "(" if len(outputs) != 1 else ""

        for i, n in enumerate(outputs):
            # TODO: the Tensor default here is sketchy
            arg_type = self.mangle(n.type.__name__ if n.type is not None else "Tensor", '::')
            arg_sig = arg_sig + sep + arg_type
            sep = ", "

        if len(outputs) != 1:
            arg_sig = arg_sig + ")"

        return arg_sig

    # Generate a meta function for a fused op by composing the individual
    # operations.
    # TODO: this only works when the fused op is a nice "funnel", i.e. the first
    # op takes all the inputs and chains the rest to subsequent ops.
    # See functools.partial and inspect.signature().parameters
    def generate_meta_function(
        self,
        inputs: List[torch.fx.Node],
        outputs: List[torch.fx.Node],
        nodes: List[torch.fx.Node],
        kwargs: Dict[str, Dict[str, torch.fx.node.Argument]]
    ) -> Callable:
        fns = [n.target for n in nodes]
        return compose(*fns)

    # Regsiter schema for the given 'op' in the given 'lib'.
    def register_op_schema(self, library: str, op: str, sig: str):
        op = self.mangle(op, '::').replace("torch::ops::", "")
        logger.debug(f"Registering schema for {op}: {sig}")
        torch.library.define(f"{op}", sig)

    # Regsiter meta function the given 'op' in the given 'lib'.
    def register_meta_function(self, library: str, op: str, meta_fn: Callable):
        # See also: torch.library.impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1)
        op = self.mangle(op, '::').replace("torch::ops::", "")
        logger.debug(f"Registering meta function for {op}: {str(meta_fn)}")
        torch.library.impl(f"{op}", "Meta", func=meta_fn)

    # Compile the code for the current "library".
    # Note: this could fail and throw a FusionFail exception.
    def build_ops(self) -> Dict[torch.fx.node.Target, Callable]:
        # prevent multiple libraries with the same name
        FusedOpGenerator.N = FusedOpGenerator.N + 1

        try:
            op_lib = f"fused_ops{self.N}"

            # Note: we could register the schema here but there
            # is no way to unregister it if the build fails, so
            # we let the C++ code register it for now.
            #
            #for k, v in self.callables.items():
            #    self.register_op_schema(op_lib, v[0], v[1])

            with tempfile.NamedTemporaryFile(
                    prefix=self.filename,
                    suffix=".cpp",
                    mode='w',
                    delete=False, # TODO: True
            ) as out:
                logger.info(f"generating code to: {out.name}")
                for l in self.fused_op:
                    out.write(l)
                    out.write('\n')
                out.close()
                build_extension(op_lib, str(out.name))
                logger.info(f"code generation success: {out.name}")

            self.N = FusedOpGenerator.N

            callables = dict()

            for k, v in self.callables.items():
                # TODO: there has to be a better way than eval?
                fn = eval(v[0])
                self.register_meta_function(op_lib, v[0], v[2])
                callables[k] = fn

            self.reset_fused_op()

            return callables

        except Exception as ex:
            raise FusionFail(ex)
