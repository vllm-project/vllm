###############################################################################
#
# Fused operation generator
# Naively converts fx graph nodes into CUDA/C++ code
#
###############################################################################

import tempfile
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import torch

from vllm.logger import init_logger

from .fused_op_generator import FusedOpGenerator, FusionFail
from .fused_op_generator_utils import (arg_schema_type, build_extension,
                                       generate_op_schema)
from .utils import (arg_swap, argument_type_str, extract_node_type,
                    node_function_target)

logger = init_logger(__name__)

EllipsisType = type(...)
NoneType = type(None)


def generate_cxx_sig(n: torch.fx.Node) -> str:
    # TODO: derive signature from schema
    #sch = torch.fx.operator_schemas.get_signature_for_torch_op(n.target)
    #print(sch)

    # Hardcode a bunch of signatures for known ops.
    trg = node_function_target(n)
    if trg == "torch.ops._C.cutlass_scaled_mm":
        return ("void(torch::Tensor& , torch::Tensor const&, "
                "torch::Tensor const&, torch::Tensor const&, "
                "torch::Tensor const&, "
                "c10::optional<torch::Tensor> const& bias)")
    elif trg == "torch.ops._C.dynamic_scaled_fp8_quant":
        return ("void(torch::Tensor& out, torch::Tensor const& input, "
                "torch::Tensor& scale)")
    elif (trg == "torch.ops._C.static_scaled_fp8_quant"):
        return ("void(torch::Tensor& out, torch::Tensor const& input, "
                "torch::Tensor const& scale)")
    elif trg == "torch.ops._C.dynamic_scaled_int8_quant":
        return ("void(torch::Tensor& out, torch::Tensor const& input, "
                "torch::Tensor& scale)")
    elif trg == "torch.ops._C.static_scaled_int8_quant":
        return ("void(torch::Tensor& out, torch::Tensor const& input, "
                "torch::Tensor const& scale)")
    elif trg == "torch.ops._C.fused_add_rms_norm":
        return ("void(torch::Tensor& input, torch::Tensor& residual, "
                "torch::Tensor& weight, double epsilon)")
    elif trg == "torch.ops._C.rms_norm":
        return ("void(torch::Tensor& out, torch::Tensor& input, "
                "torch::Tensor& weight, double epsilon)")
    elif trg == "torch.ops._C.silu_and_mul":
        return "void(torch::Tensor& out, torch::Tensor& input)"
    else:
        raise FusionFail(f"no C++ signature for: {trg}")


class NaiveFusedOpGenerator(FusedOpGenerator):
    """
    The NaiveFusedOpGenerator is a class that is responsible for generating a
    fused CUDA/C++ operation for sequences of gx graph nodes.

    Use of the class is broken up into two steps: 'make_fused_op' and
    'build_op'.

    'make_fused_op' generates the C++/CUDA code for a list of fx graph nodes and
    adds it to the current "library".  Multiple fused operations can be added to
    the current "library" using 'make_fused_op'.  'make_fused_op' then calls
    'build_op' to generate the code for the new operation.

    In order to build the code for a "library", the 'build_op' function is
    called. 'build_op' invokes the compiler on all the code for the current
    "library".  This code will be associated with a torch library named
    'fused_ops{N}' (where N is the id provided by the NaiveFusedOpGenerator
    class). 'build_op' returns a Callable.  Each call to 'build_op' will
    generate a new torch library.

    All generated code will appear in the 'torch.ops.fused_ops{N}' python
    namespace.

    In addition to generating the CUDA/C++ code, the NaiveFusedOpGenerator also
    needs to register the schemas and meta functions for torch.compile support.
    """

    # A unique id to prevent multiple instances of the same torch library being
    # created.
    N = 0

    def __init__(self):
        super().__init__()
        # base filename for generated code.
        self.filename = "fused_"
        self.reset_fused_op()
        self.N = NaiveFusedOpGenerator.N

    def reset_fused_op(self):
        """
        Set up the preamble for each "library".
        """

        self.fused_op = []
        self.fused_op.append('#include <torch/extension.h>')
        #self.fused_op.append(f'#include <iostream>')
        self.fused_op.append('#define _operator_add(a, b) ((a) + (b))')
        self.fused_op.append('#define _operator_mul(a, b) ((a) * (b))')
        self.fused_op.append(('#define TORCH_LIBRARY_EXPAND(name, mod) '
                              'TORCH_LIBRARY(name, mod)'))
        self.fused_op.append(
            ('#define TORCH_LIBRARY_IMPL_EXPAND(name, k, mod) '
             'TORCH_LIBRARY_IMPL(name, k, mod)'))
        self.fused_op.append(
            ('inline torch::Tensor to_tensor(py::object obj) '
             '{ return THPVariable_Unpack(obj.release().ptr()); }'))
        #self.fused_op.append('namespace py = pybind11;')

    def sanitize(self, s: str, rep: str = '_') -> str:
        """
        'sanitize' a python name so it can be used with C++.
        """
        s = s.replace('.', rep)
        return s

    def rename(self, s: str) -> str:
        """
        Perform any renames on python symbols so they can be compiled with C++
        """
        if s == 'torch._C._nn.linear':
            # Hack to map vllm ops to the standard torch linear op.
            return 'torch::nn::functional::linear'
        elif s.startswith('torch.ops._C.'):
            return s.replace('torch.ops._C.', '')
        elif s == 'torch::int8':
            return 'torch::kInt8'
        elif s == 'torch::uint8':
            return 'torch::kUInt8'
        elif s == 'torch::float16':
            return 'torch::kFloat16'
        elif s == 'torch::float32':
            return 'torch::kFloat32'
        elif s == 'torch::float8_e4m3fn':
            return 'torch::kFloat8_e4m3fn'
        elif s in [
                'cpu', 'cuda', 'hip', 'fpga', 'ort', 'xla', 'mps', 'xpu',
                'hpu', 've', 'ipu', 'mtia'
        ]:
            return f'torch::k{s.upper()}'
        elif s in ['meta', 'vulkan', 'metal', 'lazy']:
            return f'torch::k{s[0].upper()}{s[1:]}'
        else:
            return s.replace("_operator.", "_operator_")

    def convert_getitem_arg(self, arg: torch.fx.node.Argument) -> str:
        """
        Translate getitem arguments to C++/Python ABI
        """
        if isinstance(arg, EllipsisType):
            return "py::ellipsis()"
        elif isinstance(arg, NoneType):
            return "std::nullopt"
        elif isinstance(arg, int):
            return f"{arg}"
        elif isinstance(arg, slice):
            start = self.convert_getitem_arg(arg.start)
            stop = self.convert_getitem_arg(arg.stop)
            step = self.convert_getitem_arg(arg.step)
            return f"py::slice({start}, {stop}, {step})"
        else:
            raise FusionFail(f"unsupported getitem indexing arg: {arg}.")

    def convert_slice_args(self, args: Tuple[torch.fx.node.Argument],
                           inputs: Dict[str, torch.fx.node.Argument]) -> str:
        idx = 1 if isinstance(args[0], EllipsisType) else 0

        assert isinstance(args[idx], slice)

        start = self.convert_getitem_arg(args[idx].start)
        stop = self.convert_getitem_arg(args[idx].stop)
        step = f", {self.convert_getitem_arg(args[idx].step)}" if args[
            idx].step is not None else ""
        if args[idx].start is not None:
            start = arg_swap(args[idx].start, inputs)
        elif args[idx].stop is not None:
            stop = arg_swap(args[idx].stop, inputs)
        return f"{idx}, {start}, {stop}{step}"

    def is_simple_slice(self, arg: torch.fx.node.Argument) -> bool:
        """
        Detect simple 2d slices along a single dimension.
        """
        if not isinstance(arg, tuple) or len(arg) != 2:
            return False
        if not (
            (isinstance(arg[0], EllipsisType) and isinstance(arg[1], slice)) or
            (isinstance(arg[1], EllipsisType) and isinstance(arg[0], slice))):
            return False
        return True

    def translate_getitem(self, n: torch.fx.Node,
                          inputs: Dict[str, torch.fx.node.Argument]) -> str:
        # Note: The default (non-simple slice) implementation causes extra
        # copies of the input to be made.
        call_str = ''
        tensor = n.args[0]
        idx = n.args[1]

        #assert isinstance(idx, tuple) or self.is_simple_slice(idx)
        if not (isinstance(idx, tuple) or self.is_simple_slice(idx)
                or isinstance(idx, int)):
            raise FusionFail(f"unsupported slice: {idx}")

        if self.is_simple_slice(idx):
            call_str = (f"  auto {self.sanitize(n.name)} = "
                        f"{self.sanitize(str(tensor))}.slice("
                        f"{self.convert_slice_args(idx, inputs)});")
        elif isinstance(idx, int):
            call_str = (f"  auto {self.sanitize(n.name)} = "
                        f"{self.sanitize(str(tensor))}[{idx}];")
        else:
            # Note: this code works but requires pybind which we don't want
            # to use.
            call_str = f"  auto {self.sanitize(n.name)} = to_tensor("
            arg = self.sanitize(str(tensor))
            call_str = (call_str +
                        "py::reinterpret_steal<py::object>(THPVariable_Wrap(" +
                        arg + "}))[")
            call_str = call_str + "py::make_tuple("

            sep = ""
            for idx_arg in idx:
                call_str = call_str + sep + self.convert_getitem_arg(idx_arg)
                sep = ", "

            call_str = call_str + ")];"

        return call_str

    def last_uses(
        self, nodes: List[torch.fx.node.Argument]
    ) -> Dict[torch.fx.Node, List[torch.fx.Node]]:
        """
        Collect last uses locations for all variables in the set of nodes being
        fused.
        """
        node_to_last_use: Dict[torch.fx.Node, torch.fx.Node] = {}
        user_to_last_uses: Dict[torch.fx.Node, List[torch.fx.Node]] = {}

        def register_last_uses(n: torch.fx.Node, user: torch.fx.Node):
            if n not in node_to_last_use and arg_schema_type(n) == 'Tensor':
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            if isinstance(node, torch.fx.Node):
                torch.fx.node.map_arg(
                    node.args,
                    lambda n, node=node: register_last_uses(n, node))
                torch.fx.node.map_arg(
                    node.kwargs,
                    lambda n, node=node: register_last_uses(n, node))

        return user_to_last_uses

    def delete_unused_values(self, user_to_last_uses, user: torch.fx.Node,
                             outputs: List[torch.fx.Node]) -> str:
        """
        Generate code to delete values after their last use. This ensures that
        values that are not used in the remainder of the code are freed and the
        memory usage of the code is as good as the original python code.
        """
        if user.op == 'placeholder':
            return ''
        if user.op == 'output':
            return ''
        nodes_to_delete = user_to_last_uses.get(user, [])
        to_delete_str = ''
        sep = '  '
        for n in nodes_to_delete:
            if n in outputs:
                continue
            to_delete_str = to_delete_str + sep + self.sanitize(
                n.name) + " = torch::Tensor();"
        return to_delete_str

    def make_fused_op(
            self, op: str, inputs: OrderedDict[str, torch.fx.node.Argument],
            outputs: List[torch.fx.Node], nodes: List[torch.fx.Node],
            kwargs: Dict[str, Dict[str, torch.fx.node.Argument]]) -> Callable:
        """
        Generate naive C++/CUDA code for a stack of fused ops.

        TODO
        - handle general kwargs

        See https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?pli=1#heading=h.rmcmku6fe6ug

        Notes:
        - node.meta['tensor_meta'] will have shape and dtype fields
        - Can be called from multiple threads/workers.
        """

        fns = [n.target for n in nodes]
        logger.debug("make_fused_op: %s", fns)

        # assume no input kwargs for now.
        assert len(kwargs) == 0

        fn_names = [self.rename(node_function_target(n)) for n in nodes]

        user_to_last_uses = self.last_uses(list(inputs.values()) + nodes)

        cxx_arg_sig = ''
        sep = ''
        arg_types = [
            f"{arg_schema_type(inp, True)}" for inp in inputs.values()
        ]
        logger.debug("fused op argument types: %s", arg_types)
        for i, name in enumerate(inputs.keys()):
            # Don't use const refs here so inputs can be deleted when no
            # longer needed.
            if arg_types[i] == 'torch::Tensor':
                cxx_arg_sig = (cxx_arg_sig + sep +
                               f"{arg_types[i]}& {inputs[name]}")
            elif arg_types[i] == 'float':
                cxx_arg_sig = cxx_arg_sig + sep + f"double {name}"
            elif arg_types[i] == 'int':
                cxx_arg_sig = cxx_arg_sig + sep + f"int64_t {name}"
            else:
                cxx_arg_sig = cxx_arg_sig + sep + (f"{arg_types[i]} "
                                                   f"const& {inputs[name]}")
            sep = ", "

        arg_sig = generate_op_schema(inputs, outputs, nodes, kwargs)

        oc = '{'
        cc = '}'

        if len(outputs) == 1:
            self.fused_op.append(f'torch::Tensor {op}({cxx_arg_sig})')
        else:
            output_types = ["torch::Tensor" for output in outputs]
            self.fused_op.append(
                f'std::tuple<{", ".join(output_types)}> {op}({cxx_arg_sig})')
        self.fused_op.append('{')

        # pybind only needed for non-simple slices.
        #self.fused_op.append('  pybind11::gil_scoped_acquire gil_lock;')

        # TODO: this debug logging/print is a hack, remove it later.
        #if _default_handler.level == logging.DEBUG:
        #self.fused_op.append(f'  std::cout << "Executing: {op}" << std::endl;')

        # Lookup ops for vllm custom kernels.
        for n in nodes:
            fn = node_function_target(n)
            if fn.startswith("torch.ops._C"):
                fn_name = self.rename(fn)
                cxx_sig = generate_cxx_sig(n)
                init = (f'  static auto {fn_name} = '
                        'torch::Dispatcher::singleton()'
                        f'.findSchemaOrThrow("_C::{fn_name}", "")'
                        f'.typed<{cxx_sig}>();')
                self.fused_op.append(init)

        for n, fn in zip(nodes, fn_names):
            return_type = extract_node_type(n)

            # Total hack
            if n.op == 'call_method':
                return_type = "Size"

            input_types = [argument_type_str(inp) for inp in n.args]
            comment_str = f"  // ({', '.join(input_types)}) -> {return_type}"

            # If the node is a slice
            if fn == '_operator_getitem':
                call_str = self.translate_getitem(n, inputs)
                assert kwargs.get(n.name) is None or len(kwargs[n.name]) == 0
            else:
                if return_type is None:
                    call_str = "  "
                else:
                    call_str = f"  auto {self.sanitize(n.name)} = "
                first_arg = 0
                if n.op == 'call_method':
                    call_str = (call_str +
                                f"{self.sanitize(n.args[0].name, '::')}.")
                    first_arg = 1

                # First check is total hack here
                if fn == 'size' and len(n.args) == 1:
                    call_str = call_str + "sizes("
                elif node_function_target(n).startswith("torch.ops._C"):
                    call_str = call_str + f"{self.sanitize(fn, '::')}.call("
                else:
                    call_str = call_str + f"{self.sanitize(fn, '::')}("

                sep = ''
                for i, inp in enumerate(n.args[first_arg:]):
                    # bit of a hack for optional/empty tensor arguments
                    if inp is None:
                        # {} should work for both default Tensor and
                        # std::optional<Tensor>
                        call_str = call_str + sep + "{}"
                    elif isinstance(inp, (int, float)):
                        call_str = call_str + sep + arg_swap(inp, inputs)
                    elif isinstance(inp, tuple):
                        call_str = call_str + sep + "{" + ', '.join(
                            [arg_swap(t, inputs) for t in inp]) + "}"
                    else:
                        call_str = call_str + sep + self.rename(
                            self.sanitize(str(inp), '::'))
                    sep = ', '

                # Only handle 'empty' kwargs for now
                if n.kwargs:
                    if fn != 'torch.empty' and fn != 'torch.empty_like':
                        raise FusionFail(f"kwargs nyi on {n}, {n.kwargs}")

                    # TODO ['layout', 'memory_format']
                    supported_empty_kwargs = ['dtype', 'device']

                    if not all([(k in supported_empty_kwargs)
                                for k in n.kwargs]):
                        raise FusionFail(
                            f"unsupported kwarg type in {n.kwargs.keys()}")

                    dtype = n.kwargs.get('dtype')
                    device = n.kwargs.get('device')
                    # TBD layout + mem_format
                    #layout = n.kwargs.get('layout')
                    #mem_format = n.kwargs.get('memory_format')

                    call_str = call_str + sep + 'torch::TensorOptions()'

                    if dtype:
                        dtype_arg = self.rename(self.sanitize(
                            str(dtype), "::"))
                        call_str = call_str + f'.dtype({dtype_arg})'

                    if device:
                        if device.index:
                            dev_name = self.rename(device.type)
                            call_str = (call_str +
                                        f'.device(torch::Device({dev_name}, ' +
                                        f'{device.index}))')
                        else:
                            call_str = (call_str +
                                        f'.device({self.rename(device.type)})')

                call_str = call_str + ');'

            self.fused_op.append(comment_str)
            self.fused_op.append(call_str)
            self.fused_op.append(
                self.delete_unused_values(user_to_last_uses, n, outputs))

        output_types = [str(extract_node_type(output)) for output in outputs]
        self.fused_op.append(f"  // {', '.join(output_types)}")
        if len(outputs) == 1:
            self.fused_op.append(f"  return {self.sanitize(outputs[0].name)};")
        else:
            output_strs = [self.sanitize(output.name) for output in outputs]
            self.fused_op.append(f"  return {oc}{', '.join(output_strs)}{cc};")

        self.fused_op.append('}')
        self.fused_op.append((f'TORCH_LIBRARY_EXPAND(fused_ops{self.N}, m) '
                              f'{oc} m.def("{op}{arg_sig}"); {cc}'))
        self.fused_op.append(
            (f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, CPU, m) '
             f'{oc} m.impl("{op}", &{op}); {cc}'))
        self.fused_op.append(
            (f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, CUDA, m) '
             f'{oc} m.impl("{op}", &{op}); {cc}'))

        # For now, generate the meta function via 'generate_meta_function' even
        # though this version is probably more robust.
        self.fused_op.append(
            (f'TORCH_LIBRARY_IMPL_EXPAND(fused_ops{self.N}, Meta, m) '
             f'{oc} m.impl("{op}", &{op}); {cc}'))

        return self.build_op(op, f"torch.ops.fused_ops{self.N}.{op}", arg_sig,
                             lambda x: x)

    def build_op(self, op: str, torch_op_name: str, sig: str,
                 meta_fn: Callable) -> Callable:
        """
        Compile the code for the current "library".
        Note: this could fail and throw a FusionFail exception.
        """
        # prevent multiple libraries with the same name
        NaiveFusedOpGenerator.N = NaiveFusedOpGenerator.N + 1

        try:
            op_lib = f"fused_ops{self.N}"

            # Note: we could register the schema here but there
            # is no way to unregister it if the build fails, so
            # we let the C++ code register it for now.
            #
            # register_op_schema(op_lib, op, sig)

            with tempfile.NamedTemporaryFile(
                    prefix=self.filename,
                    suffix=".cpp",
                    mode='w',
                    delete=False,  # TODO: True to delete tmp files
            ) as out:
                logger.info("generating code to: %s", out.name)
                for line in self.fused_op:
                    out.write(line)
                    out.write('\n')
                out.close()
                build_extension(op_lib, [str(out.name)])
                logger.info("code generation success: %s", out.name)

            self.N = NaiveFusedOpGenerator.N

            # TODO: there has to be a better way than eval?
            fn = eval(torch_op_name)

            # Use C++ generated meta functions for now.
            #register_meta_function(op_lib, torch_op_name, meta_fn)

            self.reset_fused_op()

            return fn

        except Exception as ex:
            self.reset_fused_op()
            raise FusionFail(ex) from ex
