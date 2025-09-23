# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import inspect
from typing import Callable, Optional, TypeVar, Union, overload
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._dynamo.symbolic_convert import InliningInstructionTranslator

from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils import resolve_obj_by_qualname, supports_dynamo

from .monitor import start_monitoring_torch_compile

logger = init_logger(__name__)

IGNORE_COMPILE_KEY = "_ignore_compile_vllm"

_T = TypeVar("_T", bound=type[nn.Module])


def ignore_torch_compile(cls: _T) -> _T:
    """
    A decorator to ignore support_torch_compile decorator
    on the class. This is useful when a parent class has
    a support_torch_compile decorator, but we don't want to
    compile the class `cls` that inherits the parent class.
    This only ignores compiling the forward of the class the
    decorator is applied to. 

    If the parent has ignore_torch_compile but the child has
    support_torch_compile, the child will still be compiled.
    
    If the class has one or more submodules
    that have support_torch_compile decorator applied, compile will
    not be ignored for those submodules.
    """
    setattr(cls, IGNORE_COMPILE_KEY, True)
    return cls


def _should_ignore_torch_compile(cls) -> bool:
    """
    Check if the class should be ignored for torch.compile.
    """
    return getattr(cls, IGNORE_COMPILE_KEY, False)


@overload
def support_torch_compile(
    *,
    enable_if: Optional[Callable[[VllmConfig], bool]] = None,
) -> Callable[[_T], _T]:
    ...


@overload
def support_torch_compile(
    *,
    dynamic_arg_dims: Optional[dict[str, Union[int, list[int]]]],
) -> Callable[[_T], _T]:
    ...


@overload
def support_torch_compile(cls: _T) -> _T:
    ...


def support_torch_compile(
    cls: Optional[_T] = None,
    *,
    dynamic_arg_dims: Optional[dict[str, Union[int, list[int]]]] = None,
    enable_if: Optional[Callable[[VllmConfig], bool]] = None,
) -> Union[Callable[[_T], _T], _T]:
    """
    A decorator to add support for compiling the forward method of a class.

    Usage 1: use directly as a decorator without arguments:

    ```python
    @support_torch_compile
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
            ...
    ```

    Usage 2: use as a decorator with arguments:

    ```python
    @support_torch_compile(dynamic_arg_dims={"x": 0, "y": 0})
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
            ...
    ```

    `dynamic_arg_dims` is a dictionary that maps argument names to the dynamic
    dimensions of the argument. The dynamic dimensions can be either a single
    integer or a list of integers.

    if `dynamic_arg_dims` is `None`, it is inferred from the type annotation
    of the `forward` method, based on the following default rules:

    - if the argument is annotated as `torch.Tensor` or
        `Optional[torch.Tensor]`, the first dimension will be
        marked as dynamic.
    - if the argument is annotated as `IntermediateTensors`, the first
        dimension of all the tensors in the intermediate tensors
        will be marked as dynamic.

    During runtime, when we actually mark dimensions of tensors,
     it depends on the value of arguments:

    - if it is a single integer (can be negative), the corresponding dimension
        of the argument will be marked as dynamic.
    - if it is `None`, ignored.
    - if it is `IntermediateTensors`, all the tensors in the intermediate
        tensors will be marked as dynamic.
    - otherwise, it will raise an error.

    NOTE: if an argument is `None`, it should always be passed as `None` during
    the lifetime of the model, otherwise, it cannot be captured as a single
    computation graph.

    `enable_if` is a function that takes a `VllmConfig` object as input and
    returns a boolean value indicating whether to compile the model or not.
    This is useful if you want to compile the model only when certain
    conditions are met.
    """

    def cls_decorator_helper(cls: _T) -> _T:
        # helper to pass `dynamic_arg_dims`` to `_support_torch_compile``
        # to avoid too much indentation for `_support_torch_compile``
        if not hasattr(cls, 'forward'):
            raise TypeError("decorated class should have a forward method.")
        sig = inspect.signature(cls.forward)
        inferred_dynamic_arg_dims = dynamic_arg_dims
        if inferred_dynamic_arg_dims is None:
            inferred_dynamic_arg_dims = {}
            for k, v in sig.parameters.items():
                if v.annotation in [
                        torch.Tensor, Optional[torch.Tensor],
                        IntermediateTensors, Optional[IntermediateTensors]
                ]:
                    inferred_dynamic_arg_dims[k] = 0

            logger.debug(("Inferred dynamic dimensions for "
                          "forward method of %s: %s"), cls,
                         list(inferred_dynamic_arg_dims.keys()))

        if len(inferred_dynamic_arg_dims) == 0:
            raise ValueError(
                "No dynamic dimensions found in the forward method of "
                f"{cls}. Please provide dynamic_arg_dims explicitly.")

        for k in inferred_dynamic_arg_dims:
            if k not in sig.parameters:
                raise ValueError(
                    f"Argument {k} not found in the forward method of {cls}")
        return _support_torch_compile(cls, inferred_dynamic_arg_dims,
                                      enable_if)

    if cls is not None:
        # use `support_torch_compile` as a decorator without arguments
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)

    return cls_decorator_helper


def _support_torch_compile(
    cls: _T,
    dynamic_arg_dims: dict[str, Union[int, list[int]]],
    enable_if: Optional[Callable[[VllmConfig], bool]] = None,
) -> _T:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    if TorchCompileWrapperWithCustomDispatcher in cls.__bases__:
        # support decorating multiple times
        return cls

    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWrapperWithCustomDispatcher
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher, )

    old_init = cls.__init__

    setattr(cls, IGNORE_COMPILE_KEY, False)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.vllm_config = vllm_config
        enable_compile = enable_if is None or enable_if(vllm_config)
        # for CompilationLevel.DYNAMO_AS_IS , the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        self.do_not_compile = \
            vllm_config.compilation_config.level in [
            CompilationLevel.NO_COMPILATION, CompilationLevel.DYNAMO_AS_IS
        ] or not supports_dynamo() or _should_ignore_torch_compile(
            self.__class__) or not enable_compile
        if self.do_not_compile:
            return

        compilation_counter.num_models_seen += 1
        TorchCompileWrapperWithCustomDispatcher.__init__(
            self, compilation_level=vllm_config.compilation_config.level)

    cls.__init__ = __init__

    def __call__(self, *args, **kwargs):
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # the first compilation needs to have dynamic shapes marked
        if len(self.compiled_codes) < 1:
            sig = inspect.signature(self.__class__.forward)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            for k, dims in dynamic_arg_dims.items():
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(arg, torch.Tensor):
                        # In case dims is specified with negative indexing
                        dims = [
                            arg.ndim + dim if dim < 0 else dim for dim in dims
                        ]
                        torch._dynamo.mark_dynamic(arg, dims)
                    elif isinstance(arg, IntermediateTensors):
                        for tensor in arg.tensors.values():
                            # In case dims is specified with negative indexing
                            dims = [
                                tensor.ndim + dim if dim < 0 else dim
                                for dim in dims
                            ]
                            torch._dynamo.mark_dynamic(tensor, dims)
                    else:
                        raise ValueError(
                            "Unsupported dynamic dimensions"
                            f" {dims} for argument {k} with type {type(arg)}.")
            # here, it is the starting point of the `torch.compile` process
            start_monitoring_torch_compile(self.vllm_config)
            logger.debug("Start compiling function %s",
                         self.original_code_object)

        # if we don't use custom dispatcher, we can directly call the
        # compiled function and let torch.compile handle the dispatching,
        # with the overhead of guard evaluation and recompilation.
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            # it seems Dynamo reuse the compilation across instances,
            # while we need to make sure the compiled code is not reused.
            # we need to control all the compilation of the model.
            torch._dynamo.eval_frame.remove_from_cache(
                self.original_code_object)

            # collect all relevant files traced by Dynamo,
            # so that the compilation cache can trigger re-compilation
            # properly when any of these files change.

            # 1. the file containing the top-level forward function
            self.vllm_config.compilation_config.traced_files.add(
                self.original_code_object.co_filename)

            # 2. every time Dynamo sees a function call, it will inline
            # the function by calling InliningInstructionTranslator.inline_call
            # we hijack this function to know all the functions called
            # during Dynamo tracing, and their corresponding files
            inline_call = InliningInstructionTranslator.inline_call

            def patched_inline_call(parent, func, args, kwargs):
                code = func.get_code()
                self.vllm_config.compilation_config.traced_files.add(
                    code.co_filename)
                return inline_call(parent, func, args, kwargs)

            # Disable the C++ compilation of symbolic shape guards. C++-fication
            # of symbolic shape guards can improve guard overhead. But, since
            # vllm skip guards anyways, setting this flag to False can improve
            # compile time.
            dynamo_config_patches = {}
            try:
                _ = torch._dynamo.config.enable_cpp_symbolic_shape_guards
                dynamo_config_patches[
                    "enable_cpp_symbolic_shape_guards"] = False
            except AttributeError:
                # Note: this config is not available in torch 2.6, we can skip
                # if the config doesn't exist
                logger.debug(
                    "enable_cpp_symbolic_shape_guards config not available")

            with patch.object(InliningInstructionTranslator, 'inline_call',
                              patched_inline_call), torch._dynamo.config.patch(
                                  **dynamo_config_patches
                              ), maybe_use_cudagraph_partition_wrapper(
                                  self.vllm_config):
                output = self.compiled_callable(*args, **kwargs)

            return output

        # usually, capturing the model once is enough, and then we can
        # dispatch to the compiled code directly, without going through
        # the Dynamo guard mechanism.
        with self.dispatch_to_code(0):
            model_output = self.forward(*args, **kwargs)
            return model_output

    cls.__call__ = __call__
    return cls


@contextlib.contextmanager
def maybe_use_cudagraph_partition_wrapper(vllm_config: VllmConfig):
    """
    Context manager to set/unset customized cudagraph partition wrappers.

    If we're using Inductor-based graph partitioning, we currently have the
    whole `fx.Graph` before Inductor lowering and and the piecewise
    splitting happens after all graph passes and fusions. Here, we add
    a custom hook for Inductor to wrap each partition with our static
    graph wrapper class to maintain more control over static graph
    capture and replay.
    """
    from vllm.config import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if (compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and compilation_config.use_inductor_graph_partition):
        from torch._inductor.utils import CUDAGraphWrapperMetadata

        from vllm.compilation.cuda_graph import CUDAGraphOptions
        from vllm.platforms import current_platform

        static_graph_wrapper_class = resolve_obj_by_qualname(
            current_platform.get_static_graph_wrapper_cls())

        def customized_cudagraph_wrapper(f,
                                         metadata: CUDAGraphWrapperMetadata):
            partition_id = metadata.partition_index
            num_partitions = metadata.num_partitions
            return static_graph_wrapper_class(
                runnable=f,
                vllm_config=vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
                cudagraph_options=CUDAGraphOptions(
                    debug_log_enable=partition_id == 0,
                    gc_disable=partition_id != 0,
                    weak_ref_output=partition_id == num_partitions - 1,
                ))

        torch._inductor.utils.set_customized_partition_wrappers(
            customized_cudagraph_wrapper)

    yield

    if (compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and compilation_config.use_inductor_graph_partition):
        torch._inductor.utils.set_customized_partition_wrappers(None)
