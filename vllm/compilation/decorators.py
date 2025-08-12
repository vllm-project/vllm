import inspect
from typing import Dict, List, Optional, Union

import torch

import vllm.envs as envs
from vllm.compilation.levels import CompilationLevel
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils import supports_dynamo

logger = init_logger(__name__)


def support_torch_compile(
        cls: Optional[type] = None,
        dynamic_arg_dims: Optional[Dict[str, Union[int, List[int]]]] = None):
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

    - if it is a single integer, the corresponding dimension of the argument
        will be marked as dynamic.
    - if it is `None`, ignored.
    - if it is `IntermediateTensors`, all the tensors in the intermediate
        tensors will be marked as dynamic.
    - otherwise, it will raise an error.

    NOTE: if an argument is `None`, it should always be passed as `None` during
    the lifetime of the model, otherwise, it cannot be captured as a single
    computation graph.
    """

    def cls_decorator_helper(cls: type):
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
        return _support_torch_compile(cls, inferred_dynamic_arg_dims)

    if cls is not None:
        # use `support_torch_compile` as a decorator without arguments
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)

    return cls_decorator_helper


def _support_torch_compile(cls: type,
                           dynamic_arg_dims: Dict[str, Union[int, List[int]]]):
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

    old_init = cls.__init__  # type: ignore

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        # for CompilationLevel.DYNAMO_AS_IS , the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        self.do_not_compile = envs.VLLM_TORCH_COMPILE_LEVEL in [
            CompilationLevel.NO_COMPILATION, CompilationLevel.DYNAMO_AS_IS
        ] or not supports_dynamo()
        if self.do_not_compile:
            return
        TorchCompileWrapperWithCustomDispatcher.__init__(self)

    cls.__init__ = __init__  # type: ignore

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
                    if isinstance(arg, torch.Tensor):
                        torch._dynamo.mark_dynamic(arg, dims)
                    elif isinstance(arg, IntermediateTensors):
                        for tensor in arg.tensors.values():
                            torch._dynamo.mark_dynamic(tensor, dims)
                    else:
                        raise ValueError(
                            "Unsupported dynamic dimensions"
                            f" {dims} for argument {k} with type {type(arg)}.")

        # if we don't use custom dispatcher, we can directly call the
        # compiled function and let torch.compile handle the dispatching,
        # with the overhead of guard evaluation and recompilation.
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            # it seems Dynamo reuse the compilation across instances,
            # while we need to make sure the compiled code is not reused.
            # we need to control all the compilation of the model.
            torch._dynamo.eval_frame.remove_from_cache(
                self.original_code_object)
            return self.compiled_callable(*args, **kwargs)

        # usually, capturing the model once is enough, and then we can
        # dispatch to the compiled code directly, without going through
        # the Dynamo guard mechanism.
        with self.dispatch_to_code(0):
            model_output = self.forward(*args, **kwargs)
            return model_output

    cls.__call__ = __call__  # type: ignore
    return cls
