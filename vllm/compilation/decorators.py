import inspect
from typing import Callable, Dict, List, Optional, TypeVar, Union, overload

import torch
import torch.nn as nn

from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils import supports_dynamo

from .monitor import start_monitoring_torch_compile

logger = init_logger(__name__)

_T = TypeVar("_T", bound=type[nn.Module])


@overload
def support_torch_compile(
    *,
    dynamic_arg_dims: Optional[Dict[str, Union[int, List[int]]]],
) -> Callable[[_T], _T]:
    ...


@overload
def support_torch_compile(cls: _T) -> _T:
    ...

import time
after_dynamo_time: float = 0

def simple_backend(gm, example_inputs):
    def returned_gm(*args):
        global after_dynamo_time
        after_dynamo_time = time.perf_counter()
        return gm(*args)
    return returned_gm

def support_torch_compile(
    cls: Optional[_T] = None,
    *,
    dynamic_arg_dims: Optional[Dict[str, Union[int, List[int]]]] = None,
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
        return _support_torch_compile(cls, inferred_dynamic_arg_dims)

    if cls is not None:
        # use `support_torch_compile` as a decorator without arguments
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)

    return cls_decorator_helper


def _support_torch_compile(
    cls: _T,
    dynamic_arg_dims: Dict[str, Union[int, List[int]]],
) -> _T:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    # if TorchCompileWrapperWithCustomDispatcher in cls.__bases__:
    #     # support decorating multiple times
    #     return cls

    # # take care of method resolution order
    # # make sure super().__init__ is called on the base class
    # #  other than TorchCompileWrapperWithCustomDispatcher
    # cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher, )

    old_init = cls.__init__

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.vllm_config = vllm_config
        # for CompilationLevel.DYNAMO_AS_IS , the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        self.do_not_compile = False
        compilation_counter.num_models_seen += 1
        self.num_runs = 0
        self.compiled_callable = torch.compile(self.forward, backend=simple_backend, dynamic=False)
        self.seen_bs = set()

    cls.__init__ = __init__

    def __call__(self, *args, **kwargs):
        self.num_runs += 1
        if self.num_runs <= 1:
            # profile run, skip compilation
            return self.forward(*args, **kwargs)

        input_ids = args[0]
        bs = input_ids.size(0)
        # if bs not in self.seen_bs, it means we are seeing a new batch size
        # and we need to compile the forward method for this batch size.
        # only measure the dynamo overhead for the following runs
        if bs in self.seen_bs:
            before_dynamo_time = time.perf_counter()
        output = self.compiled_callable(*args, **kwargs)
        if bs in self.seen_bs:
            logger.info("Time taken for dynamo guard evaluation for bs %s: %f (ms)", bs, (after_dynamo_time - before_dynamo_time) * 1000)
        self.seen_bs.add(bs)
        return output

    cls.__call__ = __call__
    return cls
