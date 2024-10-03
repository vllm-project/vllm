import os
import sys
import weakref
from abc import abstractmethod
from contextlib import contextmanager
from types import CodeType
from typing import Any, Callable, List, Optional, Tuple

import torch

import vllm.envs as envs


class TorchCompileWrapperWithCustomDispatcher:
    """
    A wrapper class for torch.compile, with a custom dispatch logic.
    Subclasses should:
    1. Implement the forward method
    2. Implement the dispatch logic in the __call__ method
        It can use `self.compiled_codes` to access the compiled bytecode,
        and `with self.dispatch_to_code(index):` to dispatch to
        the compiled code.
    3. Implement the `__init__` method to determine how to call
        `torch.compile` over the forward method.
    """

    def __init__(self, compiled_callable: Optional[Callable] = None):

        if compiled_callable is None:
            # default compilation settings
            # compiling the forward method

            # choose the compile backend

            # if the user has set the backend, use it
            from vllm.plugins import get_torch_compile_backend
            backend = get_torch_compile_backend()
            if backend is None:
                from vllm.compilation.backends import select_default_backend
                backend = select_default_backend(
                    envs.VLLM_TEST_TORCH_COMPILE_LEVEL)
                if not isinstance(backend, str):
                    from functools import partial
                    backend = partial(backend, model_ref=weakref.ref(self))

            compiled_callable = torch.compile(
                self.forward,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend)

        self.compiled_callable = compiled_callable
        self.original_code_object = self.__class__.forward.__code__
        self.compiled_codes: List[CodeType] = []
        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

        # read the env var to determine whether to use the custom dispatcher
        # subclasses can use this to switch between the custom dispatcher
        # and the default Dynamo guard mechanism.
        self.use_custom_dispatcher: bool = \
            envs.VLLM_DYNAMO_USE_CUSTOM_DISPATCHER

        self.sizes_to_specialize = []

    def __call__(self, *args, **kwargs):
        """Implement the dispatch logic here, beyond the torch.compile level.
        NOTE: this function can have additional arguments beyond the forward
         method, for directly dispatching to the compiled code.
        """
        return self.compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
        """Hook to save the compiled bytecode for direct execution."""
        if old_code is not self.original_code_object:
            return
        # code borrowed from https://github.com/thuml/depyf/blob/f4ad79fadee27ea113b4c75202db1eb1a11c0dbc/depyf/explain/enable_debugging.py#L25
        frame = sys._getframe()
        while True:
            frame = frame.f_back
            code_name = frame.f_code.co_name
            file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
            if code_name == "_compile" and file_name == "convert_frame.py":
                break
        frame = frame.f_locals["frame"]
        assert frame.f_code == old_code

        if frame.f_locals["self"] is not self:
            return

        self.compiled_codes.append(new_code)

    @contextmanager
    def dispatch_to_code(self, index: int):
        """Context manager to dispatch to the compiled code.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7 for more details.
        """ # noqa
        self.__class__.forward.__code__ = self.compiled_codes[index]
        yield
        self.__class__.forward.__code__ = self.original_code_object

    def set_sizes_to_specialize(self, sizes: List[Any]):
        """Set the sizes to specialize for the compiled code."""
        self.sizes_to_specialize = sizes

    def need_to_specialize(self, runtime_shapes: Tuple[int, ...]) -> bool:
        """Check if the current runtime shapes need to be specialized.
        If not, we can use the graph for general shapes.
        If yes, we will compile the graph for the current shapes.
        The argument `runtime_shapes` is a tuple of integers, representing
        the runtime shapes of the dimensions marked as dynamic during graph
        capture.
        """
        return False
