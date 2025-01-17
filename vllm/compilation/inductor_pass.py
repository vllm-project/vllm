import hashlib
import inspect
import types
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Optional, Union

import torch
from torch import fx

_pass_context = None


class PassContext:

    def __init__(self, runtime_shape: Optional[int]):
        self.runtime_shape = runtime_shape


def get_pass_context() -> PassContext:
    """Get the current pass context."""
    assert _pass_context is not None
    return _pass_context


@contextmanager
def pass_context(runtime_shape: Optional[int]):
    """A context manager that stores the current pass context,
    usually it is a list of sizes to specialize.
    """
    global _pass_context
    prev_context = _pass_context
    _pass_context = PassContext(runtime_shape)
    try:
        yield
    finally:
        _pass_context = prev_context


class InductorPass(ABC):
    """
    General custom inductor pass interface.
    TODO(torch==2.6) use torch._inductor.custom_graph_pass.CustomGraphPass
    """

    @abstractmethod
    def __call__(self, graph: torch.fx.Graph):
        """
        Execute the pass on the given graph.
        """
        raise NotImplementedError

    def uuid(self) -> Any:
        """
        Provide a unique identifier for the pass, used in Inductor code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        return InductorPass.hash_source(self)

    @staticmethod
    def hash_source(*srcs: Union[str, Any]):
        """
        Utility method to hash the sources of functions or objects.
        :param srcs: strings or objects to add to the hash.
        Objects and functions have their source inspected.
        :return:
        """
        hasher = hashlib.sha256()
        for src in srcs:
            if isinstance(src, str):
                src_str = src
            elif isinstance(src, types.FunctionType):
                src_str = inspect.getsource(src)
            else:
                src_str = inspect.getsource(src.__class__)
            hasher.update(src_str.encode("utf-8"))
        return hasher.digest()


class CallableInductorPass(InductorPass):
    """
    This class is a wrapper for a callable that automatically provides an
    implementation of the UUID.
    """

    def __init__(self,
                 callable: Callable[[fx.Graph], None],
                 uuid: Optional[Any] = None):
        self.callable = callable
        if uuid is None:
            uuid = InductorPass.hash_source(callable)
        self._uuid = uuid

    def __call__(self, graph: torch.fx.Graph):
        self.callable(graph)

    def uuid(self) -> Any:
        return self._uuid

    def __getstate__(self):
        """
        Pickling occurs in the Inductor code cache if a pass is not given to
        the pass manager but is instead directly added to config as a pass.
        See PostGradPassManager for more.

        TODO(torch==2.6), use the `uuid` method in CustomGraphPass instead.
        """
        return self._uuid

    def __setstate__(self, state):
        raise ValueError("Cannot unpickle CallableInductorPass")
