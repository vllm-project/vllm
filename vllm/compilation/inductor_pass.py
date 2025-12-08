# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import hashlib
import inspect
import json
import types
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily

from vllm.config.utils import Range
from vllm.utils.torch_utils import is_torch_equal_or_newer

if is_torch_equal_or_newer("2.6"):
    from torch._inductor.custom_graph_pass import CustomGraphPass
else:
    # CustomGraphPass is not present in 2.5 or lower, import our version
    from .torch25_custom_graph_pass import (
        Torch25CustomGraphPass as CustomGraphPass,
    )

_pass_context = None


class PassContext:
    def __init__(self, compile_range: Range):
        self.compile_range: Range = compile_range


def get_pass_context() -> PassContext:
    """Get the current pass context."""
    assert _pass_context is not None
    return _pass_context


@contextmanager
def pass_context(compile_range: Range):
    """A context manager that stores the current pass context,
    usually it is a list of sizes to specialize.
    """
    global _pass_context
    prev_context = _pass_context
    _pass_context = PassContext(compile_range)
    try:
        yield
    finally:
        _pass_context = prev_context


class InductorPass(CustomGraphPass):
    """
    A custom graph pass that uses a hash of its source as the UUID.
    This is defined as a convenience and should work in most cases.
    """

    def uuid(self) -> Any:
        """
        Provide a unique identifier for the pass, used in Inductor code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        return InductorPass.hash_source(self)

    @staticmethod
    def hash_source(*srcs: str | Any):
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
            elif isinstance(src, (types.FunctionType, type)):
                src_str = inspect.getsource(src)
            else:
                # object instance
                src_str = inspect.getsource(src.__class__)
            hasher.update(src_str.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def hash_dict(dict_: dict[Any, Any]):
        """
        Utility method to hash a dictionary, can alternatively be used for uuid.
        :return: A sha256 hash of the json rep of the dictionary.
        """
        encoded = json.dumps(dict_, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def is_applicable_for_range(self, compile_range: Range):
        return True


class CallableInductorPass(InductorPass):
    """
    This class is a wrapper for a callable that automatically provides an
    implementation of the UUID.
    """

    def __init__(self, callable: Callable[[fx.Graph], None], uuid: Any | None = None):
        self.callable = callable
        self._uuid = self.hash_source(callable) if uuid is None else uuid

    def __call__(self, graph: torch.fx.Graph):
        self.callable(graph)

    def uuid(self) -> Any:
        return self._uuid


def enable_fake_mode(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Applies a FakeTensorMode context. This is useful when you don't want to
    create or run things with real tensors.
    """

    @functools.wraps(fn)
    def fn_new(*args, **kwargs) -> Any:
        with torch._guards.tracing(None), unset_fake_temporarily(), FakeTensorMode():
            result = fn(*args, **kwargs)

        return result

    return fn_new
