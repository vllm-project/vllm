# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily

if TYPE_CHECKING:
    from vllm.config.utils import Range

from torch._inductor.custom_graph_pass import CustomGraphPass

_pass_context = None
P = ParamSpec("P")
R = TypeVar("R")


class PassContext:
    def __init__(self, compile_range: Range):
        self.compile_range: Range = compile_range

        # set of arg indices
        self.donated_input_ids: set[int] = set()


def get_pass_context() -> PassContext:
    """Get the current pass context."""
    assert _pass_context is not None
    return _pass_context


@contextmanager
def pass_context(compile_range: Range) -> Generator[None, None, None]:
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


def set_pass_context(compile_range: Range) -> PassContext | None:
    """Install a process-global PassContext that outlives this call and return the
    previous one (usually None) so a caller could restore it.

    Unlike the ``pass_context`` context manager (used by VllmBackend around each
    synchronous per-range compile), STOCK_TORCH_COMPILE is engine-global and its
    Inductor compiles run lazily during a later forward (profile_run / warmup), so
    there is no single span to scope. The pre/post-grad vLLM passes and
    PostGradPassManager.uuid() must all observe one live PassContext, so this must
    persist until that lazy compile fires. It is safe only while no co-resident
    VllmBackend compile also reads the global: today no stock engine runs a
    VllmBackend, but that invariant is fragile if a second SupportsStockCompile arch
    or a multimodal encoder co-resides with a VllmBackend compile. The previous
    context is returned so such a future caller can restore it.
    """
    global _pass_context
    prev_context = _pass_context
    _pass_context = PassContext(compile_range)
    return prev_context


@functools.cache
def _hash_source_cached(*srcs: str | type | types.FunctionType) -> str:
    hasher = hashlib.sha256()
    for src in srcs:
        src_str = src if isinstance(src, str) else inspect.getsource(src)
        hasher.update(src_str.encode("utf-8"))
    return hasher.hexdigest()


class InductorPass(CustomGraphPass):  # type: ignore[misc]
    """
    A custom graph pass that uses a hash of its source as the UUID.
    This is defined as a convenience and should work in most cases.
    """

    def uuid(self) -> str:
        """
        Provide a unique identifier for the pass, used in Inductor code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        return InductorPass.hash_source(self)

    @staticmethod
    def hash_source(*srcs: str | Any) -> str:
        """
        Utility method to hash the sources of functions or objects.

        Args:
            srcs: strings or objects to add to the hash.
                Objects and functions have their source inspected.
                Results are cached by resolved types to avoid repeated
                inspect.getsource() calls.
        """
        # Resolve instances to their class for a hashable cache key.
        cache_key = tuple(
            src if isinstance(src, (str, type, types.FunctionType)) else src.__class__
            for src in srcs
        )
        return _hash_source_cached(*cache_key)

    @staticmethod
    def hash_dict(dict_: dict[Any, Any]) -> str:
        """
        Utility method to hash a dictionary, can alternatively be used for uuid.

        Returns:
            A sha256 hash of the json rep of the dictionary.
        """
        encoded = json.dumps(dict_, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True


class CallableInductorPass(InductorPass):
    """
    This class is a wrapper for a callable that automatically provides an
    implementation of the UUID.
    """

    def __init__(
        self, callable: Callable[[fx.Graph], None], uuid: Any | None = None
    ) -> None:
        self.callable = callable
        self._uuid = self.hash_source(callable) if uuid is None else uuid

    def __call__(self, graph: torch.fx.Graph) -> None:
        self.callable(graph)

    def uuid(self) -> Any:
        return self._uuid


def enable_fake_mode(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Applies a FakeTensorMode context. This is useful when you don't want to
    create or run things with real tensors.
    """

    @functools.wraps(fn)
    def fn_new(*args: P.args, **kwargs: P.kwargs) -> R:
        with torch._guards.tracing(None), unset_fake_temporarily(), FakeTensorMode():
            result = fn(*args, **kwargs)

        return result

    return fn_new
