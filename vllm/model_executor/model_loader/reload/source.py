# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint source-key context for logical weight-load events.

The context is established by the outer checkpoint iterator and remains
active while the consumer handles the yielded tensor. Target-side
``weight_loader`` wrappers can therefore associate a routed target fragment
with the original checkpoint key without changing hundreds of model-specific
``load_weights`` implementations.
"""

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar, Token

import torch

__all__ = ["get_current_source_key", "observe_weight_sources"]

_CURRENT_SOURCE_KEY: ContextVar[str | None] = ContextVar(
    "vllm_weight_load_source_key", default=None)


@contextmanager
def _source_key_context(source_key: str):
    token: Token[str | None] = _CURRENT_SOURCE_KEY.set(source_key)
    try:
        yield
    finally:
        _CURRENT_SOURCE_KEY.reset(token)


def get_current_source_key() -> str | None:
    """Return the original checkpoint key currently being consumed."""
    return _CURRENT_SOURCE_KEY.get()


def observe_weight_sources(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield weights while exposing each original source key to loaders.

    A generator remains suspended inside the context manager for the whole
    duration of ``yield``. Consequently synchronous mapper/routing layers can
    rename or split the item while target loaders still observe its original
    checkpoint key. A truly deferred loader must persist the source key with
    its deferred arguments instead of relying on this transient context.
    """
    for source_key, loaded_weight in weights:
        with _source_key_context(source_key):
            yield source_key, loaded_weight
