# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expose checkpoint source names while model-specific loaders route tensors."""

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_SOURCE_NAME: ContextVar[str | None] = ContextVar("weight_source_name", default=None)


def get_current_source_name() -> str | None:
    return _SOURCE_NAME.get()


@contextmanager
def source_name_context(name: str):
    token = _SOURCE_NAME.set(name)
    try:
        yield
    finally:
        _SOURCE_NAME.reset(token)


def observe_weight_sources(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    before_yield: Callable[[str], None] | None = None,
    after_yield: Callable[[str], None] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield checkpoint tensors while exposing their source-name lifecycle.

    ``before_yield`` runs immediately before the model loader can access a
    tensor, which lets reload allocate only that source's destinations.
    ``after_yield`` runs after the loader requests the next tensor, so all
    weight-loader calls for the previous source have completed and their shard
    receipts can be used to decide whether a reload unit is complete.
    """
    for name, tensor in weights:
        if before_yield is not None:
            before_yield(name)
        with source_name_context(name):
            yield name, tensor
        if after_yield is not None:
            after_yield(name)
