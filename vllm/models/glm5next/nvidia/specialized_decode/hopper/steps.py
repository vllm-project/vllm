# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builders of the Hopper executor's stages over the stock layers as loaded on
Hopper, one per stage module (`moe`, `dense`, `kda`, `dsa`, `boundary`,
`logits`).

A builder returns None, and its stage runs stock, when the stage's module is not
part of the build, when the policy plans the stage at no token count, or when
the stage declines the layer: its `of` returns None, or its constructor raises
`ValueError` with the reason (logged once per stage and reason)."""

import importlib
from typing import Any, NamedTuple

from vllm.logger import init_logger

logger = init_logger(__name__)


def stage_class(module: str, name: str) -> Any:
    """`hopper.<module>.<name>`, or None while `hopper.<module>` is not part of
    the build (its stage then runs stock). Errors inside the module raise."""
    path = f"{__package__}.{module}"
    try:
        found = importlib.import_module(path)
    except ModuleNotFoundError as error:
        if error.name != path:
            raise
        logger.info_once(
            "Hopper specialized decode: no %s module, its stage runs stock", module
        )
        return None
    return getattr(found, name)


class Stage(NamedTuple):
    """A built stage kernel and the token counts the executor gives it."""

    name: str
    kernel: Any
    tokens: frozenset[int]

    def serves(self, tokens: int) -> bool:
        return tokens in self.tokens


def declined(stage: str, reason: object) -> None:
    logger.warning_once(
        "Hopper specialized decode runs the %s stage stock: %s", stage, reason
    )


def per_layer(
    stage: str, module: str, name: str, tokens: list[int], *of_args
) -> Stage | None:
    """A stage with the `of(*of_args)` / `supports(tokens)` / `prepare(tokens)`
    interface (the MoE and dense MLP stages), at the `tokens` it supports."""
    cls = stage_class(module, name)
    if cls is None or not tokens:
        return None
    kernel = cls.of(*of_args)
    if kernel is None:
        declined(stage, f"{name}.of declined the layer")
        return None
    tokens = [m for m in tokens if kernel.supports(m)]
    if not tokens:
        declined(stage, f"{name} supports none of the planned token counts")
        return None
    kernel.prepare(tokens)
    return Stage(stage, kernel, frozenset(tokens))


def per_attention(
    stage: str, module: str, name: str, attention, tokens: list[int], q: int
) -> Stage | None:
    """An attention stage built as `cls(attention, tokens, q)` (the KDA and DSA
    layers); `kernel.supports(tokens, q)` decides each call once its caches are
    prepared (`prepare_caches`)."""
    cls = stage_class(module, name)
    if cls is None or not tokens:
        return None
    try:
        kernel = cls(attention, tokens, q)
    except ValueError as error:
        declined(stage, error)
        return None
    return Stage(stage, kernel, frozenset(tokens))
