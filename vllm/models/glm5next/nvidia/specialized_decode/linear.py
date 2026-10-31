# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny projections for stock BF16 linear modules inside planned decode steps.

Stages that still call stock modules (the DSA attention) reach their BF16
projections through ``quant_method.apply``; ``route`` wraps that call so a planned
step's 1 or 2 token counts take the skinny projection, the counts given to
``route`` the split-K GEMM (DSA layers and the MTP draft), and every other call
(prefills, stock steps, other counts) the stock GEMM. At 4 tokens the DSA layer
measured slower with the skinny projection in place (124.6 -> 128.8 us per
layer at B4), although the projections alone are faster.
"""

from contextlib import contextmanager

import torch

from .gemm import Projection

MAX_TOKENS = 2
_active = [False]


def planned() -> bool:
    """Whether a planned decode step is being captured."""
    return _active[0]


@contextmanager
def planned_step():
    _active[0] = True
    try:
        yield
    finally:
        _active[0] = False


class _RoutedApply:
    def __init__(self, inner, projection, tokens):
        self.inner, self.projection, self.tokens = inner, projection, tokens

    def apply(self, layer, x, bias=None):
        if (
            _active[0]
            and bias is None
            and x.dim() == 2
            and x.shape[0] in self.tokens
            and x.stride(-1) == 1
            and x.stride(0) % 8 == 0
        ):
            return self.projection(x)
        return self.inner.apply(layer, x, bias)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def route(module, split_k_tokens=()) -> bool:
    """Routes a BF16 linear module's decode GEMMs; False if it cannot.
    ``split_k_tokens``: token counts above ``MAX_TOKENS`` to route to split-K."""
    weight = getattr(module, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return False
    if getattr(module, "_use_min_latency_gemm", False):
        return False
    if isinstance(module.quant_method, _RoutedApply):
        return True
    try:
        projection = Projection(weight, split_k_tokens)
    except ValueError:
        return False
    tokens = {m for m in projection.configs if m <= MAX_TOKENS}
    tokens |= {m for m in split_k_tokens if projection.supports(m)}
    if not tokens:  # no tactic splits this K
        return False
    module.quant_method = _RoutedApply(module.quant_method, projection, tokens)
    return True
