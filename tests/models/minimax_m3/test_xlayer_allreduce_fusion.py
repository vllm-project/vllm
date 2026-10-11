# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 (AMD) cross-layer FFN all-reduce deferral wiring.

Every deferred all-reduce must be completed exactly once, by the next layer's
input norm or the final norm, and an EAGLE3 capture must never record a
per-rank partial sum. The wiring only reads a few layer attributes, so it is
driven on CPU with stand-in layers.
"""

from itertools import product
from types import SimpleNamespace

import pytest

from vllm.models.minimax_m3.amd.model import MiniMaxM3Model


def _model(defers, aux=()):
    layers = [
        SimpleNamespace(ffn_all_reduce_deferred=d, fuse_input_allreduce=False)
        for d in defers
    ]
    m = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        aux_hidden_state_layers=tuple(aux),
        fuse_final_norm_allreduce=False,
    )
    MiniMaxM3Model._configure_allreduce_fusion(m)
    return m


def _simulate(m) -> tuple[bool, bool]:
    """Walk one forward pass: (captures see reduced values, AR chain balanced)."""
    pending = False
    captures_ok = balance_ok = True
    for idx, layer in enumerate(m.layers):
        if layer.fuse_input_allreduce != pending:
            balance_ok = False
        pending = layer.ffn_all_reduce_deferred
        cap = idx + 1
        if cap in m.aux_hidden_state_layers and pending:
            captures_ok &= MiniMaxM3Model._aux_needs_reduce(m, cap)
    balance_ok &= pending == m.fuse_final_norm_allreduce
    return captures_ok, balance_ok


def test_dense_then_moe_chain():
    m = _model([False] * 3 + [True] * 5)
    fused = [layer.fuse_input_allreduce for layer in m.layers]
    assert fused == [False] * 4 + [True] * 4
    assert m.fuse_final_norm_allreduce


def test_capture_reduces_only_after_a_deferring_layer():
    m = _model([True, False, True, True])
    assert MiniMaxM3Model._aux_needs_reduce(m, 1)
    assert not MiniMaxM3Model._aux_needs_reduce(m, 2)
    assert not MiniMaxM3Model._aux_needs_reduce(m, 0)


@pytest.mark.parametrize("n", range(1, 7))
def test_every_defer_pattern_and_capture_set(n):
    for defers in product([True, False], repeat=n):
        for mask in range(1 << (n + 1)):
            aux = tuple(i for i in range(n + 1) if mask >> i & 1)
            assert _simulate(_model(defers, aux)) == (True, True), (defers, aux)
