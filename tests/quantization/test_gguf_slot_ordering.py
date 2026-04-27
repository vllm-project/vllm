# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GGUF MergedColumn slot-ordering.

When a GGUF file emits merged-column slots out of declared order, the
padded buffer used to be laid out in load order, so the concatenated
output ended up shuffled (e.g. [Z, Q, K, V] instead of [Q, K, V, Z]).
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.gguf import GGUFLinearMethod


class _FakeQWeight(torch.nn.Parameter):
    """Stand-in for GGUFUninitializedParameter with just the attrs that
    _create_padded_weight_param reads."""

    def __new__(cls):
        return super().__new__(cls, torch.empty(0), requires_grad=False)


class _FakeLayer:
    def __init__(self, qweight):
        self.qweight = qweight

    def register_parameter(self, name, param):
        setattr(self, name, param)


def _make_layer(
    shard_sizes: list[tuple[int, int]],
    load_order: list[int],
) -> _FakeLayer:
    """``shard_sizes`` is in slot-index order; ``load_order`` is the order the
    GGUF file emits the slots."""
    data_container = []
    shard_id_map: dict[int, int] = {}
    # Fill each slot with a constant equal to the slot index so placement can
    # be verified by reading the buffer back.
    for pos, slot in enumerate(load_order):
        rows, cols = shard_sizes[slot]
        data_container.append(torch.full((rows, cols), float(slot)))
        shard_id_map[slot] = pos

    qweight = _FakeQWeight()
    qweight.data_container = data_container
    qweight.shard_id = list(load_order)
    qweight.shard_id_map = shard_id_map
    return _FakeLayer(qweight)


def test_padded_weight_param_uses_canonical_slot_order():
    """Slots loaded out of order should still land at index-ordered offsets."""
    # Slot 0: 4 rows, slot 1: 6 rows. GGUF emits slot 1 first, then slot 0.
    layer = _make_layer(
        shard_sizes=[(4, 8), (6, 8)],
        load_order=[1, 0],
    )

    GGUFLinearMethod._create_padded_weight_param(  # noqa: SLF001
        GGUFLinearMethod.__new__(GGUFLinearMethod), layer
    )

    # Canonical layout: slot 0 occupies rows [0:4], slot 1 occupies rows [4:10].
    assert layer.qweight.shard_offset_map[0] == (0, 4, 8)
    assert layer.qweight.shard_offset_map[1] == (4, 10, 8)
    # The actual buffer contents must reflect the canonical order too.
    assert torch.all(layer.qweight.data[0:4] == 0.0)
    assert torch.all(layer.qweight.data[4:10] == 1.0)


def test_padded_weight_param_load_order_matches_canonical():
    """Backward compatibility: canonical-order load behaves identically."""
    layer = _make_layer(
        shard_sizes=[(4, 8), (6, 8)],
        load_order=[0, 1],
    )

    GGUFLinearMethod._create_padded_weight_param(  # noqa: SLF001
        GGUFLinearMethod.__new__(GGUFLinearMethod), layer
    )

    assert layer.qweight.shard_offset_map[0] == (0, 4, 8)
    assert layer.qweight.shard_offset_map[1] == (4, 10, 8)
    assert torch.all(layer.qweight.data[0:4] == 0.0)
    assert torch.all(layer.qweight.data[4:10] == 1.0)


def test_qkv_string_keys_keep_qkv_order():
    """q/k/v string keys keep the q,k,v layout regardless of stream order."""
    data_container = [
        torch.full((3, 8), 2.0),  # v
        torch.full((3, 8), 0.0),  # q
        torch.full((3, 8), 1.0),  # k
    ]
    qweight = _FakeQWeight()
    qweight.data_container = data_container
    qweight.shard_id = ["v", "q", "k"]
    qweight.shard_id_map = {"v": 0, "q": 1, "k": 2}
    layer = _FakeLayer(qweight)

    GGUFLinearMethod._create_padded_weight_param(  # noqa: SLF001
        GGUFLinearMethod.__new__(GGUFLinearMethod), layer
    )

    assert layer.qweight.shard_offset_map["q"] == (0, 3, 8)
    assert layer.qweight.shard_offset_map["k"] == (3, 6, 8)
    assert layer.qweight.shard_offset_map["v"] == (6, 9, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
