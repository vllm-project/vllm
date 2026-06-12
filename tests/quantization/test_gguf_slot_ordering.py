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
    # The canonical iteration order is cached on the param so apply() can
    # iterate without re-sorting on every forward pass.
    assert layer.qweight.canonical_shard_id == [0, 1]


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
    assert layer.qweight.canonical_shard_id == ["q", "k", "v"]


def test_apply_concatenates_slots_in_canonical_order(monkeypatch):
    """The concatenated apply() output should keep the declared slot
    order even if slots were loaded in stream order."""
    # Build a layer with two slots loaded out of declared order, then run
    # _create_padded_weight_param to materialize qweight.
    layer = _make_layer(
        shard_sizes=[(4, 8), (6, 8)],
        load_order=[1, 0],
    )
    GGUFLinearMethod._create_padded_weight_param(  # noqa: SLF001
        GGUFLinearMethod.__new__(GGUFLinearMethod), layer
    )

    # Stand-in qweight_type; apply() reads the per-slot type from this.
    qweight_type = type(
        "QT",
        (),
        {"shard_weight_type": {0: "T0", 1: "T1"}, "weight_type": "TX"},
    )()
    setattr(layer, "qweight_type", qweight_type)  # noqa: B010

    # Stub the kernel; record which slot it was called for so we can
    # assert the iteration order, and return a sentinel column-vector
    # carrying the slot id.
    calls: list[int] = []

    def fake_kernel(x, _qweight, qweight_type):
        slot = {"T0": 0, "T1": 1}[qweight_type]
        calls.append(slot)
        return torch.full((x.shape[0], 1), float(slot))

    import vllm.model_executor.layers.quantization.gguf as gguf_mod

    monkeypatch.setattr(gguf_mod, "fused_mul_mat_gguf", fake_kernel)

    method = GGUFLinearMethod.__new__(GGUFLinearMethod)
    out = method.apply(layer, torch.zeros(2, 8))

    assert calls == [0, 1], f"apply() iterated slots in {calls}, expected [0, 1]"
    # Concatenated [slot 0 col, slot 1 col] => [[0, 1], [0, 1]].
    assert torch.equal(out, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
