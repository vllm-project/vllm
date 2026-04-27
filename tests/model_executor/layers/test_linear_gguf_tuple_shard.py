# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tuple shard_id support on the GGUF weight-loading path.

GGUF files sometimes pack several output slots into one on-disk tensor
(e.g. Qwen3.5 GDN's attn_qkv carries Q, K and V together). The loader
used to refuse such tensors with NotImplementedError. These tests check
that it now splits the fused tensor along output_dim and recurses with
single-int shard ids.

Also covers the ReplicatedLinear [N] -> [1, N] shape gap that bites
single-output linears like the MoE shared_expert_gate.
"""

import types
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)


class _FakeQWeight(torch.nn.Parameter):
    """Stand-in for a GGUFUninitializedParameter with just the attrs that
    the GGUF branch of weight_loader reads."""

    def __new__(cls):
        return super().__new__(cls, torch.empty(0), requires_grad=False)


def _make_fake_self(
    output_sizes: list[int], tp_size: int = 1, tp_rank: int = 0
) -> SimpleNamespace:
    fake_self = SimpleNamespace(
        output_sizes=output_sizes, tp_size=tp_size, tp_rank=tp_rank
    )
    fake_self.validate_shard_id = types.MethodType(
        MergedColumnParallelLinear.validate_shard_id, fake_self
    )
    fake_self.weight_loader = types.MethodType(
        MergedColumnParallelLinear.weight_loader, fake_self
    )
    return fake_self


def _make_gguf_qweight() -> _FakeQWeight:
    p = _FakeQWeight()
    p.is_gguf_weight = True
    p.output_dim = 0
    p.data_container = []
    p.shard_id = []
    p.shard_id_map = {}
    return p


def _make_gguf_qweight_type(num_slots: int) -> _FakeQWeight:
    p = _FakeQWeight()
    p.is_gguf_weight_type = True
    p.shard_weight_type = {}
    # ``param.data[idx].copy_(scalar)`` requires a buffer of length num_slots.
    p.data = torch.zeros(num_slots, dtype=torch.uint8)
    return p


def test_tuple_shard_splits_fused_gguf_weight_into_slots():
    """A fused GGUF tensor with loaded_shard_id=(0, 1, 2) should land as
    three per-slot entries in data_container."""
    output_sizes = [4, 6, 8]
    fused = torch.cat(
        [torch.full((size, 16), float(slot)) for slot, size in enumerate(output_sizes)],
        dim=0,
    )
    fake_self = _make_fake_self(output_sizes)
    param = _make_gguf_qweight()

    MergedColumnParallelLinear.weight_loader(fake_self, param, fused, (0, 1, 2))

    assert len(param.data_container) == 3
    assert param.shard_id == [0, 1, 2]
    assert param.shard_id_map == {0: 0, 1: 1, 2: 2}
    for slot, size in enumerate(output_sizes):
        slot_tensor = param.data_container[param.shard_id_map[slot]]
        assert slot_tensor.shape == (size, 16)
        assert torch.all(slot_tensor == float(slot))


def test_tuple_shard_propagates_weight_type_scalar_to_each_slot():
    """is_gguf_weight_type tensors carry one scalar shared by every slot;
    the recursion should store it under each slot."""
    output_sizes = [4, 6]
    weight_type_scalar = torch.tensor(42, dtype=torch.uint8)
    fake_self = _make_fake_self(output_sizes)
    param = _make_gguf_qweight_type(num_slots=len(output_sizes))

    MergedColumnParallelLinear.weight_loader(
        fake_self, param, weight_type_scalar, (0, 1)
    )

    assert param.shard_weight_type == {0: 42, 1: 42}


def test_tuple_shard_assert_catches_size_mismatch():
    """If the fused tensor's output-dim size doesn't match
    sum(output_sizes[loaded_shard_id]) the loader should raise."""
    output_sizes = [4, 6, 8]
    fused = torch.zeros(17, 16)  # 17 != 4+6+8
    fake_self = _make_fake_self(output_sizes)
    param = _make_gguf_qweight()

    with pytest.raises(AssertionError, match="Fused GGUF weight"):
        MergedColumnParallelLinear.weight_loader(fake_self, param, fused, (0, 1, 2))


def test_tuple_shard_per_slot_tp_sharding():
    """Each slot's narrow() should run with its own size after the split,
    so a TP=2 rank=1 load lands the second half of every slot."""
    output_sizes = [4, 6]
    fused = torch.cat(
        [
            torch.arange(size * 16, dtype=torch.float32).reshape(size, 16) + 100 * slot
            for slot, size in enumerate(output_sizes)
        ],
        dim=0,
    )
    fake_self = _make_fake_self(output_sizes, tp_size=2, tp_rank=1)
    param = _make_gguf_qweight()

    MergedColumnParallelLinear.weight_loader(fake_self, param, fused, (0, 1))

    # Each slot should have been narrowed to its second half along output_dim.
    assert len(param.data_container) == 2
    for slot, full_size in enumerate(output_sizes):
        per_rank = full_size // 2
        slot_tensor = param.data_container[param.shard_id_map[slot]]
        assert slot_tensor.shape == (per_rank, 16), (
            f"slot {slot}: got shape {slot_tensor.shape}, expected {(per_rank, 16)}"
        )
        # rank 1 took the second half of slot's portion of the fused tensor
        slot_start = sum(output_sizes[:slot])
        expected = fused.narrow(0, slot_start + per_rank, per_rank)
        assert torch.equal(slot_tensor, expected)


def test_tuple_shard_rejects_indivisible_slot_under_tp():
    """A slot whose output_size isn't divisible by tp_size should be
    flagged before silent narrow() truncation kicks in."""
    output_sizes = [3, 6]  # slot 0 not divisible by 2
    fused = torch.zeros(9, 16)
    fake_self = _make_fake_self(output_sizes, tp_size=2, tp_rank=0)
    param = _make_gguf_qweight()

    with pytest.raises(AssertionError, match="not divisible by tp_size"):
        MergedColumnParallelLinear.weight_loader(fake_self, param, fused, (0, 1))


def test_replicated_linear_unsqueezes_1d_gguf_weight_for_single_output():
    """A 1-D GGUF tensor of length hidden should land in a [1, hidden]
    parameter (e.g. shared_expert_gate)."""
    hidden = 32
    param = torch.nn.Parameter(torch.zeros(1, hidden), requires_grad=False)
    param.is_gguf_weight = True
    loaded_weight = torch.arange(hidden, dtype=torch.float32)
    fake_self = SimpleNamespace(output_size=1, input_size=hidden)

    ReplicatedLinear.weight_loader(fake_self, param, loaded_weight)

    assert param.shape == (1, hidden)
    assert torch.equal(param.data.flatten(), loaded_weight)


def test_replicated_linear_unsqueezes_before_materialize():
    """An UninitializedParameter must end up at [1, hidden], not [hidden].
    Reshaping after materialize would leave it 1-D forever."""
    from torch.nn.parameter import UninitializedParameter

    hidden = 16
    param = UninitializedParameter()
    param.is_gguf_weight = True
    loaded_weight = torch.arange(hidden, dtype=torch.float32)
    fake_self = SimpleNamespace(output_size=1, input_size=hidden)

    ReplicatedLinear.weight_loader(fake_self, param, loaded_weight)

    assert param.shape == (1, hidden)
    assert torch.equal(param.data.flatten(), loaded_weight)


def test_replicated_linear_does_not_unsqueeze_non_gguf_weight():
    """The unsqueeze fix should be GGUF-only — non-GGUF callers with a
    1-D vs 2-D mismatch should still trip the assert."""
    hidden = 8
    param = torch.nn.Parameter(torch.zeros(1, hidden), requires_grad=False)
    loaded_weight = torch.arange(hidden, dtype=torch.float32)
    fake_self = SimpleNamespace(output_size=1, input_size=hidden)

    with pytest.raises(AssertionError):
        ReplicatedLinear.weight_loader(fake_self, param, loaded_weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
