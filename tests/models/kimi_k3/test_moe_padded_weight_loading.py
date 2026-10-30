# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic MoE weight loading with an MXFP4-padded Kimi-K3 partition."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

LOGICAL_INTERMEDIATE = 3072
TP_SIZE = 32
RAW_LOCAL = LOGICAL_INTERMEDIATE // TP_SIZE
PADDED_LOCAL = 128
MXFP4_BLOCK = 32
SYNTHETIC_HIDDEN = 64


class _Loader:
    def __init__(self, tp_size: int = TP_SIZE):
        self.moe_config = SimpleNamespace(
            is_act_and_mul=True,
            moe_parallel_config=SimpleNamespace(tp_size=tp_size),
        )

    _get_hidden_dim = staticmethod(RoutedExperts._get_hidden_dim)
    _narrow_expert_data_for_padding = staticmethod(
        RoutedExperts._narrow_expert_data_for_padding
    )
    _load_w13 = RoutedExperts._load_w13
    _load_w2 = RoutedExperts._load_w2


def _rows(width: int, columns: int = 1) -> torch.Tensor:
    return torch.arange(1, width + 1).unsqueeze(1).expand(width, columns).clone()


def _columns(rows: int, width: int) -> torch.Tensor:
    return torch.arange(1, width + 1).unsqueeze(0).expand(rows, width).clone()


def test_mxfp4_allocates_expected_kimi_k3_shapes():
    method = object.__new__(Mxfp4MoEMethod)
    method.moe = SimpleNamespace(has_bias=False, w13_num_shards=2)
    layer = nn.Module()

    method.create_weights(
        layer=layer,
        num_experts=1,
        hidden_size=3584,
        intermediate_size_per_partition=PADDED_LOCAL,
        params_dtype=torch.bfloat16,
    )

    assert layer.w13_weight.shape == (1, 256, 1792)
    assert layer.w2_weight.shape == (1, 3584, 64)
    assert layer.w13_weight_scale.shape == (1, 256, 112)
    assert layer.w2_weight_scale.shape == (1, 3584, 4)
    assert torch.count_nonzero(layer.w13_weight) == 0
    assert torch.count_nonzero(layer.w2_weight) == 0
    assert torch.count_nonzero(layer.w13_weight_scale) == 0
    assert torch.count_nonzero(layer.w2_weight_scale) == 0


@pytest.mark.parametrize("shard_id", ["w1", "w3"])
@pytest.mark.parametrize("tp_rank", [0, 31])
def test_w13_rank_boundaries_and_zero_tail(shard_id: str, tp_rank: int):
    checkpoint = _rows(LOGICAL_INTERMEDIATE, SYNTHETIC_HIDDEN // 2)
    parameter = torch.zeros(2 * PADDED_LOCAL, SYNTHETIC_HIDDEN // 2, dtype=torch.int64)

    _Loader()._load_w13(parameter, 0, shard_id, checkpoint, tp_rank)

    half_start = 0 if shard_id == "w1" else PADDED_LOCAL
    half = parameter[half_start : half_start + PADDED_LOCAL]
    expected = checkpoint.narrow(0, tp_rank * RAW_LOCAL, RAW_LOCAL)
    torch.testing.assert_close(half[:RAW_LOCAL], expected)
    assert torch.count_nonzero(half[RAW_LOCAL:]) == 0
    other_start = PADDED_LOCAL if shard_id == "w1" else 0
    assert torch.count_nonzero(parameter[other_start : other_start + PADDED_LOCAL]) == 0


def test_w13_all_tp32_ranks_cover_checkpoint_once():
    checkpoint = _rows(LOGICAL_INTERMEDIATE)
    seen = torch.zeros(LOGICAL_INTERMEDIATE, dtype=torch.int32)

    for tp_rank in range(TP_SIZE):
        parameter = torch.zeros(2 * PADDED_LOCAL, 1, dtype=torch.int64)
        _Loader()._load_w13(parameter, 0, "w1", checkpoint, tp_rank)
        indices = parameter[:RAW_LOCAL, 0] - 1
        seen[indices] += 1
        assert torch.count_nonzero(parameter[RAW_LOCAL:PADDED_LOCAL]) == 0

    assert torch.equal(seen, torch.ones_like(seen))


@pytest.mark.parametrize("tp_rank", [0, 31])
def test_w2_rank_boundaries_and_zero_tail(tp_rank: int):
    packed_width = LOGICAL_INTERMEDIATE // 2
    raw_packed_local = RAW_LOCAL // 2
    checkpoint = _columns(SYNTHETIC_HIDDEN, packed_width)
    parameter = torch.zeros(SYNTHETIC_HIDDEN, PADDED_LOCAL // 2, dtype=torch.int64)

    _Loader()._load_w2(parameter, 1, checkpoint, tp_rank)

    expected = checkpoint.narrow(1, tp_rank * raw_packed_local, raw_packed_local)
    torch.testing.assert_close(parameter[:, :raw_packed_local], expected)
    assert torch.count_nonzero(parameter[:, raw_packed_local:]) == 0


def test_w2_all_tp32_ranks_cover_checkpoint_once():
    packed_width = LOGICAL_INTERMEDIATE // 2
    raw_packed_local = RAW_LOCAL // 2
    checkpoint = _columns(1, packed_width)
    seen = torch.zeros(packed_width, dtype=torch.int32)

    for tp_rank in range(TP_SIZE):
        parameter = torch.zeros(1, PADDED_LOCAL // 2, dtype=torch.int64)
        _Loader()._load_w2(parameter, 1, checkpoint, tp_rank)
        indices = parameter[0, :raw_packed_local] - 1
        seen[indices] += 1
        assert torch.count_nonzero(parameter[:, raw_packed_local:]) == 0

    assert torch.equal(seen, torch.ones_like(seen))


@pytest.mark.parametrize("shard_id", ["w1", "w3"])
@pytest.mark.parametrize("tp_rank", [0, 31])
def test_w13_scale_loading(shard_id: str, tp_rank: int):
    checkpoint = _rows(LOGICAL_INTERMEDIATE, SYNTHETIC_HIDDEN // MXFP4_BLOCK)
    parameter = torch.zeros(
        2 * PADDED_LOCAL, SYNTHETIC_HIDDEN // MXFP4_BLOCK, dtype=torch.int64
    )

    _Loader()._load_w13(parameter, 0, shard_id, checkpoint, tp_rank)

    half_start = 0 if shard_id == "w1" else PADDED_LOCAL
    half = parameter[half_start : half_start + PADDED_LOCAL]
    expected = checkpoint.narrow(0, tp_rank * RAW_LOCAL, RAW_LOCAL)
    torch.testing.assert_close(half[:RAW_LOCAL], expected)
    assert torch.count_nonzero(half[RAW_LOCAL:]) == 0


@pytest.mark.parametrize("tp_rank", [0, 31])
def test_w2_scale_loading(tp_rank: int):
    logical_scale_width = LOGICAL_INTERMEDIATE // MXFP4_BLOCK
    raw_scale_local = RAW_LOCAL // MXFP4_BLOCK
    checkpoint = _columns(SYNTHETIC_HIDDEN, logical_scale_width)
    parameter = torch.zeros(
        SYNTHETIC_HIDDEN, PADDED_LOCAL // MXFP4_BLOCK, dtype=torch.int64
    )

    _Loader()._load_w2(parameter, 1, checkpoint, tp_rank)

    expected = checkpoint.narrow(1, tp_rank * raw_scale_local, raw_scale_local)
    torch.testing.assert_close(parameter[:, :raw_scale_local], expected)
    assert torch.count_nonzero(parameter[:, raw_scale_local:]) == 0


def test_tp8_unpadded_loading_is_unchanged():
    tp_size = 8
    local = LOGICAL_INTERMEDIATE // tp_size
    checkpoint = _rows(LOGICAL_INTERMEDIATE)
    parameter = torch.zeros(2 * local, 1, dtype=torch.int64)

    _Loader(tp_size)._load_w13(parameter, 0, "w1", checkpoint, tp_rank=7)

    torch.testing.assert_close(parameter[:local], checkpoint[-local:])
    assert torch.count_nonzero(parameter[:local]) == parameter[:local].numel()
