# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
    scaled_quantize,
)
from vllm.model_executor.models.mimo_v2 import (
    MiMoV2Model,
    _shard_fp8_qkv_proj,
)


def _expanded_scales(
    scales: torch.Tensor,
    shape: torch.Size | tuple[int, int],
    block: int,
) -> torch.Tensor:
    return scales.repeat_interleave(block, 0).repeat_interleave(block, 1)[
        : shape[0], : shape[1]
    ]


def _qkv_rows_per_group(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
) -> int:
    return (num_heads // num_kv_heads) * head_dim + head_dim + v_head_dim


def _make_pro_format_fp8_qkv(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    hidden_size: int,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_per_group = _qkv_rows_per_group(
        num_heads, num_kv_heads, head_dim, v_head_dim
    )
    full = (
        torch.randn(num_kv_heads * rows_per_group, hidden_size, dtype=torch.float32)
        * 0.1
    )

    weights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for group_idx in range(num_kv_heads):
        group_start = group_idx * rows_per_group
        group = full[group_start : group_start + rows_per_group]
        pad_rows = (-rows_per_group) % block
        if pad_rows:
            group = torch.cat(
                [
                    group,
                    torch.zeros((pad_rows, hidden_size), dtype=group.dtype),
                ],
                dim=0,
            )
        group_weight, group_scales = scaled_quantize(
            group,
            GroupShape(block, block),
            torch.float8_e4m3fn,
            compute_dtype=torch.float32,
        )
        weights.append(group_weight[:rows_per_group])
        scales.append(group_scales)

    return torch.cat(weights), torch.cat(scales)


def _expected_rank_dequant(
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_rank: int,
    tp_size: int,
    block: int,
) -> torch.Tensor:
    kv_heads_per_rank = num_kv_heads // tp_size
    q_rows_per_group = (num_heads // num_kv_heads) * head_dim
    k_rows_per_group = head_dim
    rows_per_group = _qkv_rows_per_group(
        num_heads, num_kv_heads, head_dim, v_head_dim
    )
    scale_rows_per_group = scales.shape[0] // num_kv_heads

    qs: list[torch.Tensor] = []
    ks: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    for group_idx in range(
        tp_rank * kv_heads_per_rank, (tp_rank + 1) * kv_heads_per_rank
    ):
        row_start = group_idx * rows_per_group
        scale_start = group_idx * scale_rows_per_group
        group_weight = weight[row_start : row_start + rows_per_group].float()
        group_scales = scales[
            scale_start : scale_start + scale_rows_per_group
        ].float()
        group_dequant = group_weight * _expanded_scales(
            group_scales, group_weight.shape, block
        )
        qs.append(group_dequant[:q_rows_per_group])
        ks.append(
            group_dequant[
                q_rows_per_group : q_rows_per_group + k_rows_per_group
            ]
        )
        vs.append(group_dequant[q_rows_per_group + k_rows_per_group :])

    return torch.cat([torch.cat(qs), torch.cat(ks), torch.cat(vs)], dim=0)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="MiMo FP8 QKV sharding requires torch.float8_e4m3fn",
)
def test_mimo_v2_fp8_qkv_tp_sharding_reorders_multiple_kv_groups():
    torch.manual_seed(7)

    block = 128
    num_heads = 128
    num_kv_heads = 8
    head_dim = 192
    v_head_dim = 128
    tp_size = 4
    hidden_size = 128
    weight, scales = _make_pro_format_fp8_qkv(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        hidden_size=hidden_size,
        block=block,
    )

    assert weight.shape == (27136, 128)
    assert scales.shape == (216, 1)

    for tp_rank in range(tp_size):
        rank_weight, rank_scales = _shard_fp8_qkv_proj(
            weight,
            scales,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            block=block,
        )
        assert rank_weight.shape == (6784, 128)
        assert rank_scales.shape == (53, 1)

        actual = scaled_dequantize(
            rank_weight, rank_scales, GroupShape(block, block)
        )
        expected = _expected_rank_dequant(
            weight,
            scales,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            block=block,
        )

        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="MiMo FP8 QKV loading requires torch.float8_e4m3fn",
)
def test_mimo_v2_fp8_qkv_loader_pairs_weight_and_scale():
    torch.manual_seed(7)

    block = 128
    num_heads = 128
    num_kv_heads = 8
    head_dim = 192
    v_head_dim = 128
    tp_size = 4
    hidden_size = 128
    weight, scales = _make_pro_format_fp8_qkv(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        hidden_size=hidden_size,
        block=block,
    )

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = SimpleNamespace(
                total_num_heads=num_heads,
                total_num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                v_head_dim=v_head_dim,
            )

        def get_submodule(self, target: str):
            assert target == "layers.0.self_attn"
            return self.attn

    pending: dict[str, dict[str, torch.Tensor]] = {}
    loaded: set[str] = set()
    params = {
        "layers.0.self_attn.qkv_proj.weight": nn.Parameter(
            torch.empty((6784, 128), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
        "layers.0.self_attn.qkv_proj.weight_scale_inv": nn.Parameter(
            torch.empty((53, 1), dtype=torch.float32),
            requires_grad=False,
        ),
    }

    consumed = MiMoV2Model._try_load_fp8_qkv_proj(
        FakeModel(),
        "layers.0.self_attn.qkv_proj.weight",
        weight,
        pending,
        params,
        loaded,
        tp_rank=0,
        tp_size=tp_size,
    )
    assert consumed
    assert pending
    assert not loaded

    consumed = MiMoV2Model._try_load_fp8_qkv_proj(
        FakeModel(),
        "layers.0.self_attn.qkv_proj.weight_scale_inv",
        scales,
        pending,
        params,
        loaded,
        tp_rank=0,
        tp_size=tp_size,
    )
    assert consumed
    assert not pending
    assert loaded == set(params)

    actual = scaled_dequantize(
        params["layers.0.self_attn.qkv_proj.weight"].data,
        params["layers.0.self_attn.qkv_proj.weight_scale_inv"].data,
        GroupShape(block, block),
    )
    expected = _expected_rank_dequant(
        weight,
        scales,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        tp_rank=0,
        tp_size=tp_size,
        block=block,
    )
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0)
