# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A stacked parameter that received only some of its shards must be reported.

`track_weights_loading` diffs parameter names, so `qkv_proj.weight` counts as
loaded once any one shard is written. These tests pin the shard-level check
that closes that gap for `QKVParallelLinear` and `MergedColumnParallelLinear`.
"""

import pytest
import torch
from torch import nn

from vllm.config import LoadConfig
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

HIDDEN = 8
HEAD = 2

pytestmark = [pytest.mark.cpu_test, pytest.mark.usefixtures("dist_init")]


def shard(shape: tuple[int, ...], shard_id) -> torch.Tensor:
    weight = torch.ones(shape)
    weight.shard_id = shard_id
    return weight


def load(layer: nn.Module, *weights: tuple[str, torch.Tensor]) -> set[str]:
    return set(layer.load_weights(weights))


def make_qkv(bias: bool = False, total_num_heads: int = 2) -> QKVParallelLinear:
    return QKVParallelLinear(
        HIDDEN, HEAD, total_num_heads, total_num_kv_heads=1, bias=bias
    )


Q = ("weight", shard((2 * HEAD, HIDDEN), "q"))
K = ("weight", shard((HEAD, HIDDEN), "k"))
V = ("weight", shard((HEAD, HIDDEN), "v"))


def test_qkv_reports_shards_the_checkpoint_never_filled():
    layer = make_qkv()
    load(layer, Q)
    assert layer.pop_missing_shards() == {"weight": ["k", "v"]}

    load(layer, Q, K, V)
    assert layer.pop_missing_shards() == {}


def test_qkv_stacked_on_disk_fills_every_shard():
    layer = make_qkv()
    load(layer, ("weight", torch.ones(4 * HEAD, HIDDEN)))
    assert layer.pop_missing_shards() == {}


def test_record_is_per_load_so_a_reload_cannot_coast_on_the_first():
    layer = make_qkv()
    load(layer, Q, K, V)
    assert layer.pop_missing_shards() == {}

    load(layer, K, V)
    assert layer.pop_missing_shards() == {"weight": ["q"]}


def test_shard_with_no_rows_is_never_expected():
    """Cross-attention kv_proj is a QKV layer with zero query heads."""
    layer = make_qkv(total_num_heads=0)
    load(layer, K, V)
    assert layer.pop_missing_shards() == {}


def test_each_parameter_is_tracked_on_its_own():
    layer = make_qkv(bias=True)
    load(layer, ("weight", torch.ones(4 * HEAD, HIDDEN)))
    assert layer.pop_missing_shards() == {}, "untouched bias is the name check's job"

    load(layer, ("bias", shard((2 * HEAD,), "q")))
    assert layer.pop_missing_shards() == {"bias": ["k", "v"]}


def test_merged_column_tuple_shard_is_one_span():
    layer = MergedColumnParallelLinear(HIDDEN, [4, 4, 4], bias=False)
    load(layer, ("weight", shard((8, HIDDEN), (0, 1))))
    assert layer.pop_missing_shards() == {"weight": [2]}

    load(
        layer, ("weight", shard((8, HIDDEN), (0, 1))), ("weight", shard((4, HIDDEN), 2))
    )
    assert layer.pop_missing_shards() == {}


def test_loader_rejects_a_partially_loaded_stacked_weight():
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.qkv_proj = make_qkv()
    loaded = {f"layer.qkv_proj.{name}" for name in load(model.layer.qkv_proj, Q)}

    loader = DefaultModelLoader(LoadConfig())
    with pytest.raises(ValueError, match=r"layer\.qkv_proj\.weight.*\['k', 'v'\]"):
        loader.track_weights_loading(model, loaded)
