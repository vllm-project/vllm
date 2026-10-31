# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused QKV+O-gate projection loads through the parameter's
``weight_loader``, so a layerwise reload (which restores layers on meta and
replays wrapped loaders) sees the writes and takes the new weights."""

import torch
from torch import nn

from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.muse_spark.common.layers import MuseSpark1xQKVOGateParallelLinear

HIDDEN, HEAD, HEADS, KV_HEADS = 8, 4, 2, 1


def _checkpoint(seed: int) -> list[tuple[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    weights = []
    for shard_id, heads in (("q", HEADS), ("k", KV_HEADS), ("v", KV_HEADS)):
        weight = torch.randn(heads * HEAD, HIDDEN, generator=gen)
        weight.shard_id = shard_id
        weights.append(("weight", weight))
    o_gate = torch.randn(HEADS * HEAD, HIDDEN, generator=gen)
    o_gate.shard_id = "o_gate"
    weights.append(("weight", o_gate))
    return weights


def test_qkvo_reload_takes_new_weights(dist_init):
    proj = MuseSpark1xQKVOGateParallelLinear(
        HIDDEN, HEAD, HEADS, KV_HEADS, tp_rank=0, tp_size=1
    )
    model = nn.Sequential(proj)
    assert list(proj.load_weights(_checkpoint(seed=0))) == ["weight"] * 4
    record_metadata_for_reloading(model)
    data_ptr = proj.weight.data_ptr()

    initialize_layerwise_reload(model)
    list(proj.load_weights(_checkpoint(seed=1)))
    finalize_layerwise_reload(model, model_config=None)

    expected = torch.cat([w for _, w in _checkpoint(seed=1)])
    assert torch.equal(proj.weight.data, expected)
    assert proj.weight.data_ptr() == data_ptr
