# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    mamba_v2_sharded_weight_loader,
)


def _index_assignment_loader(shard_spec, tp_size, tp_rank):
    """The loader before it switched to ``copy_``."""

    def loader(param, loaded_weight):
        boundary, loaded_boundary = 0, 0
        for full_dim, extra, duplicate_groups in shard_spec:
            shard_size = full_dim // tp_size
            loaded_skip = (0 if duplicate_groups else tp_rank) * shard_size
            start = loaded_boundary + loaded_skip
            take = min(shard_size, full_dim - extra - loaded_skip)
            target = param.data[boundary : boundary + take]
            param.data[boundary : boundary + take] = loaded_weight[
                start : start + take
            ].view_as(target)
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


# (full_dim, extra, duplicate_groups) per segment, e.g. x / B / C of conv1d.
@pytest.mark.parametrize(
    "shard_spec,tp_size",
    [
        ([(16, 0, False), (8, 0, False), (8, 0, False)], 2),
        # n_groups == 1 replicated across ranks
        ([(16, 0, False), (8, 4, True), (8, 4, True)], 4),
    ],
)
@pytest.mark.parametrize("trailing", [(), (3,)])  # bias / conv1d weight
def test_loader_matches_index_assignment(shard_spec, tp_size, trailing):
    full = sum(dim - extra for dim, extra, _ in shard_spec)
    local = sum(dim // tp_size for dim, _, _ in shard_spec)
    loaded = torch.randn(full, *trailing)
    for tp_rank in range(tp_size):
        expected = torch.nn.Parameter(torch.full((local, *trailing), -7.0))
        got = torch.nn.Parameter(torch.full((local, *trailing), -7.0))
        _index_assignment_loader(shard_spec, tp_size, tp_rank)(expected, loaded)
        mamba_v2_sharded_weight_loader(shard_spec, tp_size, tp_rank)(got, loaded)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
