# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block scales of the fused GDP in_proj are sharded like its weight under TP
(quantized checkpoints once loaded another rank's scales)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.nemotron_shortcut.nvidia.gdp import install_gdp_in_proj_loaders

K = 64
SCALE_COLS = K // 32

# The NemotronShortcutGDPMixer packing, [gate, value, key, query, beta, decay], for two
# householders and small dimensions: (rows, num_householder) per block.
BLOCKS = [(32, 1), (64, 2), (32, 2), (16, 1), (16, 2), (8, 1)]
ROWS = sum(rows for rows, _ in BLOCKS)


def _row_coded(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    # Every element of row i holds i, so a loaded row says which source row it is.
    return torch.arange(rows).unsqueeze(1).expand(rows, cols).to(dtype).contiguous()


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_in_proj_scales_follow_the_weight_rows(tp_size, scale_name):
    shard = ROWS // tp_size
    for tp_rank in range(tp_size):
        scale = nn.Parameter(
            torch.empty(shard, SCALE_COLS, dtype=torch.int32), requires_grad=False
        )
        in_proj = SimpleNamespace(
            weight=nn.Parameter(torch.empty(shard, K), requires_grad=False),
            **{scale_name: scale},
        )
        install_gdp_in_proj_loaders(in_proj, BLOCKS, tp_size, tp_rank)

        weight = _row_coded(ROWS, K, torch.float32)
        in_proj.weight.weight_loader(in_proj.weight, weight)
        scale.weight_loader(scale, _row_coded(ROWS, SCALE_COLS, torch.int32))

        # Row i of the rank's scales belongs to row i of its weight.
        torch.testing.assert_close(
            scale.data, in_proj.weight.data[:, :SCALE_COLS].to(torch.int32)
        )
