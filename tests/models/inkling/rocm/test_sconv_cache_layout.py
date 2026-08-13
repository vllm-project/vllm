# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.inkling.amd.sconv_swa_attn import InklingConvState


def test_runtime_sconv_block_size_tracks_unified_cache_page():
    """The cache planner may enlarge W=4 blocks to match attention pages."""
    owner = InklingConvState.__new__(InklingConvState)
    nn.Module.__init__(owner)
    owner.block_size = 4

    owner.kv_cache = torch.tensor([])
    assert owner.cache_block_size == 4

    owner.kv_cache = torch.empty(2, 1, 32, 1024)
    assert owner.cache_block_size == 32
