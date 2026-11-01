# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVIDIA runtime conv-block-size tracking (see the ROCm twin test).

The KV-cache planner enlarges the conv group's block size (W=4 -> e.g. 8)
when the attention page is a multiple of the conv page
(``unify_kv_cache_spec_page_size``). Slot mappings are then built with the
enlarged size, so NVIDIA call sites must index with the *runtime* block size,
not the kernel window size.
"""

import torch
from torch import nn

from vllm.models.inkling.nvidia.sconv_swa_attn import InklingConvState


def test_runtime_sconv_block_size_tracks_unified_cache_page():
    owner = InklingConvState.__new__(InklingConvState)
    nn.Module.__init__(owner)
    owner.block_size = 4

    owner.kv_cache = torch.tensor([])
    assert owner.cache_block_size == 4

    owner.kv_cache = torch.empty(2, 1, 32, 1024)
    assert owner.cache_block_size == 32
