# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.models.hy_v4_flash.nvidia.engram import FlashEngramLayout


def test_ids_past_lookup_table_hash_as_pad():
    """Ids past the frozen map read as the pad id, as in the reference."""
    config = HYV4FlashConfig(
        vocab_size=16,
        engram_layer_ids=[2],
        engram_n_head_per_ngram=1,
        engram_n_embed_per_ngram=8,
        engram_pad_id=3,
        engram_layer_head_vocab_sizes={"2": [[5], [7]]},
    )
    layout = FlashEngramLayout(config)
    lookup = torch.tensor([0, 1, 1, 2, 3, 3, 4, 4, 5, 6])
    assert layout.load_lookup(lookup)

    assert layout.token_map == lookup.tolist() + [2] * 6
    # Every compressed id stays within the bound the multipliers are drawn for.
    bound = np.iinfo(np.int64).max
    assert max(layout.token_map) * int(layout.multipliers.max()) <= bound
