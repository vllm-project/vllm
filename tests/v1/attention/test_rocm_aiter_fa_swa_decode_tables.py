# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionMetadataBuilder


def test_swa_decode_tables_match_reference():
    # window and block edges, and a window start inside a block
    window, block_size = 2048, 16
    lens = torch.tensor(
        [*range(1, 3 * window, 37), window, window + 1, 200000], dtype=torch.int32
    )
    n, width = len(lens), cdiv(window, block_size) + 1
    full = torch.randperm(n * (cdiv(int(lens.max()), block_size) + 3)).view(n, -1)
    full = full.to(torch.int32)
    builder = SimpleNamespace(
        aot_sliding_window=(window - 1, 0),
        block_size=block_size,
        _swa_block_table=torch.zeros(n, width, dtype=torch.int32),
        _swa_seq_lens=torch.zeros(n, dtype=torch.int32),
        _swa_block_offsets=torch.arange(width),
    )
    table, seq_lens, max_seq_len = (
        AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
            builder, full, lens, n, int(lens.max())
        )
    )

    assert table is not None and seq_lens is not None
    first = (lens - window).clamp(min=0) // block_size
    index = (first[:, None] + torch.arange(width)).clamp(max=full.shape[1] - 1)
    assert torch.equal(table, full.gather(1, index))
    assert torch.equal(seq_lens, lens - first * block_size)
    assert seq_lens.dtype == torch.int32
    assert max_seq_len == window + block_size - 1
