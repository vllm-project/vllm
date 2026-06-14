# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4 import sparse_mla
from vllm.models.deepseek_v4.nvidia import flashmla


def test_c128a_effective_topk_width_uses_current_positions() -> None:
    assert (
        sparse_mla._c128a_effective_topk_width(
            positions=torch.tensor([0, 126], dtype=torch.int64),
            compress_ratio=128,
            max_compressed_tokens=4096,
            alignment=128,
        )
        == 128
    )
    assert (
        sparse_mla._c128a_effective_topk_width(
            positions=torch.tensor([127, 1023], dtype=torch.int64),
            compress_ratio=128,
            max_compressed_tokens=4096,
            alignment=128,
        )
        == 128
    )
    assert (
        sparse_mla._c128a_effective_topk_width(
            positions=torch.tensor([524287], dtype=torch.int64),
            compress_ratio=128,
            max_compressed_tokens=8192,
            alignment=128,
        )
        == 4096
    )
    assert (
        sparse_mla._c128a_effective_topk_width(
            positions=torch.tensor([1048575], dtype=torch.int64),
            compress_ratio=128,
            max_compressed_tokens=8192,
            alignment=128,
        )
        == 8192
    )


def test_indexed_d512_split_topk_keeps_small_c128a_prefills() -> None:
    assert not flashmla._is_indexed_d512_split_topk(128)
    assert flashmla._is_indexed_d512_split_topk(256)
    assert flashmla._is_indexed_d512_split_topk(512)
    assert flashmla._is_indexed_d512_split_topk(1152)
    assert not flashmla._is_indexed_d512_split_topk(1280)
