# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("cutlass")

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    _build_topk_mask,
    _topk_mask_shape,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_topk_mask_single_request_matches_generic_path() -> None:
    topk = torch.tensor(
        [[0, 31, 32, 63, -1], [1, 64, 127, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )

    num_words = (128 + 31) // 32
    single_out = torch.zeros(1, 2, num_words, dtype=torch.int32, device="cuda")
    generic_out = torch.zeros(2, 1, num_words, dtype=torch.int32, device="cuda")
    single_req = _build_topk_mask([topk], [2], 2, 128, single_out)
    generic = _build_topk_mask([topk[:1], topk[1:]], [1, 1], 1, 128, generic_out)

    torch.testing.assert_close(single_req[0, 0], generic[0, 0])
    torch.testing.assert_close(single_req[0, 1], generic[1, 0])


def test_topk_mask_rows_are_aligned_for_vectorized_loads() -> None:
    assert _topk_mask_shape(2, 129, 129) == (2, 256, 8)
    assert _topk_mask_shape(2, 129, 129, reserve_key_starts_word=True) == (
        2,
        256,
        8,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_topk_mask_preserves_aligned_row_storage() -> None:
    shape = _topk_mask_shape(1, 1, 129)
    out = torch.zeros(shape, dtype=torch.int32, device="cuda")
    topk = torch.tensor([[0, 128]], dtype=torch.int32, device="cuda")

    mask = _build_topk_mask([topk], [1], 1, 129, out)

    assert mask.shape == (1, 1, 8)
    assert mask[0, 0].tolist() == [1, 0, 0, 0, 1, 0, 0, 0]
