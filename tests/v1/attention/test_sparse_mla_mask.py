# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("cutlass")

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    _build_topk_mask,
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
