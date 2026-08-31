# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    triton_convert_req_index_to_global_index,
)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only conversion")
def test_rocm_sparse_conversion_uses_physical_stride_and_preserves_invalid() -> None:
    device = torch.device("cuda")
    req_id = torch.tensor([0], dtype=torch.int32, device=device)
    block_table = torch.tensor([[3, 7]], dtype=torch.int32, device=device)
    token_indices = torch.tensor([[0, 15, 16, -1]], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
    output = torch.empty(4, dtype=torch.int32, device=device)

    triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        indptr,
        output,
        BLOCK_SIZE=16,
        BLOCK_STRIDE_ROWS=64,
        NUM_TOPK_TOKENS=4,
        BLOCK_N=4,
    )

    expected = torch.tensor([192, 207, 448, -1], dtype=torch.int32, device=device)
    torch.testing.assert_close(output, expected)
