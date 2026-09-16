# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.token_to_req import scatter_token_to_req_indices

if not (current_platform.is_cuda() or current_platform.is_rocm()):
    pytest.skip(
        "The token-to-request kernel requires CUDA or ROCm.", allow_module_level=True
    )


@torch.inference_mode()
def test_scatter_token_to_req_indices_handles_empty_requests_and_padding():
    query_start_loc = torch.tensor([0, 2, 2, 5, 5], dtype=torch.int32, device="cuda")
    output = torch.full((8,), -1, dtype=torch.int32, device="cuda")

    actual = scatter_token_to_req_indices(
        query_start_loc,
        output,
        num_reqs=4,
        num_tokens=8,
        max_query_len=3,
    )

    expected = torch.tensor([0, 0, 2, 2, 2, 0, 0, 0], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(actual, expected)
