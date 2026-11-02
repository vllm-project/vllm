# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices

LENGTHS = ([1], [1, 1], [1] * 7, [1] * 8, [5] * 33, [5] * 1024, [1, 7, 0, 3])

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
        reason="Triton kernel",
    ),
]


@pytest.mark.parametrize("lengths", LENGTHS)
@pytest.mark.parametrize("mapped_ratio", [1.0, 0.75])
def test_mapping(lengths, mapped_ratio):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    qsl = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)]).to(
        current_platform.device_type
    )
    num_tokens = int(qsl[-1])
    num_mapped = int(num_tokens * mapped_ratio)
    out = torch.full((num_tokens + 64,), -1, dtype=torch.int32, device=qsl.device)
    got = compute_token_to_req_indices(qsl, out, num_mapped, num_tokens)
    tokens = torch.arange(num_tokens, device=qsl.device, dtype=torch.int64)
    expected = torch.searchsorted(qsl.long(), tokens, right=True) - 1
    expected = torch.where(tokens < num_mapped, expected.clamp(0, qsl.numel() - 2), 0)
    assert torch.equal(got, expected.int())
    assert torch.all(out[num_tokens:] == -1)
