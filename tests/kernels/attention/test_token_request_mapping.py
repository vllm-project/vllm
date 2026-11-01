# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices

DEVICE = current_platform.device_type

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
        reason="token request mapping requires a Triton-capable device.",
    ),
]


def _reference(
    query_start_loc: torch.Tensor, num_mapped: int, num_tokens: int
) -> torch.Tensor:
    tokens = torch.arange(num_tokens, device=query_start_loc.device, dtype=torch.int64)
    req = torch.searchsorted(query_start_loc.long(), tokens, right=True) - 1
    req = req.clamp(0, query_start_loc.numel() - 2)
    return torch.where(tokens < num_mapped, req, 0).to(torch.int32)


def _run(lengths: torch.Tensor, num_mapped: int):
    qsl = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)]).to(DEVICE)
    num_tokens = int(qsl[-1])
    out = torch.full((num_tokens + 64,), -1, dtype=torch.int32, device=DEVICE)
    got = compute_token_to_req_indices(qsl, out, num_mapped, num_tokens)
    expected = _reference(qsl, num_mapped, num_tokens)
    return got, expected, out, num_tokens


@pytest.mark.parametrize("num_reqs", [1, 2, 7, 33, 256, 1024])
@pytest.mark.parametrize("tokens_per_req", [1, 5, 64])
@pytest.mark.parametrize("zero_len", [False, True])
@pytest.mark.parametrize("mapped_ratio", [1.0, 0.75])
def test_token_request_mapping(
    num_reqs: int, tokens_per_req: int, zero_len: bool, mapped_ratio: float
):
    lengths = torch.full((num_reqs,), tokens_per_req, dtype=torch.int32)
    if zero_len and num_reqs > 1:
        lengths[::3] = 0
    if int(lengths.sum()) == 0:
        lengths[0] = 1
    num_mapped = max(0, int(int(lengths.sum()) * mapped_ratio))

    got, expected, out, num_tokens = _run(lengths, num_mapped)
    assert torch.equal(got, expected)
    # The kernel must not write past num_tokens.
    assert torch.all(out[num_tokens:] == -1)


@pytest.mark.parametrize("num_reqs", [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 1023, 1024])
def test_token_request_mapping_search_depth(num_reqs: int):
    # One token per request exercises every binary-search depth boundary.
    lengths = torch.ones(num_reqs, dtype=torch.int32)
    got, expected, _, _ = _run(lengths, num_reqs)
    assert torch.equal(got, expected)


def test_token_request_mapping_non_uniform():
    lengths = torch.tensor([1, 7, 0, 3, 200, 0, 64, 5], dtype=torch.int32)
    got, expected, _, _ = _run(lengths, int(lengths.sum()))
    assert torch.equal(got, expected)
