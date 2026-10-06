# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Triton ``neutralize_dcp_empty_rows_`` kernel must be bit-exact with the
eager ``all`` + ``masked_fill_`` sequence it replaces."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_utils import neutralize_dcp_empty_rows_

requires_accelerator = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="needs a CUDA or ROCm device for the Triton kernel",
)


def _reference_neutralize(
    out: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    empty_rows = (topk_indices == -1).all(dim=-1)
    out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
    lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))


@requires_accelerator
@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "head_dim"),
    [
        (4, 64, 512),
        (1, 8, 64),
        (42, 28, 320),
        (1111, 28, 576),
        (1024, 128, 512),
    ],
)
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("empty_pattern", ["none", "all", "mixed"])
def test_neutralize_dcp_empty_rows_matches_torch(
    num_tokens: int, num_heads: int, head_dim: int, topk: int, empty_pattern: str
) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)

    topk_indices = torch.randint(
        0,
        4096,
        (num_tokens, topk),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    topk_indices[topk_indices % 3 == 0] = -1
    if empty_pattern == "all":
        topk_indices[:] = -1
    elif empty_pattern == "mixed":
        topk_indices[::2] = -1
        topk_indices[1::2, 0] = 7

    out = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)
    expected_out, expected_lse = out.clone(), lse.clone()
    _reference_neutralize(expected_out, expected_lse, topk_indices)

    neutralize_dcp_empty_rows_(out, lse, topk_indices)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0, equal_nan=True)


@requires_accelerator
def test_neutralize_dcp_empty_rows_honours_strides() -> None:
    """FlashMLA sparse hands over a head-padded ``out`` and a transposed
    ``lse`` view; padding heads must stay untouched."""
    device = torch.device("cuda")
    num_tokens, num_heads, head_dim, padded_heads = 4, 8, 64, 64

    padded = torch.randn(
        num_tokens, padded_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    out = padded[:, :num_heads, :]
    assert not out.is_contiguous()

    lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device=device).t()
    assert not lse.is_contiguous()

    topk_indices = torch.full((num_tokens, 128), -1, dtype=torch.int32, device=device)
    topk_indices[1, 5] = 3

    padded_before = padded.clone()
    expected_out, expected_lse = out.clone(), lse.clone()
    _reference_neutralize(expected_out, expected_lse, topk_indices)

    neutralize_dcp_empty_rows_(out, lse, topk_indices)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        padded[:, num_heads:, :], padded_before[:, num_heads:, :], rtol=0, atol=0
    )
