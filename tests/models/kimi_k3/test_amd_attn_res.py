# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD AttnRes requires ROCm",
)


def _randn_with_row_padding(*shape: int, padding: int = 0) -> torch.Tensor:
    storage = torch.randn(
        *shape[:-1],
        shape[-1] + padding,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return storage[..., : shape[-1]]


def _reference(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    hidden_size = prefix.shape[-1]
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (hidden_size,), norm_weight, eps)
    probs = (keys @ qk_weight).softmax(dim=-1)
    return torch.matmul(probs.unsqueeze(1), values).squeeze(1)


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "block_capacity",
        "hidden_size",
        "row_padding",
    ),
    [
        pytest.param(0, 3, 5, 128, 0, id="empty"),
        pytest.param(1, 1, 2, 128, 0, id="decode-single"),
        pytest.param(17, 4, 6, 1024, 7, id="decode-padded"),
        pytest.param(320, 8, 10, 7168, 0, id="prefill-full"),
    ],
)
def test_amd_attn_res_matches_reference(
    num_tokens: int,
    num_blocks: int,
    block_capacity: int,
    hidden_size: int,
    row_padding: int,
) -> None:
    eps = 1e-5
    prefix = _randn_with_row_padding(num_tokens, hidden_size, padding=row_padding)
    blocks = _randn_with_row_padding(
        num_tokens,
        block_capacity,
        hidden_size,
        padding=row_padding,
    )
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    )
    expected = _reference(
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        num_blocks,
        eps,
    )
    original_prefix = prefix.clone()
    original_blocks = blocks.clone()

    actual = attn_res(
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        num_blocks,
        eps,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, original_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.shape == prefix.shape
    assert actual.is_contiguous()
