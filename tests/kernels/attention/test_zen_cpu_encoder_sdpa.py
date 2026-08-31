# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for encoder attention routed through the zentorch SDPA kernel."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.zentorch_sdpa import (
    should_use_zentorch_sdpa,
    zentorch_encoder_sdpa,
)

if not current_platform.is_cpu() or not current_platform.is_zen_cpu():
    pytest.skip("skipping non-Zen CPU tests", allow_module_level=True)

if not has_zentorch_op(["zentorch_sdpa"]):
    pytest.skip(
        "skipping tests: zentorch_sdpa op not available",
        allow_module_level=True,
    )

from tests.kernels.attention.test_cpu_attn import ref_varlen_encoder_attn  # noqa: E402

ATOL, RTOL = 1.5e-2, 1e-2

UNIFORM_SEQ_LENS = [128, 128, 128]
RAGGED_SEQ_LENS = [1, 67, 233, 5]


def _metadata(seq_lens: list[int]) -> SimpleNamespace:
    start_loc = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32), 0, out=start_loc[1:])
    return SimpleNamespace(query_start_loc=start_loc)


@pytest.mark.parametrize("seq_lens", [UNIFORM_SEQ_LENS, RAGGED_SEQ_LENS])
@pytest.mark.parametrize("num_heads", [(8, 2), (4, 4)])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_encoder_sdpa_matches_reference(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
) -> None:
    """Uniform batches take the dense path, ragged ones the per-sequence loop."""
    set_random_seed(0)
    tokens = sum(seq_lens)
    query = torch.randn(tokens, num_heads[0], head_size, dtype=dtype)
    key = torch.randn(tokens, num_heads[1], head_size, dtype=dtype)
    value = torch.randn(tokens, num_heads[1], head_size, dtype=dtype)
    output = torch.empty_like(query)
    scale = head_size**-0.5

    zentorch_encoder_sdpa(query, key, value, output, _metadata(seq_lens), scale)

    ref_output = ref_varlen_encoder_attn(
        query=query,
        key=key,
        value=value,
        seq_lens=seq_lens,
        scale=scale,
        sliding_window=None,
    )
    torch.testing.assert_close(output, ref_output, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "attn_type,alibi_slopes,sliding_window,dtype,expected",
    [
        (AttentionType.ENCODER_ONLY, None, -1, torch.bfloat16, True),
        (AttentionType.ENCODER, None, None, torch.float32, True),
        # zentorch_sdpa takes no bias, so a masked layer stays native.
        (AttentionType.ENCODER_ONLY, torch.ones(4), -1, torch.bfloat16, False),
        (AttentionType.ENCODER_ONLY, None, 256, torch.bfloat16, False),
        # Only encoder attention skips the KV cache.
        (AttentionType.DECODER, None, -1, torch.bfloat16, False),
        (AttentionType.ENCODER_DECODER, None, -1, torch.bfloat16, False),
        # No zentorch kernel for these dtypes.
        (AttentionType.ENCODER_ONLY, None, -1, torch.float16, False),
    ],
)
def test_should_use_zentorch_sdpa(
    attn_type: str,
    alibi_slopes: torch.Tensor | None,
    sliding_window: int | None,
    dtype: torch.dtype,
    expected: bool,
) -> None:
    assert (
        should_use_zentorch_sdpa(attn_type, alibi_slopes, sliding_window, dtype)
        is expected
    )
