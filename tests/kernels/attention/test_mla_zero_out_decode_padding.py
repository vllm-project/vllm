# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.v1.attention.backends.mla.utils import zero_out_decode_padding


def _assert_zero_out_matches_ref(
    *,
    num_tokens: int,
    num_cols: int,
    pad_positions: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
    num_heads: int = 3,
) -> None:
    out = torch.randn(num_tokens, 3, num_cols, dtype=dtype, device="cuda")
    seq_lens = torch.ones(num_tokens, dtype=torch.int32, device="cuda")
    if pad_positions:
        pad_indices = torch.tensor(pad_positions, dtype=torch.long, device="cuda")
        seq_lens[pad_indices] = 0
        # Match production behavior: padded rows may contain NaNs.
        out[pad_indices] = torch.nan

    ref = out.clone()
    ref[seq_lens == 0] = 0

    zero_out_decode_padding(out, seq_lens)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 8])
@pytest.mark.parametrize("num_cols", [257, 1024, 1500])
def test_zero_out_padding_exhaustive(num_tokens: int, num_cols: int):
    if num_tokens == 1:
        _assert_zero_out_matches_ref(
            num_tokens=1,
            num_cols=num_cols,
            pad_positions=(),
        )
        return

    for pad_start in range(1, num_tokens):
        _assert_zero_out_matches_ref(
            num_tokens=num_tokens,
            num_cols=num_cols,
            pad_positions=tuple(list(range(pad_start, num_tokens))),
        )


@pytest.mark.parametrize("num_tokens", [4, 5, 10, 13, 25] + list(range(55, 64)))
@pytest.mark.parametrize("num_cols", [257])
def test_zero_out_padding(num_tokens: int, num_cols: int) -> None:
    _assert_zero_out_matches_ref(
        num_tokens=num_tokens,
        num_cols=num_cols,
        pad_positions=(num_tokens - 2, num_tokens - 1),
    )
