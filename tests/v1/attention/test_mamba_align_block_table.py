# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for the fused mamba "align" block-table gather.

`mamba_get_block_table_tensor`'s align branch used to be five PyTorch
statements that expanded to seven kernel launches over a `(num_reqs,)` tensor.
It is now one Triton kernel. These tests pin the fused kernel to the exact
semantics of the expression it replaced, including cases a single-request
decode never exercises but other callers can hit.

`block_table` is deliberately tested non-contiguous: the original
`torch.gather(block_table, 1, ...)` honours any layout, and this is a shared
utility (`gdn_attn`, `mamba_attn`, `short_conv_attn` and `linear_attn` all call
it), so a column stride of 1 must not be assumed.
"""

import pytest
import torch

from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

WIDTH = 64


def _spec(block_size: int, num_speculative_blocks: int) -> MambaSpec:
    # Only block_size and num_speculative_blocks are read by the align branch;
    # shapes/dtypes just have to be well formed for the frozen dataclass.
    return MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        num_speculative_blocks=num_speculative_blocks,
    )


def _reference(block_table, seq_lens, block_size, num_spec_blocks):
    """The align branch exactly as it was before the fusion."""
    start_indices = (seq_lens - 1) // block_size
    start_indices = start_indices.clamp(min=0)
    offsets = torch.arange(
        1 + num_spec_blocks, device=block_table.device, dtype=torch.int32
    )
    indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)
    return torch.gather(block_table, 1, indices_to_gather)


def _make(nreq, block_size, nspec, device, noncontig=False):
    torch.manual_seed(0)
    width = WIDTH * 2 if noncontig else WIDTH
    bt = torch.randint(0, 5000, (nreq, width), dtype=torch.int32, device=device)
    if noncontig:
        # Every other column: stride(1) == 2, which a kernel assuming a unit
        # column stride would read straight through, silently returning the
        # wrong blocks.
        bt = bt[:, ::2]
        assert not bt.is_contiguous()
    # Keep start + nspec inside the row, which the gather already requires.
    hi = max(1, (WIDTH - 1 - nspec) * block_size)
    sl = torch.randint(0, hi, (nreq,), dtype=torch.int32, device=device)
    return bt, sl


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("nreq", [1, 2, 8, 64])
@pytest.mark.parametrize("block_size", [16, 64, 256])
@pytest.mark.parametrize("nspec", [0, 1, 3, 7])
def test_matches_reference(nreq, block_size, nspec):
    bt, sl = _make(nreq, block_size, nspec, "cuda")
    got = mamba_get_block_table_tensor(bt, sl, _spec(block_size, nspec), "align")
    want = _reference(bt, sl, block_size, nspec)
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("nspec", [0, 3, 7])
def test_non_contiguous_block_table(nspec):
    """A column stride != 1 must be honoured, not assumed away."""
    bt, sl = _make(8, 64, nspec, "cuda", noncontig=True)
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, nspec), "align")
    want = _reference(bt, sl, 64, nspec)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_zero_seq_len_clamps_to_first_row():
    """CUDA-graph padding rows carry seq_len 0; the old code clamped to row 0."""
    bt = torch.randint(0, 5000, (4, WIDTH), dtype=torch.int32, device="cuda")
    sl = torch.zeros(4, dtype=torch.int32, device="cuda")
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "align")
    torch.testing.assert_close(got, _reference(bt, sl, 64, 3), atol=0, rtol=0)
    # start clamps to 0, so this is just the first 1 + nspec columns.
    torch.testing.assert_close(got, bt[:, :4], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_zero_requests_returns_empty():
    """An empty batch must not launch a (0,) grid."""
    bt = torch.zeros((0, WIDTH), dtype=torch.int32, device="cuda")
    sl = torch.zeros(0, dtype=torch.int32, device="cuda")
    got = mamba_get_block_table_tensor(bt, sl, _spec(64, 3), "align")
    assert got.shape == (0, 4)
    assert got.dtype == bt.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("mode", ["all", "none"])
def test_passthrough_modes_return_input(mode):
    bt = torch.randint(0, 5000, (4, WIDTH), dtype=torch.int32, device="cuda")
    sl = torch.randint(0, 1000, (4,), dtype=torch.int32, device="cuda")
    assert mamba_get_block_table_tensor(bt, sl, _spec(64, 3), mode) is bt
