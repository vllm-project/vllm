# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused vocab-parallel argmax must match torch.argmax on the full logits.

TP ranks are emulated in one process by concatenating the per-shard partials
in rank order, which is what ``all_gather(dim=0)`` produces.
"""

import pytest
import torch

from vllm.model_executor.layers.vocab_parallel_argmax import (
    local_argmax_partials,
    merge_argmax_partials,
)

if not torch.cuda.is_available():
    pytest.skip("GPU required for Triton kernels", allow_module_level=True)

# Fills the padded columns past the vocabulary; it would win if not masked.
PAD_VALUE = 1e30


def _sharded_argmax(logits: torch.Tensor, tp_size: int, pad_multiple: int = 64):
    num_rows, vocab_size = logits.shape
    padded = -(-vocab_size // pad_multiple) * pad_multiple
    width = -(-padded // tp_size)
    full = torch.full(
        (num_rows, width * tp_size), PAD_VALUE, dtype=logits.dtype, device="cuda"
    )
    full[:, :vocab_size] = logits
    partials = []
    for rank in range(tp_size):
        start = rank * width
        num_valid = max(0, min(width, vocab_size - start))
        shard = full[:, start : start + width].contiguous()
        partials.append(local_argmax_partials(shard, num_valid, start))
    return merge_argmax_partials(torch.cat(partials), tp_size)


@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("vocab_size", [1000, 32000, 200064])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_rows", [1, 7, 96])
def test_matches_full_argmax(tp_size, vocab_size, dtype, num_rows):
    torch.manual_seed(0)
    logits = torch.randn(num_rows, vocab_size, device="cuda").to(dtype)
    expected = logits.float().argmax(-1)
    torch.testing.assert_close(_sharded_argmax(logits, tp_size), expected)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_ties_resolve_to_lowest_id(tp_size):
    """Coarse bf16 values tie constantly, also across tile and shard
    boundaries."""
    torch.manual_seed(0)
    logits = torch.randint(0, 3, (64, 200064), device="cuda").to(torch.bfloat16)
    logits[:, 150000] = 5
    logits[:, 199000] = 5
    expected = logits.float().argmax(-1)
    torch.testing.assert_close(_sharded_argmax(logits, tp_size), expected)
    assert (expected == 150000).all()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_nan_wins_like_torch_argmax(tp_size):
    torch.manual_seed(0)
    logits = torch.randn(8, 50257, device="cuda")
    logits[0, 40000] = float("nan")
    logits[1, [7, 30000]] = float("nan")
    expected = logits.argmax(-1)
    torch.testing.assert_close(_sharded_argmax(logits, tp_size), expected)


def test_unaligned_padding_is_ignored():
    torch.manual_seed(0)
    logits = torch.randn(16, 50257, device="cuda")
    torch.testing.assert_close(
        _sharded_argmax(logits, 4, pad_multiple=1000), logits.argmax(-1)
    )


def test_empty_batch():
    logits = torch.empty(0, 1000, device="cuda")
    assert _sharded_argmax(logits, 2).shape == (0,)


def test_all_negative_infinity_row_picks_token_zero():
    logits = torch.full((2, 50257), float("-inf"), device="cuda")
    torch.testing.assert_close(_sharded_argmax(logits, 4), logits.argmax(-1))
