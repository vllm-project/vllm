# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused vocab-parallel argmax kernels against the ops they replace."""

import pytest
import torch

from vllm.model_executor.layers.local_argmax import pack_top, select_top

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a GPU"
)


def _reference_pack(logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
    vals, idx = logits.max(dim=-1)
    return torch.stack([vals.float(), (idx + vocab_start).float()], dim=-1)


def _reference_select(pairs: torch.Tensor) -> torch.Tensor:
    winner = pairs[:, :, 0].argmax(dim=-1, keepdim=True)
    return pairs[:, :, 1].gather(dim=-1, index=winner).squeeze(-1).to(torch.int64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 17), (8, 1024), (5, 4099), (64, 16160)])
def test_pack_top_matches_tensor_max(dtype, shape):
    logits = torch.randn(*shape, dtype=dtype, device="cuda")
    torch.testing.assert_close(pack_top(logits, 1234), _reference_pack(logits, 1234))


def test_pack_top_breaks_ties_at_lowest_index():
    # equal maxima in different BLOCK=1024 tiles: the earliest must win
    logits = torch.full((3, 4096), -5.0, dtype=torch.float32, device="cuda")
    for row, cols in enumerate([(7, 2048), (1023, 1024), (0, 4095)]):
        logits[row, list(cols)] = 9.0
    torch.testing.assert_close(pack_top(logits, 0), _reference_pack(logits, 0))


def test_pack_top_handles_fully_masked_shard():
    # _mask_outside_vocab can leave a shard entirely -inf
    logits = torch.full((4, 300), float("-inf"), dtype=torch.float32, device="cuda")
    packed = pack_top(logits, 777)
    assert torch.isneginf(packed[:, 0]).all()
    assert (packed[:, 1] == 777).all()


@pytest.mark.parametrize("tp_size", [1, 2, 6, 8])
def test_select_top_matches_gather_reduction(tp_size):
    pairs = torch.empty(32, tp_size, 2, dtype=torch.float32, device="cuda")
    pairs[:, :, 0] = torch.randn(32, tp_size, device="cuda")
    pairs[:, :, 1] = torch.randint(0, 129280, (32, tp_size), device="cuda").float()
    torch.testing.assert_close(select_top(pairs, tp_size), _reference_select(pairs))


def test_kernels_accept_an_empty_batch():
    assert pack_top(torch.empty(0, 64, device="cuda"), 0).shape == (0, 2)
    assert select_top(torch.empty(0, 4, 2, device="cuda"), 4).shape == (0,)
