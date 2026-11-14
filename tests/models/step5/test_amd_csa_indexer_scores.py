# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Step-5 CSA prefill indexer scores vs a float64 reference."""

import pytest
import torch

from vllm.models.step5.amd.ops.indexer_scores import paged_mqa_logits
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Step5 CSA kernels are the ROCm path"
)

DEVICE = "cuda"
DIM = 256


def _fp8_bytes(shape: tuple[int, ...]) -> torch.Tensor:
    raw = torch.randn(shape, device=DEVICE).to(torch.float8_e4m3fn).view(torch.uint8)
    # 0x80 is -0 in e4m3fn but NaN in e4m3fnuz.
    return raw.masked_fill(torch.rand(shape, device=DEVICE) < 0.01, 0x80)


def _reference(q, kv_fused, weights, block_tables, seq_lens, max_seq_len):
    batch, next_n, heads, _ = q.shape
    pages, page_size = kv_fused.shape[:2]
    packed = kv_fused.view(pages, -1)
    keys = packed[:, : page_size * DIM].view(torch.float8_e4m3fn).double()
    keys = keys.view(pages, page_size, DIM)
    scales = packed[:, page_size * DIM :].view(torch.float32).double()

    k = torch.arange(max_seq_len, device=q.device)
    page = block_tables.long()[:, k // page_size]
    dots = torch.einsum("bnhd,bkd->bnhk", q.double(), keys[page, k % page_size])
    w = weights.double().view(batch, next_n, heads, 1)
    logits = (dots.relu() * w).sum(2) * scales[page, k % page_size][:, None]
    logits = logits.masked_fill(k >= seq_lens[:, None, None], float("-inf"))
    return logits.view(batch * next_n, max_seq_len).float()


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("next_n", [1, 2])
@torch.inference_mode()
def test_scores_match_float64_on_scrambled_pages(heads: int, next_n: int):
    torch.manual_seed(0)
    batch, page_size, max_seq_len = 11, 32, 700
    pages_per_req = -(-max_seq_len // page_size)
    num_pages = batch * pages_per_req + 5
    block_tables = (
        torch.randperm(num_pages, device=DEVICE)[: batch * pages_per_req]
        .view(batch, pages_per_req)
        .int()
    )
    seq_lens = torch.randint(
        0, max_seq_len + 1, (batch,), dtype=torch.int32, device=DEVICE
    )
    scales = torch.rand(num_pages, page_size, device=DEVICE)
    kv_fused = torch.cat(
        [_fp8_bytes((num_pages, page_size * DIM)), scales.view(torch.uint8)], dim=1
    ).view(num_pages, page_size, 1, DIM + 4)
    q = _fp8_bytes((batch, next_n, heads, DIM)).view(torch.float8_e4m3fn)
    weights = torch.randn(batch * next_n, heads, device=DEVICE)
    args = (q, kv_fused, weights, block_tables, seq_lens, max_seq_len)

    torch.testing.assert_close(
        paged_mqa_logits(*args), _reference(*args), rtol=1e-3, atol=1e-3
    )


@torch.inference_mode()
def test_scores_exact_on_every_fp8_byte_pair():
    """Each score is one product of two FP8 values, so it is exact in FP32."""
    codes = torch.arange(256, device=DEVICE).to(torch.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]  # Drop the NaN codes.
    n, page_size, heads = codes.numel(), 32, 4
    num_pages = -(-n // page_size)

    q = torch.zeros(n, 1, heads, DIM, dtype=torch.uint8, device=DEVICE)
    q[:, 0, 0, 0] = codes
    keys = torch.zeros(num_pages * page_size, DIM, dtype=torch.uint8, device=DEVICE)
    keys[:n, 0] = codes
    scales = torch.ones(num_pages, page_size, device=DEVICE)
    kv_fused = torch.cat(
        [keys.view(num_pages, -1), scales.view(torch.uint8)], dim=1
    ).view(num_pages, page_size, 1, DIM + 4)
    weights = torch.zeros(n, heads, device=DEVICE)
    weights[:, 0] = 1
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    seq_lens = torch.full((n,), n, dtype=torch.int32, device=DEVICE)

    got = paged_mqa_logits(
        q.view(torch.float8_e4m3fn),
        kv_fused,
        weights,
        block_tables.repeat(n, 1),
        seq_lens,
        n,
    )
    values = codes.view(torch.float8_e4m3fn).double()
    assert torch.equal(got.double(), (values[:, None] * values[None, :]).relu())
