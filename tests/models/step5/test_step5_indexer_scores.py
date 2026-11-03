# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 prefill indexer scores vs a float64 reference on scrambled pages."""

import pytest
import torch

from vllm.models.step5.common.ops.sparse_indexer_scores import paged_mqa_logits_sm90
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() or current_platform.has_device_capability(90)),
    reason="needs FP8 Triton dot",
)

DIM = 256


def _fp8_bytes(shape: tuple[int, ...]) -> torch.Tensor:
    raw = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    # 0x80 is -0 in e4m3fn but NaN in e4m3fnuz.
    return raw.masked_fill(torch.rand(shape, device="cuda") < 0.01, 0x80)


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
def test_paged_mqa_logits_sm90(heads: int, next_n: int):
    torch.manual_seed(0)
    batch, page_size, max_seq_len = 11, 32, 700
    pages_per_req = -(-max_seq_len // page_size)
    num_pages = batch * pages_per_req + 5
    block_tables = (
        torch.randperm(num_pages, device="cuda")[: batch * pages_per_req]
        .view(batch, pages_per_req)
        .int()
    )
    seq_lens = torch.randint(
        0, max_seq_len + 1, (batch,), dtype=torch.int32, device="cuda"
    )
    scales = torch.rand(num_pages, page_size, device="cuda")
    kv_fused = torch.cat(
        [_fp8_bytes((num_pages, page_size * DIM)), scales.view(torch.uint8)], dim=1
    ).view(num_pages, page_size, 1, DIM + 4)
    q = _fp8_bytes((batch, next_n, heads, DIM)).view(torch.float8_e4m3fn)
    weights = torch.randn(batch * next_n, heads, device="cuda")
    args = (q, kv_fused, weights, block_tables, seq_lens, max_seq_len)

    torch.testing.assert_close(
        paged_mqa_logits_sm90(*args), _reference(*args), rtol=1e-3, atol=1e-3
    )
