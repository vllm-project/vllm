# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx950 Kimi-K3 DSpark draft MLA decode kernel."""

import math

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()

D_QK, D_V = 576, 512
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and _on_gfx950()),
    reason="Kimi-K3 H40 draft MLA decode requires gfx950",
)


def _reference(q, kv, block_table, page, seq_len, q_scale, kv_scale, sm, req):
    tok = torch.arange(seq_len, device=q.device)
    rows = block_table[req, tok // page].long() * page + (tok % page)
    kvr = kv.view(-1, D_QK)[rows].float() * kv_scale
    qr = q[req].reshape(-1, D_QK).float() * q_scale
    return torch.softmax((qr @ kvr.T) * sm, dim=-1) @ kvr[:, :D_V]


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("query_len", [7])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("seq_len", [4096, 8321])
@pytest.mark.parametrize("page", [128, 3072])
@pytest.mark.parametrize("shuffle_pages", [True])
def test_h40_draft_mla_matches_reference(
    batch, query_len, num_heads, seq_len, page, shuffle_pages
):
    from vllm.models.kimi_k3.amd.ops.h40_draft_mla import h40_draft_mla_decode

    torch.manual_seed(0)
    dev = torch.device("cuda")
    fp8_max = torch.finfo(FP8).max
    npage = (seq_len + page - 1) // page

    q_real = torch.randn(batch, query_len, num_heads, D_QK, device=dev) * 0.5
    kv_real = torch.randn(batch * npage, page, D_QK, device=dev) * 0.5
    q_scale = float(q_real.abs().max()) / fp8_max
    kv_scale = float(kv_real.abs().max()) / fp8_max
    q = (q_real / q_scale).clamp(-fp8_max, fp8_max).to(FP8)
    kv = (kv_real / kv_scale).clamp(-fp8_max, fp8_max).to(FP8)

    pages = torch.arange(batch * npage, dtype=torch.int32, device=dev)
    if shuffle_pages:
        pages = pages[torch.randperm(batch * npage, device=dev)]
    block_table = pages.view(batch, npage).contiguous()
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)

    out = torch.zeros(
        batch * query_len, num_heads, D_V, dtype=torch.bfloat16, device=dev
    )
    sm = 1.0 / math.sqrt(192)

    ran = h40_draft_mla_decode(
        q.view(batch * query_len, num_heads, D_QK),
        kv,
        out,
        block_table,
        seq_lens,
        query_len,
        sm,
        q_scale,
        kv_scale,
    )
    assert ran, "kernel declined a shape it should support"
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()

    got = out.view(batch, query_len * num_heads, D_V)[0].float()
    ref = _reference(q, kv, block_table, page, seq_len, q_scale, kv_scale, sm, 0)
    # e4m3 P in the PV matmul is the error floor; it shrinks with context.
    assert (got - ref).norm() / ref.norm() < 3e-2


def test_h40_draft_mla_declines_unsupported_page_size():
    from vllm.models.kimi_k3.amd.ops.h40_draft_mla import h40_draft_mla_decode

    dev = torch.device("cuda")
    q = torch.zeros(14, 16, D_QK, dtype=FP8, device=dev)
    kv = torch.zeros(4, 100, D_QK, dtype=FP8, device=dev)  # 100 % 128 != 0
    out = torch.zeros(14, 16, D_V, dtype=torch.bfloat16, device=dev)
    bt = torch.zeros(2, 4, dtype=torch.int32, device=dev)
    sl = torch.full((2,), 100, dtype=torch.int32, device=dev)
    assert not h40_draft_mla_decode(q, kv, out, bt, sl, 7, 0.1, 1.0, 1.0)
