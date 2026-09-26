# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.fewhead_prefill import (
    run_fewhead_sparse_prefill,
)
from vllm.models.deepseek_v41.nvidia.triton_fewhead_sparse_prefill import (
    fewhead_sparse_mla_fwd,
)

D = 512
TOPK = 640
H_LOCAL = 8
H_PAD = 64
VALID_LEN = 512
PAD_SENTINEL = 7.0
MEAN_ABS = 2e-3
MAX_ABS = 2e-2


def _require_flashmla_sparse() -> None:
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    ok, reason = is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason or "FlashMLA sparse unavailable")


def _inputs(s_q: int, *, seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda"
    s_kv = max(s_q, TOPK)
    sm_scale = D**-0.5
    kv = torch.randn((s_kv, 1, D), dtype=torch.bfloat16, device=device)
    q64 = torch.randn((s_q, H_PAD, D), dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, s_kv, (s_q, TOPK), dtype=torch.int32, device=device)
    indices[:, VALID_LEN:] = -1
    lens = torch.full((s_q,), VALID_LEN, dtype=torch.int32, device=device)
    sink64 = torch.zeros((H_PAD,), dtype=torch.float32, device=device)
    sink64[H_LOCAL:] = float("-inf")
    return q64, kv, indices, lens, sink64, sm_scale


def _flash_local_heads(q64, kv, indices, lens, sink64, sm_scale):
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    flash_out = flash_mla_sparse_fwd(
        q=q64,
        kv=kv,
        indices=indices.unsqueeze(1),
        sm_scale=sm_scale,
        attn_sink=sink64,
        topk_length=lens,
    )
    flash_q = flash_out[0] if isinstance(flash_out, (tuple, list)) else flash_out
    return flash_q[:, :H_LOCAL]


def _assert_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    err = (got.float() - ref.float()).abs()
    assert bool(torch.isfinite(got).all())
    assert float(err.mean()) < MEAN_ABS
    assert float(err.max()) < MAX_ABS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("s_q", [64, 2048, 8192])
def test_fewhead_matches_padded_flashmla_noncontiguous_out(s_q: int):
    _require_flashmla_sparse()
    q64, kv, indices, lens, sink64, sm_scale = _inputs(s_q)
    ref = _flash_local_heads(q64, kv, indices, lens, sink64, sm_scale)

    q_view = q64[:, :H_LOCAL]
    out = torch.full_like(q64, PAD_SENTINEL)
    out_view = out[:, :H_LOCAL]
    assert not q_view.is_contiguous()
    assert not out_view.is_contiguous()

    tri = fewhead_sparse_mla_fwd(
        q_view,
        kv,
        indices,
        sm_scale,
        attn_sink=sink64[:H_LOCAL],
        topk_length=lens,
        out=out_view,
    )
    assert tri.data_ptr() == out_view.data_ptr()
    _assert_close(out[:, :H_LOCAL], ref)
    padded = out[:, H_LOCAL:]
    assert torch.equal(padded, torch.full_like(padded, PAD_SENTINEL))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_fewhead_sparse_prefill_writes_local_heads_only():
    _require_flashmla_sparse()
    s_q = 2048
    pad = 128
    q_full = torch.randn((s_q + pad, H_PAD, D), dtype=torch.bfloat16, device="cuda")
    q64, kv, indices, lens, sink64, sm_scale = _inputs(s_q, seed=1)
    q_full[pad : pad + s_q].copy_(q64)
    q_chunk = q_full[pad : pad + s_q]
    ref = _flash_local_heads(q_chunk.contiguous(), kv, indices, lens, sink64, sm_scale)

    out_full = torch.full_like(q_full, PAD_SENTINEL)
    out_chunk = out_full[pad : pad + s_q]
    returned = run_fewhead_sparse_prefill(
        q_chunk,
        kv,
        indices,
        sm_scale,
        attn_sink=sink64,
        topk_length=lens,
        out=out_chunk,
        n_local_heads=H_LOCAL,
    )
    assert returned.data_ptr() == out_chunk.data_ptr()
    assert not q_chunk[:, :H_LOCAL].is_contiguous()
    _assert_close(out_chunk[:, :H_LOCAL], ref)
    assert torch.equal(
        out_chunk[:, H_LOCAL:],
        torch.full_like(out_chunk[:, H_LOCAL:], PAD_SENTINEL),
    )
    assert torch.equal(
        out_full[:pad],
        torch.full_like(out_full[:pad], PAD_SENTINEL),
    )
    assert torch.equal(
        out_full[pad + s_q :],
        torch.full_like(out_full[pad + s_q :], PAD_SENTINEL),
    )
