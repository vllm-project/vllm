# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL fp8 paged extend attention against an fp32 torch reference.

Batches mix requests with and without context, query lengths that leave a ragged
16-token block, contexts that split the KV scan and ones that do not. Query rows
get magnitudes over two decades, so a token that reads another token's Q descale
gets a visibly different softmax.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm-only test", allow_module_level=True)

from vllm.v1.attention.ops import flydsl_fp8_paged_extend as fly  # noqa: E402

if not fly.is_flydsl_fp8_paged_extend_available():
    pytest.skip("needs gfx950 with FlyDSL", allow_module_level=True)

FP8 = torch.float8_e4m3fn
NUM_KV_HEADS = 2
NUM_HEADS = fly.GQA * NUM_KV_HEADS
D = fly.HEAD_SIZE
PAGE = fly.PAGE_SIZE

# (query lengths, context lengths, window)
CASES = {
    "unsplit": ([100], [500], 0),
    "split": ([37], [40_000], 0),
    "multi_request": ([1, 17, 300, 64, 5], [0, 20_011, 129, 33_000, 3], 0),
    "window": ([300, 7, 129], [5_000, 1_000, 0], 512),
}


def _quant(x: torch.Tensor, per_row: bool) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().flatten(1).amax(dim=1) if per_row else x.abs().amax()
    scale = amax.float().clamp(min=1e-12) / 448.0
    s = scale.view(-1, *([1] * (x.dim() - 1))) if per_row else scale
    return (x.float() / s).to(FP8), scale


def _make_batch(q_lens, ctx_lens):
    seq_lens = [q + c for q, c in zip(q_lens, ctx_lens)]
    pages = [-(-s // PAGE) for s in seq_lens]
    num_blocks = sum(pages) + 4
    max_pages = max(pages)
    perm = torch.randperm(num_blocks).to(torch.int32)
    block_table = torch.zeros(len(q_lens), max_pages, dtype=torch.int32)
    used = 0
    for r, n in enumerate(pages):
        block_table[r, :n] = perm[used : used + n]
        used += n

    kv = torch.randn(2, num_blocks, PAGE, NUM_KV_HEADS, D, device="cuda")
    k_cache, k_scale = _quant(kv[0], per_row=False)
    v_cache, v_scale = _quant(kv[1], per_row=False)

    num_tokens = sum(q_lens)
    row_scale = 10 ** (torch.rand(num_tokens, 1, 1, device="cuda") * 2 - 1)
    q = torch.randn(num_tokens, NUM_HEADS, D, device="cuda") * row_scale
    cu_q = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    return dict(
        q=q.to(torch.bfloat16),
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale.reshape(1),
        v_scale=v_scale.reshape(1),
        block_table=block_table.cuda(),
        cu_q=cu_q,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
    )


def _reference(q, b, scale, window):
    k = b["k_cache"].float() * b["k_scale"]
    v = b["v_cache"].float() * b["v_scale"]
    out = torch.empty(q.shape, dtype=torch.float32, device="cuda")
    cu = b["cu_q"].tolist()
    for r, seq_len in enumerate(b["seq_lens"].tolist()):
        q_len = cu[r + 1] - cu[r]
        rows = torch.arange(seq_len, device="cuda")
        page = b["block_table"][r, rows // PAGE].long()
        kr = k[page, rows % PAGE].repeat_interleave(fly.GQA, dim=1)
        vr = v[page, rows % PAGE].repeat_interleave(fly.GQA, dim=1)
        qr = q[cu[r] : cu[r + 1]].float()
        qpos = torch.arange(seq_len - q_len, seq_len, device="cuda")[:, None]
        mask = rows[None, :] <= qpos
        if window:
            mask &= rows[None, :] > qpos - window
        s = torch.einsum("thd,khd->htk", qr, kr) * scale
        p = s.masked_fill(~mask, float("-inf")).softmax(-1)
        out[cu[r] : cu[r + 1]] = torch.einsum("htk,khd->thd", p, vr)
    return out


@pytest.mark.parametrize("q_fused", [False, True], ids=["fp8_q", "bf16_q"])
@pytest.mark.parametrize("case", CASES)
def test_flydsl_fp8_paged_extend_matches_reference(case, q_fused):
    q_lens, ctx_lens, window = CASES[case]
    set_random_seed(0)
    b = _make_batch(q_lens, ctx_lens)
    rows, _, _ = fly._split_plan(b["cu_q"], b["seq_lens"], NUM_KV_HEADS, window)
    expect_split = case in ("split", "multi_request")
    assert (max(rows[3::4]) > 1) == expect_split

    scale = D**-0.5
    if q_fused:
        q, q_descale = b["q"], None
        q_ref = b["q"].float()
    else:
        q, q_descale = _quant(b["q"], per_row=True)
        q_ref = q.float() * q_descale[:, None, None]
    out = torch.full(b["q"].shape, float("nan"), dtype=torch.bfloat16, device="cuda")
    fly.flydsl_fp8_paged_extend_attention(
        q,
        b["k_cache"],
        b["v_cache"],
        out,
        b["cu_q"].cuda(),
        b["seq_lens"].cuda(),
        b["block_table"],
        max(q_lens),
        q_descale,
        b["k_scale"],
        b["v_scale"],
        scale,
        b["cu_q"],
        b["seq_lens"],
        window=window,
    )
    ref = _reference(q_ref, b, scale, window)

    err = (out.float() - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)
    worst = int(err.argmax())
    # In-kernel Q quantization is coarser than the per-token host quant.
    max_tol, mean_tol = (0.15, 0.06) if q_fused else (0.1, 0.04)
    assert err.max() < max_tol, f"token {worst}: rel err {err[worst]:.4f}"
    assert err.mean() < mean_tol, f"mean rel err {err.mean():.4f}"
