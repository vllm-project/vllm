# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness tests for the fused MTP depth-layer input kernel.

``embed_dual_rmsnorm_cat`` must match the unfused module sequence exactly:
each rmsnorm computes in fp32 and rounds to bf16 at the same points as the
vendored ``rmsnorm`` kernel (including the bf16 round-trip between the
chained backbone embed_norm and the depth embed_norm), and the fused row
gather matches ``F.embedding``.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

from vllm.models.inkling.nvidia.ops.norm import (
    embed_dual_rmsnorm_cat,
    embed_rmsnorm,
    rmsnorm,
)

EPS = 1e-6
VOCAB = 4096


def _ref(hidden, w_h, w_e, emb, w_pre=None):
    if w_pre is not None:
        emb = rmsnorm(emb, w_pre, EPS)
    return torch.cat([rmsnorm(hidden, w_h, EPS), rmsnorm(emb, w_e, EPS)], dim=-1)


@pytest.mark.parametrize("n", [1536, 6144])
@pytest.mark.parametrize("t", [0, 1, 7, 256])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_embed_dual_rmsnorm_cat(n: int, t: int, ids_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    dev = "cuda"
    table = (torch.randn(VOCAB, n, device=dev) * 0.3).to(torch.bfloat16)
    w_h = torch.randn(n, device=dev).to(torch.bfloat16)
    w_e = (1 + 0.01 * torch.randn(n, device=dev)).to(torch.bfloat16)
    w_pre = torch.randn(n, device=dev).to(torch.bfloat16)
    ids = torch.randint(0, VOCAB, (t,), device=dev, dtype=ids_dtype)
    hidden = (torch.randn(t, n, device=dev) * 2).to(torch.bfloat16)
    emb = table[ids.long()]

    # Fused gather + chained backbone pre-norm (the decode draft-step path).
    out = embed_dual_rmsnorm_cat(
        hidden,
        w_h,
        w_e,
        EPS,
        input_ids=ids,
        embed_table=table,
        pre_norm_weight=w_pre,
    )
    assert out.shape == (t, 2 * n)
    assert torch.equal(out, _ref(hidden, w_h, w_e, emb, w_pre))

    # Precomputed embeds, no pre-norm (draft prefill with target-merged MM
    # embeddings, already backbone-normed).
    out = embed_dual_rmsnorm_cat(hidden, w_h, w_e, EPS, embeds=emb)
    assert torch.equal(out, _ref(hidden, w_h, w_e, emb))

    # Fused gather, no pre-norm (use_embed_norm=False).
    out = embed_dual_rmsnorm_cat(
        hidden, w_h, w_e, EPS, input_ids=ids, embed_table=table
    )
    assert torch.equal(out, _ref(hidden, w_h, w_e, emb))


@pytest.mark.parametrize("n", [1536, 6144])
@pytest.mark.parametrize("t", [0, 1, 7, 256])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_embed_rmsnorm(n: int, t: int, ids_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    dev = "cuda"
    table = (torch.randn(VOCAB, n, device=dev) * 0.3).to(torch.bfloat16)
    w = torch.randn(n, device=dev).to(torch.bfloat16)
    ids = torch.randint(0, VOCAB, (t,), device=dev, dtype=ids_dtype)
    ref_emb = table[ids.long()]

    # Gather + embed_norm (base model / MTP prefill embed path).
    out = embed_rmsnorm(ids, table, w, EPS)
    assert out.shape == (t, n)
    assert torch.equal(out, rmsnorm(ref_emb, w, EPS) if t else ref_emb)

    # Pure gather (use_embed_norm=False / replicated module forward).
    out = embed_rmsnorm(ids, table, None, EPS)
    assert torch.equal(out, ref_emb)

    # Chained first-layer attn_norm (the target text-path forward): one launch
    # emits both the residual and layer 0's normed attention input.
    w_chain = (1 + 0.05 * torch.randn(n, device=dev)).to(torch.bfloat16)
    res, attn_in = embed_rmsnorm(ids, table, w, EPS, chain_weight=w_chain)
    ref_res = rmsnorm(ref_emb, w, EPS) if t else ref_emb
    assert torch.equal(res, ref_res)
    assert torch.equal(attn_in, rmsnorm(ref_res, w_chain, EPS) if t else ref_res)

    # Chained without embed_norm (use_embed_norm=False).
    res, attn_in = embed_rmsnorm(ids, table, None, EPS, chain_weight=w_chain)
    assert torch.equal(res, ref_emb)
    assert torch.equal(attn_in, rmsnorm(ref_emb, w_chain, EPS) if t else ref_emb)
