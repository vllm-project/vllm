# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashMLA fused norm + RoPE + sparse attention + RoPE + FP8 cast kernel.

The fused kernel is checked against the existing split-KV pipeline
(``flash_mla_with_kvcache`` / ``flash_mla_sparse_fwd`` + ``fused_inv_rope_fp8_quant``
+ DeepGEMM einsum) at the ``lse`` and at the ``wo_a`` output.
"""

import pytest
import torch

import vllm.v1.attention.ops.flashmla as fm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)
from vllm.models.deepseek_v4_1.common.ops import (
    fused_inv_rope_fp8_quant,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4_1.common.ops.fused_layout import (
    inverse_permutation,
    o_fused_permutation,
    permute_q_to_fused,
    permute_wo_a_,
)
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.math_utils import round_up

HEAD_DIM, ROPE_DIM, NOPE_DIM = 512, 64, 448
V4_BYTES, V4_ROW_ALIGN = 584, 576


def _skip_unless_supported():
    ok, reason = fm.is_flashmla_fused_sparse_supported()
    if not ok:
        pytest.skip(reason)


def _paged_zero_cache(num_blocks, block_size, bytes_per_token, align, device):
    """Zeroed ``[num_blocks, block_size, 1, bytes]`` view with TMA-aligned pages."""
    page = round_up(block_size * bytes_per_token, align)
    cache = torch.zeros(num_blocks, page, dtype=torch.uint8, device=device)
    return cache[:, : block_size * bytes_per_token].unflatten(
        1, (block_size, 1, bytes_per_token)
    )


def make_cos_sin_cache(max_pos: int, device) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device) / ROPE_DIM)
    )
    freqs = torch.outer(
        torch.arange(max_pos, dtype=torch.float32, device=device), inv_freq
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def rope_gptj(
    x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor
) -> torch.Tensor:
    """GPT-J (interleaved pairs) RoPE on the last 64 dims of [N, ..., 512]."""
    cs = cos_sin[positions].float()
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [ROPE_DIM // 2]
    cos = cs[:, : ROPE_DIM // 2].view(shape)
    sin = cs[:, ROPE_DIM // 2 :].view(shape)
    r = x[..., NOPE_DIM:].float().unflatten(-1, (ROPE_DIM // 2, 2))
    x0, x1 = r[..., 0], r[..., 1]
    rot = torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1).flatten(-2)
    return torch.cat([x[..., :NOPE_DIM].float(), rot], -1).to(x.dtype)


def build_v4_cache(k: torch.Tensor, block_size: int) -> torch.Tensor:
    """Quantize RoPE'd rows ``k [T, 512]`` into a V4 (584 B) paged cache."""
    num_tokens = k.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    block_bytes = round_up(block_size * V4_BYTES, V4_ROW_ALIGN)
    cache = torch.zeros(num_blocks, block_bytes, dtype=torch.uint8, device=k.device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=k.device)
    quantize_and_insert_k_cache(k, cache, slots, block_size=block_size)
    return cache[:, : block_size * V4_BYTES].unflatten(1, (block_size, 1, V4_BYTES))


def dequant_fused_output(out_fp8: torch.Tensor, out_sf: torch.Tensor) -> torch.Tensor:
    """[s_q, G, 4096] fp32 from e4m3 values and packed ue8m0 per-32 scales."""
    sf_bytes = out_sf.contiguous().view(torch.uint8).view(*out_sf.shape[:2], 128)
    scale = torch.exp2(sf_bytes.float() - 127.0)
    vals = out_fp8.float().unflatten(-1, (128, 32))
    return (vals * scale.unsqueeze(-1)).flatten(-2)


def _quant_rows_per32(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = w.float().unflatten(-1, (-1, 32))
    exp = torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448.0))
    q = (grouped / torch.exp2(exp).unsqueeze(-1)).clamp(-448, 448)
    return q.flatten(-2).to(torch.float8_e4m3fn), (exp + 127).to(torch.uint8)


def make_wo_a(n_groups: int, device):
    """Random MXFP8 wo_a: (fused-permuted, standard) DeepGEMM (weight, sf) pairs."""
    w = torch.randn(n_groups * 1024, 8 * HEAD_DIM, device=device) * 0.02
    q, s = _quant_rows_per32(w)
    q_perm, s_perm = q.clone(), s.clone()
    permute_wo_a_(q_perm, s_perm, heads_per_group=8)
    std = deepgemm_post_process_fp8_weight_block(
        wq=q,
        ws=s,
        quant_block_shape=(1, 32),
        use_e8m0=False,
        is_bmm=True,
        bmm_batch_size=n_groups,
    )
    perm = deepgemm_post_process_fp8_weight_block(
        wq=q_perm,
        ws=s_perm,
        quant_block_shape=(1, 32),
        use_e8m0=False,
        is_bmm=True,
        bmm_batch_size=n_groups,
    )
    return perm[0], perm[1], std[0], std[1]


def _random_indices(s_q, topk, num_slots, device, min_len=1):
    lens = torch.randint(min_len, topk + 1, (s_q,), device=device, dtype=torch.int32)
    idx = torch.randint(0, num_slots, (s_q, topk), device=device, dtype=torch.int32)
    idx[torch.arange(topk, device=device).view(1, -1) >= lens.view(-1, 1)] = -1
    return idx, lens


def test_fused_decode_smoke_shapes():
    _skip_unless_supported()
    device = torch.device("cuda")
    s_q, h_q, topk, page = 3, 64, 128, 32
    q = torch.zeros(s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_cache = _paged_zero_cache(4, page, V4_BYTES, V4_ROW_ALIGN, device)
    indices = torch.full((s_q, topk), -1, dtype=torch.int32, device=device)
    indices[:, 0] = 0
    positions = torch.zeros(s_q, dtype=torch.int32, device=device)
    cos_sin = torch.zeros(16, ROPE_DIM, dtype=torch.float32, device=device)
    cos_sin[:, : ROPE_DIM // 2] = 1.0
    out_fp8, out_sf, lse = fm.flash_mla_fused_sparse_decode(
        q, k_cache, indices, HEAD_DIM**-0.5, positions, cos_sin, n_wv_group=h_q // 8
    )
    assert out_fp8.shape == (s_q, h_q // 8, 4096)
    assert out_fp8.dtype == torch.float8_e4m3fn
    assert out_sf.shape == (s_q, h_q // 8, 32) and out_sf.dtype == torch.int32
    assert out_sf.stride(0) == 1
    assert lse.shape == (s_q, h_q)


@pytest.mark.parametrize("s_q", [1, 7, 64, 300])
@pytest.mark.parametrize("with_extra", [False, True])
def test_fused_decode_matches_split_kv_pipeline(s_q: int, with_extra: bool):
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    h_q, n_groups, block_size = 64, 8, 32
    topk_swa, topk_extra = 128, 512
    scale = HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(8192, device)
    positions = torch.randint(0, 8192, (s_q,), device=device)
    q_std = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)
    sink = torch.randn(h_q, device=device) - 2.0
    sink[h_q // 2 :] = -float("inf")

    k_swa = torch.randn(2048, HEAD_DIM, device=device, dtype=torch.bfloat16)
    swa_cache = build_v4_cache(k_swa, block_size)
    swa_idx, swa_len = _random_indices(s_q, topk_swa, k_swa.shape[0], device)
    extra_cache = extra_idx = extra_len = None
    if with_extra:
        k_extra = torch.randn(4096, HEAD_DIM, device=device, dtype=torch.bfloat16)
        extra_cache = build_v4_cache(k_extra, 128)
        extra_idx, extra_len = _random_indices(
            s_q, topk_extra, k_extra.shape[0], device, min_len=0
        )

    out_ref, lse_ref = fm.flash_mla_with_kvcache(
        q=rope_gptj(q_std, positions, cos_sin).unsqueeze(1),
        k_cache=swa_cache,
        block_table=None,
        head_dim_v=HEAD_DIM,
        tile_scheduler_metadata=fm.FlashMLASchedMeta(),
        cache_seqlens=None,
        is_fp8_kvcache=True,
        indices=swa_idx.view(s_q, 1, topk_swa),
        topk_length=swa_len,
        softmax_scale=scale,
        attn_sink=sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=None
        if extra_idx is None
        else extra_idx.view(s_q, 1, topk_extra),
        extra_topk_length=extra_len,
    )
    o_ref_fp8, o_ref_sf = fused_inv_rope_fp8_quant(
        out_ref.squeeze(1),
        positions,
        cos_sin,
        n_groups=n_groups,
        heads_per_group=8,
        quant_group_size=32,
        tma_aligned_scales=True,
    )

    out_fp8, out_sf, lse = fm.flash_mla_fused_sparse_decode(
        permute_q_to_fused(q_std),
        swa_cache,
        swa_idx,
        scale,
        positions.to(torch.int32),
        cos_sin,
        n_groups,
        attn_sink=sink,
        topk_length=swa_len,
        extra_k_cache=extra_cache,
        extra_indices=extra_idx,
        extra_topk_length=extra_len,
    )

    torch.testing.assert_close(lse, lse_ref.view(s_q, h_q), rtol=1e-3, atol=1e-3)
    inv = inverse_permutation(o_fused_permutation(8, HEAD_DIM)).to(device)
    deq = dequant_fused_output(out_fp8, out_sf)[..., inv]
    deq_ref = dequant_fused_output(o_ref_fp8, o_ref_sf)
    rel = (deq - deq_ref).abs().mean() / deq_ref.abs().mean()
    assert rel < 2e-2, rel

    w_perm, sf_perm, w_std, sf_std = make_wo_a(n_groups, device)
    z = torch.empty(s_q, n_groups, 1024, device=device, dtype=torch.bfloat16)
    z_ref = torch.empty_like(z)
    fp8_einsum(
        "bhr,hdr->bhd", (out_fp8, out_sf), (w_perm, sf_perm), z, recipe=(1, 1, 32)
    )
    fp8_einsum(
        "bhr,hdr->bhd", (o_ref_fp8, o_ref_sf), (w_std, sf_std), z_ref, recipe=(1, 1, 32)
    )
    torch.testing.assert_close(
        z, z_ref, rtol=2e-2, atol=2e-2 * z_ref.abs().max().item()
    )


def test_fused_output_sliced_groups_feed_einsum():
    """TP>1 keeps only the first local groups; DeepGEMM must accept the slice."""
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    s_q, n_groups, keep = 33, 8, 2
    cos_sin = make_cos_sin_cache(64, device)
    positions = torch.zeros(s_q, dtype=torch.int32, device=device)
    q = torch.randn(s_q, 64, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = build_v4_cache(
        torch.randn(256, HEAD_DIM, device=device, dtype=torch.bfloat16), 32
    )
    idx, lens = _random_indices(s_q, 128, 256, device)
    out_fp8, out_sf, _ = fm.flash_mla_fused_sparse_decode(
        q, cache, idx, HEAD_DIM**-0.5, positions, cos_sin, n_groups, topk_length=lens
    )
    w_perm, sf_perm, _, _ = make_wo_a(n_groups, device)
    z_full = torch.empty(s_q, n_groups, 1024, device=device, dtype=torch.bfloat16)
    fp8_einsum(
        "bhr,hdr->bhd", (out_fp8, out_sf), (w_perm, sf_perm), z_full, recipe=(1, 1, 32)
    )
    z_slice = torch.empty(s_q, keep, 1024, device=device, dtype=torch.bfloat16)
    fp8_einsum(
        "bhr,hdr->bhd",
        (out_fp8[:, :keep], out_sf[:, :keep]),
        (w_perm[:keep], sf_perm[:keep]),
        z_slice,
        recipe=(1, 1, 32),
    )
    torch.testing.assert_close(z_slice, z_full[:, :keep], rtol=0, atol=0)


@pytest.mark.parametrize("s_q", [1, 184, 2123])
def test_fused_prefill_matches_sparse_fwd_pipeline(s_q: int):
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    h_q, n_groups, topk, s_kv = 64, 8, 640, 4096
    scale = HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(8192, device)
    positions = torch.randint(0, 8192, (s_q,), device=device)
    q_std = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv = torch.randn(s_kv, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    idx, lens = _random_indices(s_q, topk, s_kv, device)
    sink = torch.randn(h_q, device=device) - 2.0

    out_ref, max_logits_ref, lse_ref = fm.flash_mla_sparse_fwd(
        rope_gptj(q_std, positions, cos_sin),
        kv,
        idx.view(s_q, 1, topk),
        scale,
        attn_sink=sink,
        topk_length=lens,
    )
    o_ref_fp8, o_ref_sf = fused_inv_rope_fp8_quant(
        out_ref,
        positions,
        cos_sin,
        n_groups=n_groups,
        heads_per_group=8,
        quant_group_size=32,
        tma_aligned_scales=True,
    )
    out_fp8, out_sf, max_logits, lse = fm.flash_mla_fused_sparse_prefill(
        permute_q_to_fused(q_std),
        kv,
        idx.view(s_q, 1, topk),
        scale,
        positions.to(torch.int32),
        cos_sin,
        n_groups,
        attn_sink=sink,
        topk_length=lens,
    )
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(max_logits, max_logits_ref, rtol=1e-3, atol=1e-3)
    w_perm, sf_perm, w_std, sf_std = make_wo_a(n_groups, device)
    z = torch.empty(s_q, n_groups, 1024, device=device, dtype=torch.bfloat16)
    z_ref = torch.empty_like(z)
    fp8_einsum(
        "bhr,hdr->bhd", (out_fp8, out_sf), (w_perm, sf_perm), z, recipe=(1, 1, 32)
    )
    fp8_einsum(
        "bhr,hdr->bhd", (o_ref_fp8, o_ref_sf), (w_std, sf_std), z_ref, recipe=(1, 1, 32)
    )
    torch.testing.assert_close(
        z, z_ref, rtol=2e-2, atol=2e-2 * z_ref.abs().max().item()
    )
