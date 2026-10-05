# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU / torch golden model of vLLM's GLM-5.2 decode indexer path on ROCm gfx950.

One decode step, T rows (one token per request), from the indexer layer's normed input
``h`` to the CSR physical slot list the MonoKernel consumes. Each stage names the vLLM
code it mirrors:

 1. q_c = bf16(rms_norm(bf16(h @ W_qa[:q_lora]^T), w_qa_norm, eps))
      fused_qkv_a_proj q rows; fused_norm_rope program 2 (_rms_norm), bf16 q_c_out
 2. iq = bf16(q_c @ W_qb^T).view(T, 32, 128)
      indexer.wq_b
 3. iq = rope_interleave(iq[..., :64], pos); (iq_fp8, s_q) = ue8m0_fp8(iq) per
    (row, head)
      _fused_q_kernel program 1 (fused_q; INDEX_ROPE_INTERLEAVE for GLM-5.2)
 4. w = float(kw[:, 128:]) * s_q * head_dim^-0.5 * n_head^-0.5
      same program ("index weights update")
 5. k = layer_norm(float(kw[:, :128]), w_k, b_k, 1e-6); rope_interleave(k[:64], pos);
    (k_fp8, s_k) = ue8m0_fp8(k), written at ``slot`` into the uint8 [blocks, 16, 132]
    index cache in the SHUFFLE layout
      fused_norm_rope program 0 + _fp8_quant_and_cache_write;
      kw = bf16(h @ W_wk^T) = indexer.wk_weights_proj ([wk (128) | weights_proj (32)])
 6. logits[r, t] = s_k[t] * sum_h w[r, h] * relu(iq_fp8[r, h] . k_fp8[t]),
    t < seq_len[r]
      fp8_paged_mqa_logits_torch (v1/attention/ops/rocm_aiter_mla_sparse.py); the
      production deepgemm_fp8_paged_mqa_logits computes the same sum in another order
 7. top-k: torch.ops._C.top_k_per_row_decode (radix histogram select)
      seq_len <= 2048: indices 0..L-1 in order, then -1; else an exact top-2048 SET by
      value in a nondeterministic order, threshold ties taken in arrival order, values
      compared by float bits (-0.0 < +0.0). ``topk_ref`` returns the canonical choice
      (lowest index among threshold ties) plus the tie set, so comparisons are
      set-equality modulo threshold ties.
 8. CSR: indptr = cumsum(min(seq_len, 2048)) (generate_sparse_seqlen_kernel);
    indices[indptr[r] + i] = bt[r, tok // 16] * 16 + tok % 16, or 0 for tok < 0 or an
    out-of-range block (triton_convert_req_index_to_global_index)

Bit-faithfulness: stages 3-5 (RoPE, ue8m0 FP8, cache bytes) follow the Triton op order
in fp32 and are expected bit-exact for bit-identical inputs (Triton may contract
a*c + s*p into an FMA; a 1-ulp difference flips an FP8 rounding only at exact
midpoints). The bf16 GEMMs (1, 2, 5) and the logits (6) accumulate in a different order
than the GPU, so compare them with tolerances; the top-k set is then exact except where
two logits differ by less than the logits error at the threshold.
"""

from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn  # gfx950: OCP E4M3 (USE_FNUZ False), FP8_MAX 448
FP8_MAX = 448.0
BLOCK = 16  # GLM-5.2 vLLM block size (index cache block == MLA block)
HEAD_DIM = 128  # index_head_dim
N_HEAD = 32  # index_n_heads
ROPE = 64  # qk_rope_head_dim (index rope covers dims [0, 64), interleaved pairs)
TOPK = 2048
# _INDEXER_CACHE_BLOCK_TILE (tokens) == _INDEXER_CACHE_HEAD_TILE_BYTES (bytes), fp8
TILE = 16


# ------------------------------------------------------------------ elementwise stages
def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def rms_norm(x, w, eps):
    x = x.float()
    rrms = torch.rsqrt((x * x).sum(-1, keepdim=True) / x.shape[-1] + eps)
    return (x * rrms) * w.float()


def layer_norm(x, w, b, eps):
    x = x.float()
    mean = x.sum(-1, keepdim=True) / x.shape[-1]
    diff = x - mean
    var = (diff * diff).sum(-1, keepdim=True) / x.shape[-1]
    rstd = torch.rsqrt(var + eps)
    return (x - mean) * rstd * w.float() + b.float()


def rope_interleave(
    x: torch.Tensor, cos_sin: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    """X [..., D] fp32 with the row axis first; pairs (2i, 2i+1), i < ROPE/2, rotated by
    cos_sin[pos] = [cos | sin] (kernels.py: roped = x * cos_full + sign * partner *
    sin_full, sign -1 on even lanes)."""
    half = cos_sin.shape[-1] // 2
    cs = cos_sin[pos.long()].float()
    shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (half,)
    c, s = cs[:, :half].reshape(shape), cs[:, half:].reshape(shape)
    out = x.clone()
    e, o = x[..., 0 : 2 * half : 2], x[..., 1 : 2 * half : 2]
    out[..., 0 : 2 * half : 2] = e * c + (-o) * s
    out[..., 1 : 2 * half : 2] = o * c + e * s
    return out


def ue8m0_fp8(v: torch.Tensor, exact: bool = False):
    """_fp8_ue8m0_quantize over the last dim: scale = 2^ceil(log2(max(amax, 1e-4) /
    448)); q = fp8(v / scale). exact=False: fp32 log2 as vLLM / torch evaluate it -- for
    x = amax/448 a few ulps above a power of two the fp32 log2 rounds to that integer,
    so the scale is 2^k and amax/scale slightly exceeds 448 (E4M3 RN brings it back to
    448). exact=True: the mathematically exact ceil (exponent bits + mantissa != 0),
    which the fused kernel's _ue8m0_scale computes (kernel/glm/kernel.py); the two
    differ by 2x in that window only."""
    v = v.float()
    amax = v.abs().amax(-1, keepdim=True)
    x = torch.clamp(amax, min=1e-4) / FP8_MAX
    if exact:
        bits = x.contiguous().view(torch.int32)
        e = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        scale = (e << 23).view(torch.float32)
    else:
        scale = torch.exp2(torch.ceil(torch.log2(x)))
    return (v / scale).to(FP8), scale.squeeze(-1)


# ------------------------------------------------------------------ index cache
def value_offset(block_offset: int | torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """SHUFFLE layout byte offset inside a block (kernels.py
    _fp8_quant_and_cache_write): [blk/16, D/16, 16 tokens, 16 bytes]."""
    return (
        block_offset // TILE * TILE * HEAD_DIM
        + block_offset % TILE * TILE
        + d // TILE * TILE * TILE
        + d % TILE
    )


def cache_write(cache: torch.Tensor, slot: int, k_fp8: torch.Tensor, scale: float):
    blk, off = slot // BLOCK, slot % BLOCK
    flat = cache.view(cache.shape[0], -1)
    flat[blk, value_offset(off, torch.arange(HEAD_DIM))] = k_fp8.view(torch.uint8)
    flat[blk, BLOCK * HEAD_DIM + off * 4 : BLOCK * HEAD_DIM + off * 4 + 4] = (
        torch.tensor([scale], dtype=torch.float32).view(torch.uint8)
    )


def cache_read(cache: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(fp8 values [16, 128], fp32 scales [16]) of one physical block."""
    flat = cache.view(cache.shape[0], -1)[block]
    vals = flat[value_offset(torch.arange(BLOCK)[:, None], torch.arange(HEAD_DIM))]
    vals = vals.view(FP8)
    scales = flat[BLOCK * HEAD_DIM : BLOCK * HEAD_DIM + 4 * BLOCK].view(torch.float32)
    return vals, scales


# ----------------------------------------------------------- score / select / CSR
def paged_logits(iq_fp8, w, cache, block_table, seq_lens, max_len: int) -> torch.Tensor:
    """[T, max_len] fp32, -inf past seq_len (fp8_paged_mqa_logits_torch,
    next_n == 1)."""
    T = iq_fp8.shape[0]
    out = torch.full((T, max_len), float("-inf"))
    for r in range(T):
        L = int(seq_lens[r])
        nb = (L + BLOCK - 1) // BLOCK
        vals, scales = zip(
            *(cache_read(cache, int(block_table[r, b])) for b in range(nb))
        )
        K = torch.cat(vals).float()[:L]  # [L, 128]
        S = torch.cat(scales)[:L]
        q = iq_fp8[r].float()  # [32, 128]
        score = torch.relu(K @ q.T) * w[r].float()[None, :]  # [L, 32]
        out[r, :L] = score.sum(1) * S
    return out


def order_key(x: torch.Tensor) -> torch.Tensor:
    """Total order the radix top-k uses (sampler.cu extractBinIdx / isPartialMatch): on
    the float BITS, so -0.0 ranks below +0.0 and equal values are equal bit patterns.
    Returned increasing with the value (int64)."""
    b = x.contiguous().view(torch.int32).long()
    return torch.where(b >= 0, b ^ (-(2**31)), ~b) & 0xFFFFFFFF


def topk_ref(logits: torch.Tensor, seq_lens: torch.Tensor, k: int = TOPK):
    """-> (canonical indices [T, k] int32, per-row dict(must=set, ties=set, need=int)).
    Canonical = all values above the k-th value plus the lowest-index threshold ties
    (the kernel's choice among ties is arrival order)."""
    T = logits.shape[0]
    idx = torch.full((T, k), -1, dtype=torch.int32)
    info = []
    for r in range(T):
        L = int(seq_lens[r])
        if k >= L:
            idx[r, :L] = torch.arange(L, dtype=torch.int32)
            info.append(dict(must=set(range(L)), ties=set(), need=0))
            continue
        key = order_key(logits[r, :L])
        thr = torch.topk(key, k).values[-1]
        above = torch.nonzero(key > thr).view(-1)
        ties = torch.nonzero(key == thr).view(-1)
        need = k - above.numel()
        sel = torch.cat([above, ties[:need]])
        idx[r] = sel.to(torch.int32)
        info.append(dict(must=set(above.tolist()), ties=set(ties.tolist()), need=need))
    return idx, info


def topk_equal_mod_ties(got: torch.Tensor, info: list) -> list[bool]:
    """Set comparison: every 'must' index present, the rest drawn from the threshold
    ties, no duplicates."""
    res = []
    for r, inf in enumerate(info):
        g = [int(x) for x in got[r].tolist() if x >= 0]
        gs = set(g)
        ok = (
            len(gs) == len(g)
            and inf["must"] <= gs
            and gs - inf["must"] <= inf["ties"]
            and len(gs - inf["must"]) == inf["need"]
        )
        res.append(ok)
    return res


def csr(
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    k: int = TOPK,
):
    """(indptr [T+1] int32, indices [indptr[-1]] int32) as generate_sparse_seqlen +
    triton_convert produce them."""
    T = topk_idx.shape[0]
    n = torch.clamp(seq_lens.long(), max=k)
    indptr = torch.zeros(T + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(n, 0).to(torch.int32)
    out = torch.zeros(int(indptr[-1]), dtype=torch.int32)
    maxb = block_table.shape[1]
    for r in range(T):
        tok = topk_idx[r, : int(n[r])].long()
        b = torch.div(tok, BLOCK, rounding_mode="floor")
        valid = (tok >= 0) & (b >= 0) & (b < maxb)
        base = block_table[r, b.clamp(0, maxb - 1)].long()
        val = torch.where(valid, base * BLOCK + tok % BLOCK, torch.zeros_like(tok))
        out[int(indptr[r]) : int(indptr[r + 1])] = val.to(torch.int32)
    return indptr, out


# ------------------------------------------------------------------ whole decode step
def decode_step(
    W: dict,
    h: torch.Tensor,
    pos: torch.Tensor,
    slots: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    cache: torch.Tensor,
    cos_sin: torch.Tensor,
    eps_q: float = 1e-5,
) -> dict:
    """W: w_qa [q_lora + ..., H] (only the first q_lora rows used), w_qa_norm [q_lora],
    w_qb [32*128, q_lora], w_wk [128 + 32, H], k_norm_w / k_norm_b [128]. h: [T, H] bf16
    (the layer's normed input). ``cache`` is updated in place at ``slots`` (the current
    tokens) before scoring, as in vLLM."""
    T = h.shape[0]
    q_lora = W["w_qa_norm"].shape[0]
    q_raw = bf16(h.float() @ W["w_qa"][:q_lora].float().T)
    q_c = bf16(rms_norm(q_raw, W["w_qa_norm"], eps_q))
    iq = bf16(q_c.float() @ W["w_qb"].float().T).view(T, N_HEAD, HEAD_DIM).float()
    iq = rope_interleave(iq, cos_sin, pos)
    iq_fp8, s_q = ue8m0_fp8(iq)  # [T, 32, 128], [T, 32]
    kw = bf16(h.float() @ W["w_wk"].float().T)
    w = kw[:, HEAD_DIM:].float() * s_q * (HEAD_DIM**-0.5) * (N_HEAD**-0.5)
    k = layer_norm(kw[:, :HEAD_DIM], W["k_norm_w"], W["k_norm_b"], 1e-6)
    k = rope_interleave(k, cos_sin, pos)
    k_fp8, s_k = ue8m0_fp8(k)
    for r in range(T):
        if int(slots[r]) >= 0:
            cache_write(cache, int(slots[r]), k_fp8[r], float(s_k[r]))
    logits = paged_logits(iq_fp8, w, cache, block_table, seq_lens, int(seq_lens.max()))
    top, info = topk_ref(logits, seq_lens)
    indptr, _ = csr(top, seq_lens, block_table)
    return dict(
        iq_fp8=iq_fp8,
        weights=w,
        k_fp8=k_fp8,
        logits=logits,
        topk=top,
        topk_info=info,
        indptr=indptr,
    )
