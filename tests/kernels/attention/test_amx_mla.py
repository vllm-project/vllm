# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the AMX-only CPU MLA backend: the vendored
decode/extend/bmm kernels, the KV cache write, and the ``AMXMLAImpl``
backend built on top of them.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

if not torch.cpu._is_amx_tile_supported():
    pytest.skip("AMX MLA requires an AMX-capable host", allow_module_level=True)

torch.cpu._init_amx()

from vllm import _custom_ops as ops  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402
from vllm.v1.attention.backends.mla.amx_mla import (  # noqa: E402
    AMXMLAImpl,
    _compute_num_kv_splits,
    _expand_block_table,
)

KV_LORA_RANK = 64
QK_ROPE_HEAD_DIM = 32
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 96, kv-cache row width
# The extend/decode kernels transpose by 512-bit lanes: both the cache row
# width and the (kv_lora_rank-wide) value width must be 32-element aligned.
DTYPE = torch.bfloat16
ATOL = 2e-2
RTOL = 2e-2
# Backend-level tests chain absorb -> attend -> de-absorb through bf16,
# accumulating more rounding error than a single kernel op.
_IMPL_ATOL = 1.5e-1
_IMPL_RTOL = 1.5e-1


def _flatten_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    """(num_blocks, block_size, head_size) -> (num_slots, head_size)."""
    return kv_cache.view(-1, kv_cache.size(-1))


def _random_paged_cache(num_slots: int, head_size: int) -> torch.Tensor:
    return torch.randn(num_slots, head_size, dtype=DTYPE)


def _ref_latent_attn(
    q: torch.Tensor,
    keys: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
) -> torch.Tensor:
    """Plain-PyTorch causal-free latent-space MQA attention for one request.

    q: (num_heads, head_size); keys: (ctx_len, head_size). Values are the
    first ``kv_lora_rank`` columns of the same latent rows (MLA's K/V
    aliasing), matching what decode_attention_cpu/extend_attention_cpu
    compute internally for the MLA-shaped case.
    """
    values = keys[:, :kv_lora_rank]
    scores = (q.float() @ keys.float().T) * scale
    probs = torch.softmax(scores, dim=-1)
    return (probs @ values.float()).to(q.dtype)


def test_bmm_cpu_matches_torch_bmm():
    set_random_seed(0)
    # bmm_cpu requires the output's last dim (mat2's out-features) to be a
    # multiple of 32 (tinygemm's tile width); the contraction dim is free.
    n, b, p, l = 8, 32, 32, 32  # noqa: E741
    mat1 = torch.randn(n, b, p, dtype=DTYPE)
    mat2 = torch.randn(n, l, p, dtype=DTYPE)  # (N, OUT, IN) Linear convention

    ref = torch.bmm(mat1.float(), mat2.float().transpose(1, 2)).to(DTYPE)

    out = torch.empty(n, b, l, dtype=DTYPE)
    ops.bmm_cpu(out, mat1, mat2, False, None)
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    packed = torch.ops._C.convert_weight_packed(mat2.contiguous())
    out_vnni = torch.empty(n, b, l, dtype=DTYPE)
    ops.bmm_cpu(out_vnni, mat1, packed, True, None)
    torch.testing.assert_close(out_vnni, ref, atol=ATOL, rtol=RTOL)


def test_concat_and_cache_mla_round_trip():
    set_random_seed(0)
    num_tokens = 37
    num_blocks, block_size = 8, 16
    kv_cache = torch.zeros(num_blocks, block_size, HEAD_SIZE, dtype=DTYPE)

    kv_c_normed = torch.randn(num_tokens, KV_LORA_RANK, dtype=DTYPE)
    k_pe = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=DTYPE)
    slot_mapping = torch.randperm(num_blocks * block_size)[:num_tokens].to(torch.int64)

    ops.amx_mla_concat_and_cache(kv_c_normed, k_pe, kv_cache, slot_mapping)

    flat = _flatten_cache(kv_cache)
    read_back = flat[slot_mapping]
    expected = torch.cat([kv_c_normed, k_pe], dim=-1)
    torch.testing.assert_close(read_back, expected)

    untouched = torch.ones(num_blocks * block_size, dtype=torch.bool)
    untouched[slot_mapping] = False
    assert (flat[untouched] == 0).all()


@pytest.mark.parametrize("seq_lens", [[5], [1, 300, 33], [513, 1, 129, 7]])
def test_decode_attention_cpu_matches_reference(seq_lens):
    set_random_seed(0)
    num_seqs = len(seq_lens)
    num_heads = 4
    block_size = 32
    max_len = max(seq_lens)
    max_blocks = (max_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks

    kv_cache = _random_paged_cache(num_blocks * block_size, HEAD_SIZE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int64)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)
    scale = HEAD_SIZE**-0.5

    q = torch.randn(num_seqs, num_heads, HEAD_SIZE, dtype=DTYPE)
    kv_cache_flat = kv_cache.view(-1, 1, HEAD_SIZE)
    v_buffer = kv_cache_flat[..., :KV_LORA_RANK]

    o = torch.zeros(num_seqs, num_heads, KV_LORA_RANK, dtype=DTYPE)
    attn_logits = torch.zeros(
        num_seqs, num_heads, 4, KV_LORA_RANK + 1, dtype=torch.float32
    )
    ops.cpu_mla_decode(
        q,
        kv_cache_flat,
        v_buffer,
        o,
        None,
        None,
        None,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens_t,
        scale,
        0.0,
        False,
        0,
        None,
        None,
    )

    flat_cache = _flatten_cache(kv_cache)
    for i, seq_len in enumerate(seq_lens):
        token_ids = req_to_token[i, :seq_len]
        keys = flat_cache[token_ids]
        ref = _ref_latent_attn(q[i], keys, scale, KV_LORA_RANK)
        torch.testing.assert_close(o[i], ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "prefix_lens,extend_lens",
    [
        ([0], [9]),  # fresh prefill, no cached prefix
        ([64], [17]),  # continuation of a cached prefix
        ([0, 40, 128], [23, 5, 31]),  # mixed batch
    ],
)
def test_extend_attention_cpu_matches_reference(prefix_lens, extend_lens):
    set_random_seed(0)
    num_seqs = len(prefix_lens)
    num_heads = 4
    block_size = 32
    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    max_len = max(seq_lens)
    max_blocks = (max_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks

    kv_cache = _random_paged_cache(num_blocks * block_size, HEAD_SIZE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int64)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)
    extend_seq_lens_t = torch.tensor(extend_lens, dtype=torch.int64)
    extend_start_loc = torch.cumsum(
        torch.tensor([0, *extend_lens[:-1]], dtype=torch.int64), dim=0
    )
    scale = HEAD_SIZE**-0.5
    total_new_tokens = sum(extend_lens)

    # The new tokens' latent K/V, already written into the cache (mirrors
    # do_kv_cache_update running before forward_mha in real usage).
    k_extend_flat = torch.randn(total_new_tokens, HEAD_SIZE, dtype=DTYPE)
    flat_cache = _flatten_cache(kv_cache)
    for i in range(num_seqs):
        prefix_len = prefix_lens[i]
        ext_len = extend_lens[i]
        new_token_ids = req_to_token[i, prefix_len : prefix_len + ext_len]
        start = extend_start_loc[i].item()
        flat_cache[new_token_ids] = k_extend_flat[start : start + ext_len]

    q_extend = torch.randn(total_new_tokens, num_heads, HEAD_SIZE, dtype=DTYPE)
    k_extend = k_extend_flat.unsqueeze(1)
    v_extend = k_extend[..., :KV_LORA_RANK]
    kv_cache_flat = kv_cache.view(-1, 1, HEAD_SIZE)
    v_buffer = kv_cache_flat[..., :KV_LORA_RANK]

    o_extend = torch.empty(total_new_tokens, num_heads, KV_LORA_RANK, dtype=DTYPE)
    ops.cpu_mla_extend(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        kv_cache_flat,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens_t,
        extend_seq_lens_t,
        extend_start_loc,
        max(extend_lens),
        scale,
        0.0,
        False,
        0,
        None,
        None,
        None,
    )

    for i in range(num_seqs):
        prefix_len = prefix_lens[i]
        ext_len = extend_lens[i]
        start = extend_start_loc[i].item()
        for t in range(ext_len):
            visible = prefix_len + t + 1
            token_ids = req_to_token[i, :visible]
            keys = flat_cache[token_ids]
            ref = _ref_latent_attn(q_extend[start + t], keys, scale, KV_LORA_RANK)
            torch.testing.assert_close(o_extend[start + t], ref, atol=ATOL, rtol=RTOL)


class _FakeLinear:
    """Minimal duck-typed stand-in for kv_b_proj: only .weight and
    .quant_method are read by get_and_maybe_dequant_weights()."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.quant_method = None


def _make_amx_mla_impl(num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank):
    weight = torch.randn(
        num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank, dtype=DTYPE
    )
    kv_b_proj = _FakeLinear(weight)
    impl = AMXMLAImpl(
        num_heads=num_heads,
        head_size=kv_lora_rank + QK_ROPE_HEAD_DIM,
        scale=(qk_nope_head_dim + QK_ROPE_HEAD_DIM) ** -0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_head_dim=qk_nope_head_dim + QK_ROPE_HEAD_DIM,
        v_head_dim=v_head_dim,
        kv_b_proj=kv_b_proj,
    )
    impl.process_weights_after_loading(DTYPE)
    return impl, weight


class _FakePrefillMetadata:
    def __init__(
        self, block_table, cpu_seq_lens, query_start_loc, req_to_token, req_pool_indices
    ):
        self.block_table = block_table
        self.cpu_seq_lens = cpu_seq_lens
        query_start_loc_i64 = query_start_loc.to(torch.int64)
        extend_seq_lens = query_start_loc_i64[1:] - query_start_loc_i64[:-1]
        self.extend_seq_lens = extend_seq_lens
        self.extend_start_loc = query_start_loc_i64[:-1]
        self.max_len_extend = int(extend_seq_lens.max().item())
        self.req_to_token = req_to_token
        self.req_pool_indices = req_pool_indices


class _FakeDecodeMetadata:
    def __init__(self, block_table, seq_lens, req_to_token, req_pool_indices):
        self.block_table = block_table
        self.seq_lens_i64 = seq_lens.to(torch.int64)
        self.req_to_token = req_to_token
        self.req_pool_indices = req_pool_indices
        self.num_kv_splits = _compute_num_kv_splits(
            int(seq_lens.max().item()), current_platform.num_compute_units()
        )


class _FakeAttnMetadata:
    def __init__(self, decode=None, prefill=None, max_seq_len=0):
        self.decode = decode
        self.prefill = prefill
        self.max_seq_len = max_seq_len


def test_amx_mla_impl_forward_mqa_matches_reference(default_vllm_config):
    """forward_mqa receives already-absorbed Q (as MLAAttention.forward_impl
    would produce via layer.W_UK_T) and must reproduce plain per-head MLA
    decode attention through W_UK/W_UV."""
    set_random_seed(1)
    num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank = 4, 32, 32, 64
    impl, kv_b_weight = _make_amx_mla_impl(
        num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank
    )
    w_uk, w_uv = kv_b_weight.T.view(
        kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim
    ).split([qk_nope_head_dim, v_head_dim], dim=-1)

    seq_lens = [1, 40, 5]
    num_seqs = len(seq_lens)
    block_size = 32
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks
    head_size = kv_lora_rank + QK_ROPE_HEAD_DIM

    kv_cache = torch.randn(num_blocks, block_size, head_size, dtype=DTYPE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)

    q_nope = torch.randn(num_seqs, num_heads, qk_nope_head_dim, dtype=DTYPE)
    q_pe = torch.randn(num_seqs, num_heads, QK_ROPE_HEAD_DIM, dtype=DTYPE)
    # Pre-absorb, mirroring MLAAttention.forward_impl's own bmm against W_UK_T.
    ql_nope = torch.einsum("bnp,lnp->bnl", q_nope.float(), w_uk.float()).to(DTYPE)

    attn_metadata = _FakeAttnMetadata(
        decode=_FakeDecodeMetadata(
            block_table=block_table,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            req_to_token=req_to_token,
            req_pool_indices=torch.arange(num_seqs, dtype=torch.int64),
        ),
        max_seq_len=max(seq_lens),
    )
    o, lse = impl.forward_mqa((ql_nope, q_pe), kv_cache, attn_metadata, layer=None)
    assert lse is None
    # De-absorb, mirroring MLAAttention.forward_impl's own bmm against
    # layer._v_up_proj, to get back into real per-head V space for
    # comparison against a from-scratch reference.
    out_real = torch.einsum("bnl,lnv->bnv", o.float(), w_uv.float())

    flat_cache = _flatten_cache(kv_cache)
    scale = impl.scale
    for i, seq_len in enumerate(seq_lens):
        token_ids = req_to_token[i, :seq_len]
        keys_latent = flat_cache[token_ids]  # (L, head_size)
        k_nope_latent, k_pe = keys_latent.split(
            [kv_lora_rank, QK_ROPE_HEAD_DIM], dim=-1
        )
        k_real = torch.einsum("lp,pnd->nld", k_nope_latent.float(), w_uk.float())
        k_pe_b = k_pe.float().unsqueeze(0).expand(num_heads, -1, -1)
        k_full = torch.cat([k_real, k_pe_b], dim=-1)  # (num_heads, L, qk_head_dim)
        q_full = torch.cat(
            [q_nope[i], q_pe[i]], dim=-1
        ).float()  # (num_heads, qk_head_dim)
        scores = torch.einsum("nd,nld->nl", q_full, k_full) * scale
        probs = torch.softmax(scores, dim=-1)
        v_real = torch.einsum("lp,pnv->nlv", k_nope_latent.float(), w_uv.float())
        ref = torch.einsum("nl,nlv->nv", probs, v_real)
        torch.testing.assert_close(out_real[i], ref, atol=_IMPL_ATOL, rtol=_IMPL_RTOL)


def test_amx_mla_impl_forward_mha_matches_reference(default_vllm_config):
    """forward_mha receives raw unabsorbed Q and must attend correctly for a
    prefill batch with a mix of fresh and continued (cached-prefix)
    sequences."""
    set_random_seed(2)
    num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank = 4, 32, 32, 64
    impl, kv_b_weight = _make_amx_mla_impl(
        num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank
    )
    w_uk, w_uv = kv_b_weight.T.view(
        kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim
    ).split([qk_nope_head_dim, v_head_dim], dim=-1)

    prefix_lens = [0, 20]
    extend_lens = [13, 7]
    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    num_seqs = len(prefix_lens)
    block_size = 32
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks
    head_size = kv_lora_rank + QK_ROPE_HEAD_DIM

    kv_cache = torch.randn(num_blocks, block_size, head_size, dtype=DTYPE)
    block_table = torch.randperm(num_blocks).view(num_seqs, max_blocks).to(torch.int64)
    req_to_token = _expand_block_table(block_table, block_size)
    flat_cache = _flatten_cache(kv_cache)

    total_new_tokens = sum(extend_lens)
    query_start_loc = torch.tensor(
        [0, *torch.cumsum(torch.tensor(extend_lens), dim=0).tolist()],
        dtype=torch.int64,
    )

    q_nope = torch.randn(total_new_tokens, num_heads, qk_nope_head_dim, dtype=DTYPE)
    q_pe = torch.randn(total_new_tokens, num_heads, QK_ROPE_HEAD_DIM, dtype=DTYPE)
    q = torch.cat([q_nope, q_pe], dim=-1)

    kv_c_normed = torch.randn(total_new_tokens, kv_lora_rank, dtype=DTYPE)
    k_pe = torch.randn(total_new_tokens, 1, QK_ROPE_HEAD_DIM, dtype=DTYPE)

    # Write the new tokens' latent K/V into the cache ahead of time, mirroring
    # do_kv_cache_update running before forward_mha in real usage.
    for i in range(num_seqs):
        prefix_len = prefix_lens[i]
        ext_len = extend_lens[i]
        new_token_ids = req_to_token[i, prefix_len : prefix_len + ext_len]
        start = query_start_loc[i].item()
        flat_cache[new_token_ids, :kv_lora_rank] = kv_c_normed[start : start + ext_len]
        flat_cache[new_token_ids, kv_lora_rank:] = k_pe[start : start + ext_len, 0]

    attn_metadata = _FakeAttnMetadata(
        prefill=_FakePrefillMetadata(
            block_table=block_table,
            cpu_seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            query_start_loc=query_start_loc,
            req_to_token=req_to_token,
            req_pool_indices=torch.arange(num_seqs, dtype=torch.int64),
        )
    )
    output = torch.empty(total_new_tokens, num_heads * v_head_dim, dtype=DTYPE)
    impl.forward_mha(
        q, kv_c_normed, k_pe, kv_cache, attn_metadata, k_scale=None, output=output
    )
    out_view = output.view(total_new_tokens, num_heads, v_head_dim).float()

    scale = impl.scale
    for i in range(num_seqs):
        prefix_len = prefix_lens[i]
        ext_len = extend_lens[i]
        start = query_start_loc[i].item()
        for t in range(ext_len):
            visible = prefix_len + t + 1
            token_ids = req_to_token[i, :visible]
            keys_latent = flat_cache[token_ids]
            k_nope_latent, k_pe_ref = keys_latent.split(
                [kv_lora_rank, QK_ROPE_HEAD_DIM], dim=-1
            )
            k_real = torch.einsum("lp,pnd->nld", k_nope_latent.float(), w_uk.float())
            k_pe_b = k_pe_ref.float().unsqueeze(0).expand(num_heads, -1, -1)
            k_full = torch.cat([k_real, k_pe_b], dim=-1)
            q_full = torch.cat([q_nope[start + t], q_pe[start + t]], dim=-1).float()
            scores = torch.einsum("nd,nld->nl", q_full, k_full) * scale
            probs = torch.softmax(scores, dim=-1)
            v_real = torch.einsum("lp,pnv->nlv", k_nope_latent.float(), w_uv.float())
            ref = torch.einsum("nl,nlv->nv", probs, v_real)
            torch.testing.assert_close(
                out_view[start + t], ref, atol=_IMPL_ATOL, rtol=_IMPL_RTOL
            )
