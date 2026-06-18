# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HPC attention backend integration tests.

Part 1 - Framework: HpcAttentionImpl.forward() end-to-end, covering
         decode/prefill routing, KV cache write, and mixed batch handling.
Part 2 - RopeNorm + Attention: the full HunYuanAttention-style pipeline
         (hpc.rope_norm_blocked_kvcache → hpc.attention_*_bf16).

KV cache layout: (num_blocks, 2, block_size, num_kv_heads, head_size)
  kv_cache[:, 0] -> key_cache,  kv_cache[:, 1] -> value_cache
"""

import pytest
import torch
import torch.nn as nn

from vllm.platforms import current_platform

if not current_platform.has_device_capability(90):
    pytest.skip(
        reason="HPC attention requires compute capability >= SM90.",
        allow_module_level=True,
    )

try:
    import hpc

    from vllm.v1.attention.backends.hpc_attn import (
        HpcAttentionImpl,
        HpcAttnMetadata,
    )
except ImportError:
    pytest.skip(
        reason="HPC attention requires hpc module.",
        allow_module_level=True,
    )

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.hpc.rope_norm import (
    HpcRopeNorm,
    _hpc_rope_norm_instances,
)
from vllm.utils.torch_utils import set_random_seed

# =====================================================================
# Constants
# =====================================================================

HEAD_SIZE = 128  # HPC only supports head_dim=128
BLOCK_SIZE = 64  # HPC only supports block_size=64
NUM_BLOCKS = 2048

# (num_query_heads, num_kv_heads)
# Prefill supports GQA ratio in {4, 8}
PREFILL_HEADS = [(8, 2), (32, 4), (32, 8)]
# Decode: works when num_kv_heads==1 (any ratio), or ratio==8.
DECODE_HEADS = [(4, 1), (8, 1), (16, 2), (32, 4)]
# rope_norm_store_kv supports GQA ratio == 8 (e.g. (8, 1), (64, 8)).
ROPE_NORM_HEADS = [(8, 1), (64, 8)]


# =====================================================================
# Helpers
# =====================================================================


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Pure PyTorch paged attention with causal mask."""
    block_tables_cpu = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start = 0
    for i in range(len(query_lens)):
        q_len, kv_len = query_lens[i], kv_lens[i]
        q = query[start : start + q_len] * scale

        n_blks = (kv_len + block_size - 1) // block_size
        blk_idx = block_tables_cpu[i, :n_blks]
        k = key_cache[blk_idx].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[blk_idx].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            ratio = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, ratio, dim=1)
            v = torch.repeat_interleave(v, ratio, dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=q.device),
            diagonal=kv_len - q_len + 1,
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))
        start += q_len

    return torch.cat(outputs, dim=0)


def _build_kv_env(
    kv_lens: list[int],
    num_kv_heads: int,
    dtype: torch.dtype,
    *,
    zero_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paged KV cache and block_tables.

    Unused slots in last block are zeroed (HPC decode requirement).
    If zero_cache=True, the entire cache starts as zeros (for prefill write).
    """
    max_kv = max(kv_lens)
    max_blks_per_seq = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_seqs = len(kv_lens)

    if zero_cache:
        kv_cache = torch.zeros(
            NUM_BLOCKS, 2, BLOCK_SIZE, num_kv_heads, HEAD_SIZE, dtype=dtype
        )
    else:
        kv_cache = torch.randn(
            NUM_BLOCKS, 2, BLOCK_SIZE, num_kv_heads, HEAD_SIZE, dtype=dtype
        )

    block_tables = torch.zeros(
        num_seqs, max_blks_per_seq, dtype=torch.int32, device="cuda"
    )
    blk = 0
    for i, kv_len in enumerate(kv_lens):
        n = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(n):
            block_tables[i, b] = blk + b
        remainder = kv_len % BLOCK_SIZE
        if remainder:
            kv_cache[blk + n - 1, :, remainder:, :, :] = 0
        blk += n

    return kv_cache, block_tables


def _slot_mapping_for_seqs(
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """Build slot_mapping: one slot per query token."""
    slots = []
    for i, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        for t in range(q_len):
            pos = kv_len - q_len + t  # decode: kv_len-1; prefill: 0..kv_len-1
            blk = block_tables[i, pos // BLOCK_SIZE].item()
            slots.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
    return torch.tensor(slots, dtype=torch.long, device="cuda")


def _full_kv_slot_mapping(
    kv_lens: list[int],
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """Build slot_mapping covering all kv_len positions (for KV cache write)."""
    slots = []
    for i, kv_len in enumerate(kv_lens):
        for pos in range(kv_len):
            blk = block_tables[i, pos // BLOCK_SIZE].item()
            slots.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
    return torch.tensor(slots, dtype=torch.long, device="cuda")


def _write_kv(key, value, kv_cache, slot_mapping):
    """Write K/V tokens into paged cache via slot_mapping."""
    for t in range(key.shape[0]):
        s = slot_mapping[t].item()
        kv_cache[s // BLOCK_SIZE, 0, s % BLOCK_SIZE] = key[t]
        kv_cache[s // BLOCK_SIZE, 1, s % BLOCK_SIZE] = value[t]


class _MockLayer(nn.Module):
    """Minimal mock of Attention layer for HpcAttentionImpl.forward."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_k_scale", torch.tensor(1.0))
        self.register_buffer("_v_scale", torch.tensor(1.0))


def _make_metadata(
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    hpc_kv_written: bool = False,
) -> "HpcAttnMetadata":
    """Build HpcAttnMetadata (mirrors HpcAttnMetadataBuilder.build)."""
    total = sum(query_lens)
    # Decode seqs come first (query_len <= 1)
    n_dec = 0
    for ql in query_lens:
        if ql <= 1:
            n_dec += 1
        else:
            break
    n_dec_tok = sum(query_lens[:n_dec])
    n_pf = len(query_lens) - n_dec
    n_pf_tok = total - n_dec_tok

    qo_indptr = None
    if n_pf > 0:
        pf_qlens = query_lens[n_dec:]
        cum = [0]
        for ql in pf_qlens:
            cum.append(cum[-1] + ql)
        qo_indptr = torch.tensor(cum, dtype=torch.int32, device="cuda")

    return HpcAttnMetadata(
        num_actual_tokens=total,
        num_decodes=n_dec,
        num_decode_tokens=n_dec_tok,
        num_prefills=n_pf,
        num_prefill_tokens=n_pf_tok,
        max_query_len=max(query_lens),
        slot_mapping=slot_mapping,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        block_table_tensor=block_tables,
        qo_indptr=qo_indptr,
        hpc_kv_written=hpc_kv_written,
    )


def _run_impl_forward(
    query,
    key,
    value,
    kv_cache,
    query_lens,
    kv_lens,
    block_tables,
    slot_mapping,
    num_query_heads,
    num_kv_heads,
):
    """Run HpcAttentionImpl.forward and return result tensor."""
    meta = _make_metadata(query_lens, kv_lens, block_tables, slot_mapping)
    impl = HpcAttentionImpl(
        num_heads=num_query_heads,
        head_size=HEAD_SIZE,
        scale=HEAD_SIZE**-0.5,
        num_kv_heads=num_kv_heads,
        kv_cache_dtype="auto",
    )
    output = torch.empty_like(query)
    return impl.forward(
        layer=_MockLayer(),
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=meta,
        output=output,
    )


# =====================================================================
# Part 1: Framework tests — HpcAttentionImpl.forward
# =====================================================================


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(128, 128)],
        [(7, 7), (33, 33), (129, 129)],
        [(5, 18), (129, 463)],  # chunked prefill
    ],
)
@pytest.mark.parametrize("num_heads", PREFILL_HEADS)
@torch.inference_mode()
def test_impl_forward_prefill(seq_lens, num_heads):
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads

    q_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    tot_q = sum(q_lens)
    tot_kv = sum(kv_lens)

    kv_cache, bt = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    sm = _slot_mapping_for_seqs(q_lens, kv_lens, bt)
    kv_sm = _full_kv_slot_mapping(kv_lens, bt)

    query = torch.randn(tot_q, nqh, HEAD_SIZE, dtype=dtype)
    key = torch.randn(tot_kv, nkh, HEAD_SIZE, dtype=dtype)
    value = torch.randn(tot_kv, nkh, HEAD_SIZE, dtype=dtype)
    _write_kv(key, value, kv_cache, kv_sm)

    result = _run_impl_forward(
        query, key, value, kv_cache, q_lens, kv_lens, bt, sm, nqh, nkh
    )
    ref = ref_paged_attn(
        query, kv_cache[:, 0], kv_cache[:, 1], q_lens, kv_lens, bt, HEAD_SIZE**-0.5
    )
    torch.testing.assert_close(result[:tot_q], ref, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "kv_lens",
    [
        [128],
        [64, 256, 512],
        [1, 37, 128, 2011],
    ],
)
@pytest.mark.parametrize("num_heads", DECODE_HEADS)
@torch.inference_mode()
def test_impl_forward_decode(kv_lens, num_heads):
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads
    n = len(kv_lens)
    q_lens = [1] * n

    kv_cache, bt = _build_kv_env(kv_lens, nkh, dtype)
    sm = _slot_mapping_for_seqs(q_lens, kv_lens, bt)

    query = torch.randn(n, nqh, HEAD_SIZE, dtype=dtype)
    key = torch.randn(n, nkh, HEAD_SIZE, dtype=dtype)
    value = torch.randn(n, nkh, HEAD_SIZE, dtype=dtype)

    result = _run_impl_forward(
        query, key, value, kv_cache, q_lens, kv_lens, bt, sm, nqh, nkh
    )
    ref = ref_paged_attn(
        query, kv_cache[:, 0], kv_cache[:, 1], q_lens, kv_lens, bt, HEAD_SIZE**-0.5
    )
    torch.testing.assert_close(result[:n], ref, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "decode_kv_lens, prefill_seq_lens",
    [
        ([128, 256], [(64, 64)]),
        ([64, 128, 256], [(32, 32), (128, 128)]),
        ([1, 523, 37], [(5, 18), (129, 463)]),
    ],
)
@pytest.mark.parametrize("num_heads", DECODE_HEADS)
@torch.inference_mode()
def test_impl_forward_mixed(decode_kv_lens, prefill_seq_lens, num_heads):
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads

    q_lens = [1] * len(decode_kv_lens) + [s[0] for s in prefill_seq_lens]
    kv_lens = list(decode_kv_lens) + [s[1] for s in prefill_seq_lens]
    n_dec = len(decode_kv_lens)
    tot_q = sum(q_lens)

    kv_cache, bt = _build_kv_env(kv_lens, nkh, dtype)
    sm = _slot_mapping_for_seqs(q_lens, kv_lens, bt)

    # Write prefill KV into cache (full kv_len tokens per seq)
    pf_kv_total = sum(kv_lens[n_dec:])
    k_pf = torch.randn(pf_kv_total, nkh, HEAD_SIZE, dtype=dtype)
    v_pf = torch.randn(pf_kv_total, nkh, HEAD_SIZE, dtype=dtype)
    pf_sm = _full_kv_slot_mapping(kv_lens[n_dec:], bt[n_dec:])
    _write_kv(k_pf, v_pf, kv_cache, pf_sm)

    query = torch.randn(tot_q, nqh, HEAD_SIZE, dtype=dtype)
    key = torch.randn(tot_q, nkh, HEAD_SIZE, dtype=dtype)
    value = torch.randn(tot_q, nkh, HEAD_SIZE, dtype=dtype)

    result = _run_impl_forward(
        query, key, value, kv_cache, q_lens, kv_lens, bt, sm, nqh, nkh
    )
    ref = ref_paged_attn(
        query, kv_cache[:, 0], kv_cache[:, 1], q_lens, kv_lens, bt, HEAD_SIZE**-0.5
    )
    torch.testing.assert_close(result[:tot_q], ref, atol=1.5e-2, rtol=1e-2)


@torch.inference_mode()
def test_impl_forward_none_metadata():
    """Output should be zeroed when attn_metadata is None."""
    torch.set_default_device("cuda")
    dtype = torch.bfloat16
    nqh, nkh = 8, 1

    impl = HpcAttentionImpl(
        num_heads=nqh,
        head_size=HEAD_SIZE,
        scale=HEAD_SIZE**-0.5,
        num_kv_heads=nkh,
        kv_cache_dtype="auto",
    )
    kv_cache = torch.randn(32, 2, BLOCK_SIZE, nkh, HEAD_SIZE, dtype=dtype)
    q = torch.randn(4, nqh, HEAD_SIZE, dtype=dtype)
    out = torch.ones(4, nqh, HEAD_SIZE, dtype=dtype)

    result = impl.forward(
        layer=_MockLayer(),
        query=q,
        key=q[:, :nkh],
        value=q[:, :nkh],
        kv_cache=kv_cache,
        attn_metadata=None,
        output=out,
    )
    assert torch.all(result == 0)


def test_impl_rejects_invalid_head_size():
    with pytest.raises(ValueError, match="head_dim=128"):
        HpcAttentionImpl(num_heads=8, head_size=64, scale=0.125, num_kv_heads=1)


def test_impl_rejects_invalid_gqa_ratio():
    with pytest.raises(ValueError, match="head_per_group"):
        HpcAttentionImpl(
            num_heads=6, head_size=128, scale=HEAD_SIZE**-0.5, num_kv_heads=2
        )


def test_impl_rejects_invalid_kv_cache_dtype():
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        HpcAttentionImpl(
            num_heads=8,
            head_size=128,
            scale=HEAD_SIZE**-0.5,
            num_kv_heads=1,
            kv_cache_dtype="int8",
        )


# =====================================================================
# Part 2: RopeNorm + Attention integration tests
#
# Verifies the full HunYuanAttention-style pipeline:
#   hpc.rope_norm_blocked_kvcache → hpc.attention_*_bf16
# against a PyTorch reference (RoPE → RMSNorm → manual KV write → attn).
# =====================================================================


def _make_cos_sin_cache(max_pos: int) -> torch.Tensor:
    """RotaryEmbedding cos_sin_cache: (max_pos, head_dim), neox-style."""
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, HEAD_SIZE, 2, dtype=torch.float) / HEAD_SIZE)
    )
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _ref_rope_neox(x, cos_sin_cache, positions):
    """Apply neox-style RoPE. x: (num_tokens, num_heads, head_dim)."""
    half = HEAD_SIZE // 2
    cs = cos_sin_cache[positions]
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:].unsqueeze(1)
    return torch.cat(
        [
            x[..., :half] * cos - x[..., half:] * sin,
            x[..., half:] * cos + x[..., :half] * sin,
        ],
        dim=-1,
    ).to(x.dtype)


def _ref_rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / rms(x)."""
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * weight).to(x.dtype)


def _ref_rope_norm_write_kv(
    qkv,
    cos_sin_cache,
    positions,
    kv_cache,
    block_tables,
    kv_lens,
    query_lens,
    nqh,
    nkh,
    use_qk_norm,
    qw,
    kw,
):
    """Reference: RoPE → optional RMSNorm → write K/V to cache → return Q."""
    q_sz = nqh * HEAD_SIZE
    kv_sz = nkh * HEAD_SIZE
    qf, kf, vf = qkv.split([q_sz, kv_sz, kv_sz], dim=-1)
    q = _ref_rope_neox(qf.view(-1, nqh, HEAD_SIZE), cos_sin_cache, positions)
    k = _ref_rope_neox(kf.view(-1, nkh, HEAD_SIZE), cos_sin_cache, positions)
    v = vf.view(-1, nkh, HEAD_SIZE)

    if use_qk_norm and qw is not None:
        q = _ref_rms_norm(q, qw)
        k = _ref_rms_norm(k, kw)

    tok = 0
    for si in range(len(kv_lens)):
        for t in range(query_lens[si]):
            pos = kv_lens[si] - query_lens[si] + t
            bi = block_tables[si, pos // BLOCK_SIZE].item()
            off = pos % BLOCK_SIZE
            kv_cache[bi, 0, off] = k[tok]
            kv_cache[bi, 1, off] = v[tok]
            tok += 1
    return q


def _positions_for_seqs(query_lens, kv_lens):
    """Position IDs per token. Decode: [kv_len-1]; Prefill: [0..kv_len-1]."""
    pos: list[int] = []
    for ql, kl in zip(query_lens, kv_lens):
        pos.extend(range(kl - ql, kl))
    return torch.tensor(pos, dtype=torch.long, device="cuda")


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(128, 128)],
        [(7, 7), (33, 33), (129, 129)],
        [(5, 18), (129, 463)],
    ],
)
@pytest.mark.parametrize("num_heads", ROPE_NORM_HEADS)
@pytest.mark.parametrize("use_qk_norm", [True, False])
@torch.inference_mode()
def test_rope_norm_prefill(seq_lens, num_heads, use_qk_norm):
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads
    scale = HEAD_SIZE**-0.5

    q_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    tot = sum(q_lens)

    cs = _make_cos_sin_cache(max(kv_lens) + 128).cuda()
    positions = _positions_for_seqs(q_lens, kv_lens)
    qw = (
        torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
        if use_qk_norm
        else None
    )
    kw = (
        torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
        if use_qk_norm
        else None
    )

    qkv = torch.randn(tot, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    # HPC path
    kv_hpc, bt = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    cu = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    sl = torch.tensor(kv_lens, dtype=torch.int32)

    q_hpc = hpc.rope_norm_store_kv(
        kv_hpc[:, 0],
        kv_hpc[:, 1],
        qkv,
        cs,
        sl,
        cu,
        bt,
        True,
        q_norm_weight=qw,
        k_norm_weight=kw,
        qk_norm_policy=(1 if use_qk_norm else 0),
    )

    out_hpc = torch.empty(tot, nqh, HEAD_SIZE, dtype=dtype)
    hpc.attention_with_kvcache_prefill_bf16(
        q_hpc, kv_hpc[:, 0], kv_hpc[:, 1], cu, bt, sl, max(q_lens), output=out_hpc
    )

    # Reference path
    kv_ref, _ = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    q_ref = _ref_rope_norm_write_kv(
        qkv, cs, positions, kv_ref, bt, kv_lens, q_lens, nqh, nkh, use_qk_norm, qw, kw
    )
    out_ref = ref_paged_attn(
        q_ref, kv_ref[:, 0], kv_ref[:, 1], q_lens, kv_lens, bt, scale
    )

    torch.testing.assert_close(out_hpc, out_ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "kv_lens",
    [
        [128],
        [64, 256, 512],
        [37, 128, 523],
    ],
)
@pytest.mark.parametrize("num_heads", ROPE_NORM_HEADS)
@pytest.mark.parametrize("use_qk_norm", [True, False])
@torch.inference_mode()
def test_rope_norm_decode(kv_lens, num_heads, use_qk_norm):
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads
    scale = HEAD_SIZE**-0.5
    n = len(kv_lens)
    q_lens = [1] * n

    cs = _make_cos_sin_cache(max(kv_lens) + 128).cuda()
    positions = _positions_for_seqs(q_lens, kv_lens)
    qw = (
        torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
        if use_qk_norm
        else None
    )
    kw = (
        torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
        if use_qk_norm
        else None
    )

    qkv = torch.randn(n, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    # HPC path (pre-fill history with random data)
    kv_hpc, bt = _build_kv_env(kv_lens, nkh, dtype)
    kv_ref = kv_hpc.clone()
    sl = torch.tensor(kv_lens, dtype=torch.int32)

    q_hpc = hpc.rope_norm_store_kv(
        kv_hpc[:, 0],
        kv_hpc[:, 1],
        qkv,
        cs,
        sl,
        torch.arange(sl.numel() + 1, dtype=torch.int32, device=sl.device),
        bt,
        False,
        q_norm_weight=qw,
        k_norm_weight=kw,
        qk_norm_policy=(1 if use_qk_norm else 0),
    )

    out_hpc = torch.empty(n, nqh, HEAD_SIZE, dtype=dtype)
    hpc.attention_decode_bf16(
        q_hpc,
        kv_hpc[:, 0],
        kv_hpc[:, 1],
        bt,
        sl,
        output=out_hpc,
        new_kv_included=True,
        splitk=True,
    )

    # Reference path
    q_ref = _ref_rope_norm_write_kv(
        qkv, cs, positions, kv_ref, bt, kv_lens, q_lens, nqh, nkh, use_qk_norm, qw, kw
    )
    out_ref = ref_paged_attn(
        q_ref, kv_ref[:, 0], kv_ref[:, 1], q_lens, kv_lens, bt, scale
    )

    torch.testing.assert_close(out_hpc, out_ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "decode_kv_lens, prefill_seq_lens",
    [
        ([128, 256], [(64, 64)]),
        ([64, 128], [(32, 32), (128, 128)]),
    ],
)
@pytest.mark.parametrize("num_heads", ROPE_NORM_HEADS)
@pytest.mark.parametrize("use_qk_norm", [True, False])
@torch.inference_mode()
def test_rope_norm_mixed(decode_kv_lens, prefill_seq_lens, num_heads, use_qk_norm):
    """RopeNorm + mixed decode/prefill (same batch ordering as vLLM)."""
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads
    scale = HEAD_SIZE**-0.5

    n_dec = len(decode_kv_lens)
    q_lens = [1] * n_dec + [s[0] for s in prefill_seq_lens]
    kv_lens = list(decode_kv_lens) + [s[1] for s in prefill_seq_lens]
    tot = sum(q_lens)
    n_dec_tok = n_dec

    cs = _make_cos_sin_cache(max(kv_lens) + 128).cuda()
    qw = (
        torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
        if use_qk_norm
        else None
    )
    kw = (
        torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
        if use_qk_norm
        else None
    )
    qkv = torch.randn(tot, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    # Build cache: random for decode history, zero for prefill blocks
    kv_hpc, bt = _build_kv_env(kv_lens, nkh, dtype)
    blk = 0
    for i, kl in enumerate(kv_lens):
        nb = (kl + BLOCK_SIZE - 1) // BLOCK_SIZE
        if i >= n_dec:
            for b in range(nb):
                kv_hpc[blk + b] = 0
        blk += nb

    kv_ref = kv_hpc.clone()
    sl = torch.tensor(kv_lens, dtype=torch.int32)
    out_hpc = torch.empty(tot, nqh, HEAD_SIZE, dtype=dtype)

    # HPC: prefill part
    if prefill_seq_lens:
        pf_qlens = q_lens[n_dec:]
        cu_pf = torch.tensor([0] + pf_qlens, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        q_pf = hpc.rope_norm_store_kv(
            kv_hpc[:, 0],
            kv_hpc[:, 1],
            qkv[n_dec_tok:],
            cs,
            sl[n_dec:],
            cu_pf,
            bt[n_dec:],
            True,
            q_norm_weight=qw,
            k_norm_weight=kw,
            qk_norm_policy=(1 if use_qk_norm else 0),
        )
        hpc.attention_with_kvcache_prefill_bf16(
            q_pf,
            kv_hpc[:, 0],
            kv_hpc[:, 1],
            cu_pf,
            bt[n_dec:],
            sl[n_dec:],
            max(pf_qlens),
            output=out_hpc[n_dec_tok:],
        )

    # HPC: decode part
    if decode_kv_lens:
        q_dc = hpc.rope_norm_store_kv(
            kv_hpc[:, 0],
            kv_hpc[:, 1],
            qkv[:n_dec_tok],
            cs,
            sl[:n_dec],
            torch.arange(n_dec + 1, dtype=torch.int32, device=sl.device),
            bt[:n_dec],
            False,
            q_norm_weight=qw,
            k_norm_weight=kw,
            qk_norm_policy=(1 if use_qk_norm else 0),
        )
        hpc.attention_decode_bf16(
            q_dc,
            kv_hpc[:, 0],
            kv_hpc[:, 1],
            bt[:n_dec],
            sl[:n_dec],
            output=out_hpc[:n_dec_tok],
            new_kv_included=True,
            splitk=True,
        )

    # Reference path
    positions = _positions_for_seqs(q_lens, kv_lens)
    q_ref = _ref_rope_norm_write_kv(
        qkv, cs, positions, kv_ref, bt, kv_lens, q_lens, nqh, nkh, use_qk_norm, qw, kw
    )
    out_ref = ref_paged_attn(
        q_ref, kv_ref[:, 0], kv_ref[:, 1], q_lens, kv_lens, bt, scale
    )

    torch.testing.assert_close(out_hpc, out_ref, atol=2e-2, rtol=1e-2)


# =====================================================================
# Part 3: FP8 Framework-level integration tests
#
# Tests the REAL framework layer classes:
#   HpcRopeNorm._forward_impl()  →  attn_metadata.hpc_*  →
#   HpcAttentionImpl.forward()
#
# This reproduces the exact same call-chain as the real model
# (hunyuan_v1.py), including:
#   - ForwardContext with attn_metadata dict (shared metadata object)
#   - HpcRopeNorm writing FP8 KV + setting hpc_* fields on metadata
#   - HpcAttentionImpl consuming hpc_* fields and resetting them
#   - Attention.forward() output dtype (output_dtype = query.dtype)
#   - kv_cache allocated as uint8, viewed as float8_e4m3fn
#
# Reference: the already-validated BF16 operator-level pipeline from
# Part 2, which is known to be correct.
# =====================================================================


# FP8 KV cache path requires VLLM_ENABLE_HPC_ROPE_NORM=1 (HpcRopeNorm dependency).
# Set it here so that HpcAttentionImpl(kv_cache_dtype="fp8_e4m3") does not
# raise ValueError during test initialization.
envs.VLLM_ENABLE_HPC_ROPE_NORM = True

FP8_HEADS = [(8, 1)]

# Minimal VllmConfig needed for CustomOp (HpcRopeNorm) initialization.
# CustomOp.__init__ -> dispatch_forward -> get_cached_compilation_config()
_test_vllm_config = VllmConfig()


class _MockAttnLayer(nn.Module):
    """Mock the Attention layer object that sits in no_compile_layers.

    In the real framework:
    - forward_context.no_compile_layers[layer_name] is the Attention instance
    - attn_layer.kv_cache[virtual_engine] is the KV cache tensor
    - attn_layer._k_scale / _v_scale are the quantization scales
    """

    def __init__(self, kv_cache: torch.Tensor):
        super().__init__()
        self.kv_cache = [kv_cache]  # list indexed by virtual_engine
        self.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))


def _build_fp8_kv_cache(
    kv_lens: list[int],
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate uint8 kv_cache (like the real framework) + block_tables."""
    max_kv = max(kv_lens)
    max_blks = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_seqs = len(kv_lens)

    kv_cache = torch.zeros(
        NUM_BLOCKS,
        2,
        BLOCK_SIZE,
        num_kv_heads,
        HEAD_SIZE,
        dtype=torch.uint8,
        device="cuda",
    )
    block_tables = torch.zeros(
        num_seqs,
        max_blks,
        dtype=torch.int32,
        device="cuda",
    )
    blk = 0
    for i, kv_len in enumerate(kv_lens):
        n = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(n):
            block_tables[i, b] = blk + b
        blk += n
    return kv_cache, block_tables


def _create_rope_norm(
    nqh: int,
    nkh: int,
    cos_sin_cache: torch.Tensor,
    qw: torch.Tensor | None,
    kw: torch.Tensor | None,
    layer_name: str,
) -> HpcRopeNorm:
    """Create and register an HpcRopeNorm instance (FP8 mode).

    HpcRopeNorm is a CustomOp subclass; its __init__ needs VllmConfig
    to dispatch the forward method.
    """
    # Build dummy fallback norm modules to carry weights
    fallback_qnorm = None
    fallback_knorm = None
    if qw is not None:
        fallback_qnorm = nn.Module()
        fallback_qnorm.weight = nn.Parameter(qw.clone())
    if kw is not None:
        fallback_knorm = nn.Module()
        fallback_knorm.weight = nn.Parameter(kw.clone())

    with set_current_vllm_config(_test_vllm_config):
        rope_norm = HpcRopeNorm(
            num_heads=nqh,
            num_kv_heads=nkh,
            head_dim=HEAD_SIZE,
            cos_sin_cache=cos_sin_cache,
            use_qk_norm=True,  # w8c8_dqskv requires qk_norm
            fallback_qnorm=fallback_qnorm,
            fallback_knorm=fallback_knorm,
            kv_cache_dtype="fp8_e4m3",
        )
    rope_norm.process_weights_after_loading()
    rope_norm.register_layer_name(layer_name)
    return rope_norm


def _run_fp8_framework_prefill(
    qkv: torch.Tensor,
    kv_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_tables: torch.Tensor,
    q_lens: list[int],
    kv_lens: list[int],
    nqh: int,
    nkh: int,
    qw: torch.Tensor | None,
    kw: torch.Tensor | None,
) -> torch.Tensor:
    """Run the full FP8 framework pipeline for pure prefill.

    Replicates the exact call sequence in the real model:
      1. HpcRopeNorm._forward_impl(qkv, kv_cache, attn_metadata, attn_layer, output)
         → writes FP8 KV into kv_cache
         → sets attn_metadata.hpc_kv_written, hpc_prefill_q_scale, etc.
         → returns FP8 query in output
      2. HpcAttentionImpl.forward(layer, query, key, value, kv_cache, attn_metadata,
      output)
         → reads hpc_* fields from attn_metadata
         → runs attention_prefill_fp8
         → resets hpc_* fields
    """
    layer_name = "test_fp8_layer"
    tot = sum(q_lens)

    # Build metadata (pure prefill: 0 decodes)
    qo_indptr = torch.tensor([0] + q_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    metadata = HpcAttnMetadata(
        num_actual_tokens=tot,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=len(q_lens),
        num_prefill_tokens=tot,
        max_query_len=max(q_lens),
        slot_mapping=torch.zeros(tot, dtype=torch.long, device="cuda"),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        block_table_tensor=block_tables,
        qo_indptr=qo_indptr,
    )

    # Setup framework objects
    mock_layer = _MockAttnLayer(kv_cache)
    rope_norm = _create_rope_norm(nqh, nkh, cos_sin_cache, qw, kw, layer_name)

    attn_impl = HpcAttentionImpl(
        num_heads=nqh,
        head_size=HEAD_SIZE,
        scale=HEAD_SIZE**-0.5,
        num_kv_heads=nkh,
        kv_cache_dtype="fp8_e4m3",
    )

    # Build ForwardContext (same dict-based metadata as real framework)
    forward_ctx = ForwardContext(
        no_compile_layers={layer_name: mock_layer},
        attn_metadata={layer_name: metadata},
        slot_mapping={},
    )

    with override_forward_context(forward_ctx):
        # Step 1: HpcRopeNorm._forward_impl()
        # This is what hpc_rope_norm_forward() calls internally
        q_output = torch.empty(
            tot,
            nqh,
            HEAD_SIZE,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        rope_norm._forward_impl(
            qkv,
            kv_cache,
            metadata,
            mock_layer,
            q_output,
        )

        # Verify that metadata fields were set by RopeNorm
        assert metadata.hpc_kv_written is True, (
            "RopeNorm should set hpc_kv_written=True"
        )
        assert metadata.hpc_prefill_q_scale is not None, (
            "RopeNorm should set hpc_prefill_q_scale for prefill"
        )

        # Step 2: HpcAttentionImpl.forward()
        # query is FP8 (from RopeNorm), key/value are dummy (not used)
        dummy_kv = torch.zeros(
            tot,
            nkh,
            HEAD_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        attn_output = torch.empty(
            tot,
            nqh,
            HEAD_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        attn_impl.forward(
            layer=mock_layer,
            query=q_output,
            key=dummy_kv,
            value=dummy_kv,
            kv_cache=kv_cache,
            attn_metadata=metadata,
            output=attn_output,
        )

        # Verify that metadata fields were reset by HpcAttentionImpl
        assert metadata.hpc_kv_written is False, (
            "HpcAttentionImpl should reset hpc_kv_written=False"
        )
        assert metadata.hpc_prefill_q_scale is None, (
            "HpcAttentionImpl should reset hpc_prefill_q_scale=None"
        )

    return attn_output


def _run_fp8_framework_decode(
    qkv_hist: torch.Tensor,
    qkv_dec: torch.Tensor,
    kv_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_tables: torch.Tensor,
    history_lens: list[int],
    kv_lens: list[int],
    nqh: int,
    nkh: int,
    qw: torch.Tensor | None,
    kw: torch.Tensor | None,
) -> torch.Tensor:
    """Run the full FP8 framework pipeline: prefill history then decode.

    Step 1: Prefill history tokens to populate KV cache (FP8).
    Step 2: Decode 1 new token per sequence.
    """
    layer_name = "test_fp8_layer"
    n = len(kv_lens)
    tot_hist = sum(history_lens)

    mock_layer = _MockAttnLayer(kv_cache)
    rope_norm = _create_rope_norm(nqh, nkh, cos_sin_cache, qw, kw, layer_name)

    attn_impl = HpcAttentionImpl(
        num_heads=nqh,
        head_size=HEAD_SIZE,
        scale=HEAD_SIZE**-0.5,
        num_kv_heads=nkh,
        kv_cache_dtype="fp8_e4m3",
    )

    # --- Step 1: Prefill history ---
    hist_qo_indptr = torch.tensor(
        [0] + history_lens, dtype=torch.int32, device="cuda"
    ).cumsum(0, dtype=torch.int32)
    hist_metadata = HpcAttnMetadata(
        num_actual_tokens=tot_hist,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=n,
        num_prefill_tokens=tot_hist,
        max_query_len=max(history_lens),
        slot_mapping=torch.zeros(tot_hist, dtype=torch.long, device="cuda"),
        seq_lens=torch.tensor(history_lens, dtype=torch.int32, device="cuda"),
        block_table_tensor=block_tables,
        qo_indptr=hist_qo_indptr,
    )

    hist_ctx = ForwardContext(
        no_compile_layers={layer_name: mock_layer},
        attn_metadata={layer_name: hist_metadata},
        slot_mapping={},
    )

    with override_forward_context(hist_ctx):
        hist_q_output = torch.empty(
            tot_hist,
            nqh,
            HEAD_SIZE,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        rope_norm._forward_impl(
            qkv_hist,
            kv_cache,
            hist_metadata,
            mock_layer,
            hist_q_output,
        )
        # We only need the KV write; skip attention for history.
        # Reset metadata manually (as HpcAttentionImpl would do).
        hist_metadata.hpc_kv_written = False

    # --- Step 2: Decode ---
    dec_metadata = HpcAttnMetadata(
        num_actual_tokens=n,
        num_decodes=n,
        num_decode_tokens=n,
        num_prefills=0,
        num_prefill_tokens=0,
        max_query_len=1,
        slot_mapping=torch.zeros(n, dtype=torch.long, device="cuda"),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        block_table_tensor=block_tables,
        qo_indptr=None,
    )

    dec_ctx = ForwardContext(
        no_compile_layers={layer_name: mock_layer},
        attn_metadata={layer_name: dec_metadata},
        slot_mapping={},
    )

    with override_forward_context(dec_ctx):
        dec_q_output = torch.empty(
            n,
            nqh,
            HEAD_SIZE,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        rope_norm._forward_impl(
            qkv_dec,
            kv_cache,
            dec_metadata,
            mock_layer,
            dec_q_output,
        )

        assert dec_metadata.hpc_kv_written is True
        assert dec_metadata.hpc_decode_q_scale is not None
        assert dec_metadata.hpc_split_k_flag is not None

        dummy_kv = torch.zeros(n, nkh, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
        attn_output = torch.empty(
            n, nqh, HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        attn_impl.forward(
            layer=mock_layer,
            query=dec_q_output,
            key=dummy_kv,
            value=dummy_kv,
            kv_cache=kv_cache,
            attn_metadata=dec_metadata,
            output=attn_output,
        )

        assert dec_metadata.hpc_kv_written is False

    return attn_output


# -----------------------------------------------------------------
# BF16 operator-level reference (reused from Part 2, known correct)
# -----------------------------------------------------------------


def _bf16_ref_prefill(
    qkv,
    kv_cache,
    cos_sin_cache,
    block_tables,
    q_lens,
    kv_lens,
    nqh,
    nkh,
    qw,
    kw,
):
    """BF16 operator-level prefill reference."""
    tot = sum(q_lens)
    cu = torch.tensor([0] + q_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    sl = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    q = hpc.rope_norm_store_kv(
        kv_cache[:, 0],
        kv_cache[:, 1],
        qkv,
        cos_sin_cache,
        sl,
        cu,
        block_tables,
        True,
        q_norm_weight=qw,
        k_norm_weight=kw,
        qk_norm_policy=1,
    )
    output = torch.empty(tot, nqh, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    hpc.attention_with_kvcache_prefill_bf16(
        q,
        kv_cache[:, 0],
        kv_cache[:, 1],
        cu,
        block_tables,
        sl,
        max(q_lens),
        output=output,
    )
    return output


def _bf16_ref_prefill_then_decode(
    qkv_hist,
    qkv_dec,
    kv_cache,
    cos_sin_cache,
    block_tables,
    history_lens,
    kv_lens,
    nqh,
    nkh,
    qw,
    kw,
):
    """BF16 operator-level: prefill history then decode."""
    n = len(kv_lens)
    # Prefill history
    cu_hist = torch.tensor([0] + history_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    sl_hist = torch.tensor(history_lens, dtype=torch.int32, device="cuda")
    hpc.rope_norm_store_kv(
        kv_cache[:, 0],
        kv_cache[:, 1],
        qkv_hist,
        cos_sin_cache,
        sl_hist,
        cu_hist,
        block_tables,
        True,
        q_norm_weight=qw,
        k_norm_weight=kw,
        qk_norm_policy=1,
    )
    # Decode
    sl_dec = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    q_dec = hpc.rope_norm_store_kv(
        kv_cache[:, 0],
        kv_cache[:, 1],
        qkv_dec,
        cos_sin_cache,
        sl_dec,
        torch.arange(sl_dec.numel() + 1, dtype=torch.int32, device=sl_dec.device),
        block_tables,
        False,
        q_norm_weight=qw,
        k_norm_weight=kw,
        qk_norm_policy=1,
    )
    output = torch.empty(n, nqh, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    hpc.attention_decode_bf16(
        q_dec,
        kv_cache[:, 0],
        kv_cache[:, 1],
        block_tables,
        sl_dec,
        output=output,
        new_kv_included=True,
        splitk=True,
    )
    return output


# -----------------------------------------------------------------
# Test: FP8 framework prefill
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(128, 128)],
        [(7, 7), (33, 33), (129, 129)],
        [(5, 18), (129, 463)],
    ],
)
@pytest.mark.parametrize("num_heads", FP8_HEADS)
@torch.inference_mode()
def test_fp8_framework_prefill(seq_lens, num_heads):
    """FP8 framework-level prefill vs BF16 operator-level reference.

    Exercises the full call chain:
      HpcRopeNorm._forward_impl() → attn_metadata.hpc_* →
      HpcAttentionImpl.forward()
    """
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads

    q_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    tot = sum(q_lens)

    cs = _make_cos_sin_cache(max(kv_lens) + 128).cuda()
    qw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
    kw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")

    qkv = torch.randn(tot, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    # FP8 framework path: uint8 kv_cache
    kv_fp8, bt = _build_fp8_kv_cache(kv_lens, nkh)
    out_fp8 = _run_fp8_framework_prefill(
        qkv,
        kv_fp8,
        cs,
        bt,
        q_lens,
        kv_lens,
        nqh,
        nkh,
        qw,
        kw,
    )

    # BF16 operator-level reference
    kv_bf16, _ = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    out_bf16 = _bf16_ref_prefill(
        qkv,
        kv_bf16,
        cs,
        bt,
        q_lens,
        kv_lens,
        nqh,
        nkh,
        qw,
        kw,
    )

    # Verify output is bf16 (not FP8!)
    assert out_fp8.dtype == torch.bfloat16, (
        f"Framework output should be bf16, got {out_fp8.dtype}"
    )
    assert not torch.isnan(out_fp8).any(), "Output contains NaN"
    assert not torch.isinf(out_fp8).any(), "Output contains Inf"

    torch.testing.assert_close(out_fp8, out_bf16, atol=0.25, rtol=0.15)


# -----------------------------------------------------------------
# Test: FP8 framework decode (with FP8-prefilled history)
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kv_lens",
    [
        [128],
        [64, 256, 512],
        [37, 128, 523],
    ],
)
@pytest.mark.parametrize("num_heads", FP8_HEADS)
@torch.inference_mode()
def test_fp8_framework_decode(kv_lens, num_heads):
    """FP8 framework-level decode vs BF16 operator-level reference.

    Both paths populate history via their own prefill, then decode
    1 new token per sequence.
    """
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads
    n = len(kv_lens)

    history_lens = [kl - 1 for kl in kv_lens]
    assert all(h > 0 for h in history_lens)

    cs = _make_cos_sin_cache(max(kv_lens) + 128).cuda()
    qw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
    kw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")

    tot_hist = sum(history_lens)
    qkv_hist = torch.randn(tot_hist, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)
    qkv_dec = torch.randn(n, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    # FP8 framework path
    kv_fp8, bt = _build_fp8_kv_cache(kv_lens, nkh)
    out_fp8 = _run_fp8_framework_decode(
        qkv_hist,
        qkv_dec,
        kv_fp8,
        cs,
        bt,
        history_lens,
        kv_lens,
        nqh,
        nkh,
        qw,
        kw,
    )

    # BF16 operator-level reference
    kv_bf16, _ = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    out_bf16 = _bf16_ref_prefill_then_decode(
        qkv_hist,
        qkv_dec,
        kv_bf16,
        cs,
        bt,
        history_lens,
        kv_lens,
        nqh,
        nkh,
        qw,
        kw,
    )

    assert out_fp8.dtype == torch.bfloat16
    assert not torch.isnan(out_fp8).any(), "Output contains NaN"

    torch.testing.assert_close(out_fp8, out_bf16, atol=0.25, rtol=0.15)


# -----------------------------------------------------------------
# Test: FP8 framework metadata state machine
# -----------------------------------------------------------------


@torch.inference_mode()
def test_fp8_framework_metadata_lifecycle():
    """Verify attn_metadata.hpc_* fields are correctly set and reset.

    Simulates two consecutive layers sharing the same metadata object
    (as in the real framework where all layers in a kv_cache_group
    share one HpcAttnMetadata instance).
    """
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = 8, 1
    q_lens = [64]
    kv_lens = [64]
    tot = sum(q_lens)

    cs = _make_cos_sin_cache(256).cuda()
    qw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
    kw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")

    kv_cache, bt = _build_fp8_kv_cache(kv_lens, nkh)

    qo_indptr = torch.tensor([0] + q_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )

    # ONE shared metadata object (like the real framework)
    shared_metadata = HpcAttnMetadata(
        num_actual_tokens=tot,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=len(q_lens),
        num_prefill_tokens=tot,
        max_query_len=max(q_lens),
        slot_mapping=torch.zeros(tot, dtype=torch.long, device="cuda"),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        block_table_tensor=bt,
        qo_indptr=qo_indptr,
    )

    attn_impl = HpcAttentionImpl(
        num_heads=nqh,
        head_size=HEAD_SIZE,
        scale=HEAD_SIZE**-0.5,
        num_kv_heads=nkh,
        kv_cache_dtype="fp8_e4m3",
    )

    # Simulate 2 layers sharing the same metadata
    for layer_idx in range(2):
        layer_name = f"test_layer_{layer_idx}"
        rope_norm = _create_rope_norm(nqh, nkh, cs, qw, kw, layer_name)

        # Need separate kv_cache per layer (in real framework each layer
        # has its own kv_cache but shares metadata)
        kv_cache_i, bt_i = _build_fp8_kv_cache(kv_lens, nkh)
        mock_layer_i = _MockAttnLayer(kv_cache_i)

        forward_ctx = ForwardContext(
            no_compile_layers={layer_name: mock_layer_i},
            attn_metadata={layer_name: shared_metadata},
            slot_mapping={},
        )

        qkv = torch.randn(tot, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

        with override_forward_context(forward_ctx):
            # Before RopeNorm: metadata should be clean
            assert shared_metadata.hpc_kv_written is False, (
                f"Layer {layer_idx}: metadata should be clean before RopeNorm"
            )

            # RopeNorm sets fields
            q_out = torch.empty(
                tot, nqh, HEAD_SIZE, dtype=torch.float8_e4m3fn, device="cuda"
            )
            rope_norm._forward_impl(
                qkv,
                kv_cache_i,
                shared_metadata,
                mock_layer_i,
                q_out,
            )
            assert shared_metadata.hpc_kv_written is True, (
                f"Layer {layer_idx}: RopeNorm should set hpc_kv_written"
            )

            # HpcAttentionImpl consumes and resets
            dummy_kv = torch.zeros(tot, nkh, HEAD_SIZE, dtype=dtype, device="cuda")
            attn_out = torch.empty(tot, nqh, HEAD_SIZE, dtype=dtype, device="cuda")
            attn_impl.forward(
                layer=mock_layer_i,
                query=q_out,
                key=dummy_kv,
                value=dummy_kv,
                kv_cache=kv_cache_i,
                attn_metadata=shared_metadata,
                output=attn_out,
            )
            assert shared_metadata.hpc_kv_written is False, (
                f"Layer {layer_idx}: HpcAttentionImpl should reset metadata"
            )

    # Clean up global registry
    for i in range(2):
        _hpc_rope_norm_instances.pop(f"test_layer_{i}", None)


# -----------------------------------------------------------------
# Test: FP8 framework prefill vs pure-PyTorch reference
# -----------------------------------------------------------------


@pytest.mark.parametrize("num_heads", FP8_HEADS)
@torch.inference_mode()
def test_fp8_framework_prefill_against_torch_ref(num_heads):
    """FP8 framework-level prefill vs pure-PyTorch reference.

    End-to-end validation: framework FP8 path → vs →
    manual RoPE + RMSNorm + PyTorch attention.
    """
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads
    scale = HEAD_SIZE**-0.5

    q_lens = [64, 128]
    kv_lens = [64, 128]
    tot = sum(q_lens)

    cs = _make_cos_sin_cache(max(kv_lens) + 128).cuda()
    qw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
    kw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")

    qkv = torch.randn(tot, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    # FP8 framework path
    kv_fp8, bt = _build_fp8_kv_cache(kv_lens, nkh)
    out_fp8 = _run_fp8_framework_prefill(
        qkv,
        kv_fp8,
        cs,
        bt,
        q_lens,
        kv_lens,
        nqh,
        nkh,
        qw,
        kw,
    )

    # Pure PyTorch reference
    positions = _positions_for_seqs(q_lens, kv_lens)
    kv_ref, _ = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    q_ref = _ref_rope_norm_write_kv(
        qkv,
        cs,
        positions,
        kv_ref,
        bt,
        kv_lens,
        q_lens,
        nqh,
        nkh,
        True,
        qw,
        kw,
    )
    out_ref = ref_paged_attn(
        q_ref,
        kv_ref[:, 0],
        kv_ref[:, 1],
        q_lens,
        kv_lens,
        bt,
        scale,
    )

    assert out_fp8.dtype == torch.bfloat16
    torch.testing.assert_close(out_fp8, out_ref, atol=0.25, rtol=0.15)
