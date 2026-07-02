# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPC attention backend integration tests.

Covers three paths:
  1. HpcAttentionImpl.forward() — decode / prefill / mixed (BF16)
  2. HpcRopeNorm + attention pipeline (BF16 operator-level)
  3. HpcRopeNorm + HpcAttentionImpl end-to-end (FP8 framework)
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

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.hpc import HpcRopeNorm, QkNormPolicy
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import AttentionBackendEnum

HEAD_SIZE = 128
BLOCK_SIZE = 64
NUM_BLOCKS = 2048

# (num_query_heads, num_kv_heads)
PREFILL_HEADS = [(8, 2), (32, 4), (32, 8)]
DECODE_HEADS = [(4, 1), (8, 1), (32, 4)]
ROPE_NORM_HEADS = [(8, 1), (64, 8)]
FP8_HEADS = [(8, 1)]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def ref_paged_attn(query, key_cache, value_cache, query_lens, kv_lens,
                   block_tables, scale):
    """Pure PyTorch paged attention with causal mask."""
    block_tables_cpu = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start = 0
    for i in range(len(query_lens)):
        q_len, kv_len = query_lens[i], kv_lens[i]
        q = query[start:start + q_len] * scale
        n_blks = (kv_len + block_size - 1) // block_size
        blk_idx = block_tables_cpu[i, :n_blks]
        k = key_cache[blk_idx].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[blk_idx].view(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            ratio = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, ratio, dim=1)
            v = torch.repeat_interleave(v, ratio, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(torch.ones(q_len, kv_len, device=q.device),
                          diagonal=kv_len - q_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))
        start += q_len
    return torch.cat(outputs, dim=0)


def _build_kv_env(kv_lens, num_kv_heads, dtype, *, zero_cache=False):
    max_kv = max(kv_lens)
    max_blks = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_seqs = len(kv_lens)
    fn = torch.zeros if zero_cache else torch.randn
    kv_cache = fn(NUM_BLOCKS, 2, BLOCK_SIZE, num_kv_heads, HEAD_SIZE, dtype=dtype)
    block_tables = torch.zeros(num_seqs, max_blks, dtype=torch.int32, device="cuda")
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


def _slot_mapping(query_lens, kv_lens, block_tables):
    slots = []
    for i, (ql, kl) in enumerate(zip(query_lens, kv_lens)):
        for t in range(ql):
            pos = kl - ql + t
            blk = block_tables[i, pos // BLOCK_SIZE].item()
            slots.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
    return torch.tensor(slots, dtype=torch.long, device="cuda")


def _full_kv_slot_mapping(kv_lens, block_tables):
    slots = []
    for i, kv_len in enumerate(kv_lens):
        for pos in range(kv_len):
            blk = block_tables[i, pos // BLOCK_SIZE].item()
            slots.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
    return torch.tensor(slots, dtype=torch.long, device="cuda")


def _write_kv(key, value, kv_cache, slot_mapping):
    for t in range(key.shape[0]):
        s = slot_mapping[t].item()
        kv_cache[s // BLOCK_SIZE, 0, s % BLOCK_SIZE] = key[t]
        kv_cache[s // BLOCK_SIZE, 1, s % BLOCK_SIZE] = value[t]


class _MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_k_scale", torch.tensor(1.0))
        self.register_buffer("_v_scale", torch.tensor(1.0))


def _make_metadata(query_lens, kv_lens, block_tables, slot_mapping, *,
                   hpc_kv_written=False):
    total = sum(query_lens)
    n_dec = 0
    for ql in query_lens:
        if ql <= 1:
            n_dec += 1
        else:
            break
    n_dec_tok = sum(query_lens[:n_dec])
    n_pf = len(query_lens) - n_dec

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
        num_prefill_tokens=total - n_dec_tok,
        max_query_len=max(query_lens),
        slot_mapping=slot_mapping,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        block_table_tensor=block_tables,
        qo_indptr=qo_indptr,
        hpc_kv_written=hpc_kv_written,
    )


def _run_impl_forward(query, key, value, kv_cache, query_lens, kv_lens,
                      block_tables, slot_mapping, nqh, nkh):
    meta = _make_metadata(query_lens, kv_lens, block_tables, slot_mapping)
    impl = HpcAttentionImpl(num_heads=nqh, head_size=HEAD_SIZE,
                            scale=HEAD_SIZE**-0.5, num_kv_heads=nkh,
                            kv_cache_dtype="auto")
    output = torch.empty_like(query)
    return impl.forward(layer=_MockLayer(), query=query, key=key, value=value,
                        kv_cache=kv_cache, attn_metadata=meta, output=output)


# ---------------------------------------------------------------------
# Part 1: HpcAttentionImpl.forward (BF16, non-RopeNorm path)
# ---------------------------------------------------------------------


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
    sm = _slot_mapping(q_lens, kv_lens, bt)
    kv_sm = _full_kv_slot_mapping(kv_lens, bt)

    query = torch.randn(tot_q, nqh, HEAD_SIZE, dtype=dtype)
    key = torch.randn(tot_kv, nkh, HEAD_SIZE, dtype=dtype)
    value = torch.randn(tot_kv, nkh, HEAD_SIZE, dtype=dtype)
    _write_kv(key, value, kv_cache, kv_sm)

    result = _run_impl_forward(query, key, value, kv_cache, q_lens, kv_lens,
                               bt, sm, nqh, nkh)
    ref = ref_paged_attn(query, kv_cache[:, 0], kv_cache[:, 1], q_lens,
                         kv_lens, bt, HEAD_SIZE**-0.5)
    torch.testing.assert_close(result[:tot_q], ref, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "kv_lens",
    [
        [128],
        [64, 256, 512],
        [1, 37, 128, 2011],  # tail-block edge case
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
    sm = _slot_mapping(q_lens, kv_lens, bt)
    query = torch.randn(n, nqh, HEAD_SIZE, dtype=dtype)
    key = torch.randn(n, nkh, HEAD_SIZE, dtype=dtype)
    value = torch.randn(n, nkh, HEAD_SIZE, dtype=dtype)

    result = _run_impl_forward(query, key, value, kv_cache, q_lens, kv_lens,
                               bt, sm, nqh, nkh)
    ref = ref_paged_attn(query, kv_cache[:, 0], kv_cache[:, 1], q_lens,
                         kv_lens, bt, HEAD_SIZE**-0.5)
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
    sm = _slot_mapping(q_lens, kv_lens, bt)

    pf_kv_total = sum(kv_lens[n_dec:])
    k_pf = torch.randn(pf_kv_total, nkh, HEAD_SIZE, dtype=dtype)
    v_pf = torch.randn(pf_kv_total, nkh, HEAD_SIZE, dtype=dtype)
    pf_sm = _full_kv_slot_mapping(kv_lens[n_dec:], bt[n_dec:])
    _write_kv(k_pf, v_pf, kv_cache, pf_sm)

    query = torch.randn(tot_q, nqh, HEAD_SIZE, dtype=dtype)
    key = torch.randn(tot_q, nkh, HEAD_SIZE, dtype=dtype)
    value = torch.randn(tot_q, nkh, HEAD_SIZE, dtype=dtype)

    result = _run_impl_forward(query, key, value, kv_cache, q_lens, kv_lens,
                               bt, sm, nqh, nkh)
    ref = ref_paged_attn(query, kv_cache[:, 0], kv_cache[:, 1], q_lens,
                         kv_lens, bt, HEAD_SIZE**-0.5)
    torch.testing.assert_close(result[:tot_q], ref, atol=1.5e-2, rtol=1e-2)


@torch.inference_mode()
def test_impl_forward_none_metadata():
    """Output should be zeroed when attn_metadata is None (profiling run)."""
    torch.set_default_device("cuda")
    dtype = torch.bfloat16
    impl = HpcAttentionImpl(num_heads=8, head_size=HEAD_SIZE,
                            scale=HEAD_SIZE**-0.5, num_kv_heads=1,
                            kv_cache_dtype="auto")
    kv_cache = torch.randn(32, 2, BLOCK_SIZE, 1, HEAD_SIZE, dtype=dtype)
    q = torch.randn(4, 8, HEAD_SIZE, dtype=dtype)
    out = torch.ones(4, 8, HEAD_SIZE, dtype=dtype)
    result = impl.forward(layer=_MockLayer(), query=q, key=q[:, :1],
                          value=q[:, :1], kv_cache=kv_cache,
                          attn_metadata=None, output=out)
    assert torch.all(result == 0)


@pytest.mark.parametrize(
    "kwargs, err_match",
    [
        (dict(num_heads=8, head_size=64, scale=0.125, num_kv_heads=1),
         "head_dim=128"),
        (dict(num_heads=6, head_size=128, scale=HEAD_SIZE**-0.5, num_kv_heads=2),
         "head_per_group"),
        (dict(num_heads=8, head_size=128, scale=HEAD_SIZE**-0.5, num_kv_heads=1,
              kv_cache_dtype="int8"),
         "kv_cache_dtype"),
    ],
)
def test_impl_rejects_invalid_config(kwargs, err_match):
    """Impl should reject invalid head_size / GQA ratio / kv_cache_dtype."""
    with pytest.raises(ValueError, match=err_match):
        HpcAttentionImpl(**kwargs)


# ---------------------------------------------------------------------
# Part 2: RopeNorm + attention pipeline (BF16 operator-level)
# ---------------------------------------------------------------------


def _make_cos_sin_cache(max_pos):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_SIZE, 2,
                                                dtype=torch.float) / HEAD_SIZE))
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _ref_rope_neox(x, cos_sin_cache, positions):
    half = HEAD_SIZE // 2
    cs = cos_sin_cache[positions]
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:].unsqueeze(1)
    return torch.cat([x[..., :half] * cos - x[..., half:] * sin,
                      x[..., half:] * cos + x[..., :half] * sin], dim=-1).to(x.dtype)


def _ref_rms_norm(x, weight, eps=1e-6):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * weight).to(x.dtype)


def _ref_rope_norm_write_kv(qkv, cos_sin_cache, positions, kv_cache,
                            block_tables, kv_lens, query_lens, nqh, nkh,
                            use_qk_norm, qw, kw):
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
    pos = []
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
    qw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda") \
        if use_qk_norm else None
    kw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda") \
        if use_qk_norm else None

    qkv = torch.randn(tot, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    kv_hpc, bt = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    cu = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    sl = torch.tensor(kv_lens, dtype=torch.int32)

    q_hpc = hpc.rope_norm_store_kv(
        kv_hpc[:, 0], kv_hpc[:, 1], qkv, cs, sl, cu, bt, True,
        q_norm_weight=qw, k_norm_weight=kw,
        qk_norm_policy=(QkNormPolicy.ROPE_THEN_NORM if use_qk_norm
                        else QkNormPolicy.NONE),
    )
    out_hpc = torch.empty(tot, nqh, HEAD_SIZE, dtype=dtype)
    hpc.attention_with_kvcache_prefill_bf16(
        q_hpc, kv_hpc[:, 0], kv_hpc[:, 1], cu, bt, sl, max(q_lens),
        output=out_hpc,
    )

    kv_ref, _ = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    q_ref = _ref_rope_norm_write_kv(qkv, cs, positions, kv_ref, bt,
                                    kv_lens, q_lens, nqh, nkh,
                                    use_qk_norm, qw, kw)
    out_ref = ref_paged_attn(q_ref, kv_ref[:, 0], kv_ref[:, 1],
                             q_lens, kv_lens, bt, scale)
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
    qw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda") \
        if use_qk_norm else None
    kw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda") \
        if use_qk_norm else None

    qkv = torch.randn(n, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    kv_hpc, bt = _build_kv_env(kv_lens, nkh, dtype)
    kv_ref = kv_hpc.clone()
    sl = torch.tensor(kv_lens, dtype=torch.int32)

    q_hpc = hpc.rope_norm_store_kv(
        kv_hpc[:, 0], kv_hpc[:, 1], qkv, cs, sl,
        torch.arange(sl.numel() + 1, dtype=torch.int32, device=sl.device),
        bt, False,
        q_norm_weight=qw, k_norm_weight=kw,
        qk_norm_policy=(QkNormPolicy.ROPE_THEN_NORM if use_qk_norm
                        else QkNormPolicy.NONE),
    )
    out_hpc = torch.empty(n, nqh, HEAD_SIZE, dtype=dtype)
    hpc.attention_decode_bf16(q_hpc, kv_hpc[:, 0], kv_hpc[:, 1], bt, sl,
                              output=out_hpc, new_kv_included=True, splitk=True)

    q_ref = _ref_rope_norm_write_kv(qkv, cs, positions, kv_ref, bt,
                                    kv_lens, q_lens, nqh, nkh,
                                    use_qk_norm, qw, kw)
    out_ref = ref_paged_attn(q_ref, kv_ref[:, 0], kv_ref[:, 1],
                             q_lens, kv_lens, bt, scale)
    torch.testing.assert_close(out_hpc, out_ref, atol=2e-2, rtol=1e-2)


# ---------------------------------------------------------------------
# Part 3: FP8 framework end-to-end
# (HpcRopeNorm._forward_impl → attn_metadata.hpc_* → HpcAttentionImpl.forward)
# ---------------------------------------------------------------------

_test_vllm_config = VllmConfig()
_test_vllm_config.attention_config.backend = AttentionBackendEnum.HPC_ATTN


class _MockAttnLayer(nn.Module):
    def __init__(self, kv_cache):
        super().__init__()
        self.kv_cache = [kv_cache]
        self.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))


def _build_fp8_kv_cache(kv_lens, num_kv_heads):
    max_kv = max(kv_lens)
    max_blks = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_seqs = len(kv_lens)
    kv_cache = torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, num_kv_heads, HEAD_SIZE,
                           dtype=torch.uint8, device="cuda")
    block_tables = torch.zeros(num_seqs, max_blks, dtype=torch.int32,
                               device="cuda")
    blk = 0
    for i, kv_len in enumerate(kv_lens):
        n = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(n):
            block_tables[i, b] = blk + b
        blk += n
    return kv_cache, block_tables


def _create_rope_norm(nqh, nkh, cos_sin_cache, qw, kw, layer_name):
    qmod = nn.Module()
    qmod.weight = nn.Parameter(qw.clone())
    kmod = nn.Module()
    kmod.weight = nn.Parameter(kw.clone())
    with set_current_vllm_config(_test_vllm_config):
        rope_norm = HpcRopeNorm(
            num_heads=nqh, num_kv_heads=nkh, head_dim=HEAD_SIZE,
            cos_sin_cache=cos_sin_cache, use_qk_norm=True,
            fallback_qnorm=qmod, fallback_knorm=kmod,
            kv_cache_dtype="fp8_e4m3", layer_name=layer_name,
        )
    rope_norm.process_weights_after_loading()
    return rope_norm


def _bf16_ref_prefill(qkv, cos_sin_cache, block_tables, q_lens, kv_lens,
                      nqh, nkh, qw, kw):
    """BF16 operator-level reference for prefill (validated by Part 2)."""
    dtype = torch.bfloat16
    tot = sum(q_lens)
    kv_bf16, _ = _build_kv_env(kv_lens, nkh, dtype, zero_cache=True)
    cu = torch.tensor([0] + q_lens, dtype=torch.int32,
                      device="cuda").cumsum(0, dtype=torch.int32)
    sl = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    q_ref = hpc.rope_norm_store_kv(
        kv_bf16[:, 0], kv_bf16[:, 1], qkv, cos_sin_cache, sl, cu, block_tables,
        True, q_norm_weight=qw, k_norm_weight=kw,
        qk_norm_policy=QkNormPolicy.ROPE_THEN_NORM,
    )
    out = torch.empty(tot, nqh, HEAD_SIZE, dtype=dtype, device="cuda")
    hpc.attention_with_kvcache_prefill_bf16(
        q_ref, kv_bf16[:, 0], kv_bf16[:, 1], cu, block_tables, sl,
        max(q_lens), output=out,
    )
    return out


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
    """End-to-end FP8 pipeline: HpcRopeNorm → HpcAttentionImpl → bf16 output."""
    torch.set_default_device("cuda")
    set_random_seed(0)
    dtype = torch.bfloat16
    nqh, nkh = num_heads
    q_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    tot = sum(q_lens)
    layer_name = "test_fp8_prefill_layer"

    cs = _make_cos_sin_cache(max(kv_lens) + 128).cuda()
    qw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
    kw = torch.randn(HEAD_SIZE, dtype=torch.float32, device="cuda")
    qkv = torch.randn(tot, nqh * HEAD_SIZE + 2 * nkh * HEAD_SIZE, dtype=dtype)

    kv_fp8, bt = _build_fp8_kv_cache(kv_lens, nkh)
    qo_indptr = torch.tensor([0] + q_lens, dtype=torch.int32,
                             device="cuda").cumsum(0, dtype=torch.int32)
    metadata = HpcAttnMetadata(
        num_actual_tokens=tot, num_decodes=0, num_decode_tokens=0,
        num_prefills=len(q_lens), num_prefill_tokens=tot,
        max_query_len=max(q_lens),
        slot_mapping=torch.zeros(tot, dtype=torch.long, device="cuda"),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        block_table_tensor=bt, qo_indptr=qo_indptr,
        hpc_kv_written=True,
    )
    mock_layer = _MockAttnLayer(kv_fp8)
    rope_norm = _create_rope_norm(nqh, nkh, cs, qw, kw, layer_name)
    attn_impl = HpcAttentionImpl(num_heads=nqh, head_size=HEAD_SIZE,
                                 scale=HEAD_SIZE**-0.5, num_kv_heads=nkh,
                                 kv_cache_dtype="fp8_e4m3")
    forward_ctx = ForwardContext(
        no_compile_layers={layer_name: mock_layer},
        attn_metadata={layer_name: metadata},
        slot_mapping={},
    )
    with override_forward_context(forward_ctx):
        q_output = torch.empty(tot, nqh, HEAD_SIZE, dtype=torch.float8_e4m3fn,
                               device="cuda")
        rope_norm._forward_impl(qkv, kv_fp8, metadata, mock_layer, q_output)
        dummy_kv = torch.zeros(tot, nkh, HEAD_SIZE, dtype=dtype, device="cuda")
        out_fp8 = torch.empty(tot, nqh, HEAD_SIZE, dtype=dtype, device="cuda")
        attn_impl.forward(layer=mock_layer, query=q_output, key=dummy_kv,
                          value=dummy_kv, kv_cache=kv_fp8,
                          attn_metadata=metadata, output=out_fp8)

    out_bf16 = _bf16_ref_prefill(qkv, cs, bt, q_lens, kv_lens, nqh, nkh, qw, kw)

    assert out_fp8.dtype == torch.bfloat16
    assert not torch.isnan(out_fp8).any()
    torch.testing.assert_close(out_fp8, out_bf16, atol=0.25, rtol=0.15)
