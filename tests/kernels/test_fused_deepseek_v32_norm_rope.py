# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the horizontally-fused deepseek_v32 (NVIDIA SM100) Triton
kernels used by the specialized DSA model:

  fused_norm_rope
    - q  : q_lora RMSNorm
    - kv : kv_lora RMSNorm + (interleaved) RoPE on k_pe + MLA cache insert
           (bf16 or per-tensor fp8)
    - idx: indexer-K LayerNorm + RoPE (interleaved or NeoX) + UE8M0 fp8 quant +
           packed indexer cache insert; plus the top-k buffer (-1) fill
  fused_q
    - mqa: ql_nope + (interleaved) RoPE'd q_pe, concat-quantized to the fp8 MQA
           query
    - idx: indexer-Q RoPE (interleaved or NeoX) + UE8M0 fp8 quant + folded
           index weights
  fused_eh_norm (MTP): zero-at-pos-0 + enorm RMSNorm(embeds) + hnorm
           RMSNorm(prev), concatenated side-by-side

Each kernel is compared against a PyTorch reference. The kernel keeps the whole
pipeline in fp32 and rounds once, so it can land on the opposite side of a
round-to-nearest tie from the reference for a few elements: deterministic fp8
outputs are checked within 1 representable-step (ULP); bf16 norm/RoPE outputs use
rtol/atol=1e-2 (the tolerance the sibling deepseek_v4 fused-kernel test uses).
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.cuda_vmm import (
    create_rank_major_peer_view,
)
from vllm.model_executor.layers.attention.pcp_peer_cache import (
    PCPPeerCacheFence,
    make_rank_major_tensor_view,
)
from vllm.models.deepseek_v32.nvidia import kernels as K
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

# GLM-5.2 / DeepSeek-V3.2 shapes (TP8 local heads).
Q_LORA = 2048
KV_LORA = 512
ROPE_DIM = 64
NUM_HEADS = 8
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
HIDDEN = 6144
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="deepseek_v32 fused kernels require CUDA with fp8 (SM89+)",
)


def test_platform_capability_queries_are_constant_during_compile(monkeypatch):
    monkeypatch.setattr(
        K.current_platform, "has_device_capability", lambda capability: True
    )
    monkeypatch.setattr(K, "has_cutedsl", lambda: True)
    monkeypatch.setattr(K.current_platform, "is_arch_support_pdl", lambda: True)

    def capability_branches(x: torch.Tensor) -> torch.Tensor:
        if K._can_use_fused_q_cutedsl() and K._is_arch_support_pdl():
            return x + 1
        return x - 1

    compiled = torch.compile(capability_branches, backend="eager", fullgraph=True)
    x = torch.zeros(1, device="cuda")
    torch.testing.assert_close(compiled(x), torch.ones_like(x))


def test_pdl_is_disabled_before_blackwell(monkeypatch):
    monkeypatch.setattr(
        K.current_platform, "has_device_capability", lambda capability: False
    )
    monkeypatch.setattr(K.current_platform, "is_arch_support_pdl", lambda: True)

    assert not K._is_arch_support_pdl()


# ── reference helpers ────────────────────────────────────────────────────────


def make_cos_sin(max_pos: int, rot_dim: int, device) -> torch.Tensor:
    """cos||sin cache: row[pos] = [cos(theta)(rot/2), sin(theta)(rot/2)]."""
    half = rot_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMSNorm matching kernels._rms_norm (fp32, eps inside rsqrt). Returns fp32."""
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(ms + EPS) * w.float()


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    return (xf - mean) * torch.rsqrt(var + EPS) * w.float() + b.float()


def rope(
    x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, interleave: bool
) -> torch.Tensor:
    """Apply RoPE to the first ``rot_dim`` elements of x's last dim.

    x: [..., head_dim] fp32. ``cos_sin`` is [max_pos, rot_dim]. ``interleave``
    selects adjacent-pair (GLM) vs split-half NeoX (DeepSeek-V3.2) layout.
    """
    rot = cos_sin.shape[-1]
    half = rot // 2
    cs = cos_sin[pos.long()]
    cos, sin = cs[..., :half], cs[..., half:]
    out = x.float().clone()
    r = out[..., :rot]
    if interleave:
        x1, x2 = r[..., 0::2].clone(), r[..., 1::2].clone()
        r[..., 0::2] = x1 * cos - x2 * sin
        r[..., 1::2] = x2 * cos + x1 * sin
    else:
        x1, x2 = r[..., :half].clone(), r[..., half:].clone()
        r[..., :half] = x1 * cos - x2 * sin
        r[..., half:] = x2 * cos + x1 * sin
    return out


def ue8m0_quant(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (last dim) UE8M0 fp8 quant matching kernels._fp8_ue8m0_quantize."""
    amax = vals.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-4) / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (vals.float() / scale).to(FP8)
    return q, scale.squeeze(-1)


def _bf16_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        return torch.where(u >= 0x8000, 0xFFFF - u, u + 0x8000)

    return int((key(a) - key(b)).abs().max().item())


def _fp8_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return int((key(a) - key(b)).abs().max().item())


def assert_bf16(got: torch.Tensor, ref_fp32: torch.Tensor, msg: str):
    # Kernel keeps RMSNorm/RoPE in fp32 and rounds to bf16 once; the fp32
    # reduction/FMA order differs from torch, so a few elements land on the
    # opposite side of a round-to-nearest tie. Use the same tolerance the
    # sibling deepseek_v4 fused-kernel test uses for this bf16 norm+rope class.
    torch.testing.assert_close(
        got.float(), ref_fp32.float(), rtol=1e-2, atol=1e-2, msg=lambda m: f"{msg}: {m}"
    )


def assert_fp8(got: torch.Tensor, ref: torch.Tensor, msg: str):
    assert _fp8_ulp(got, ref) <= 1, f"{msg}: >1 fp8 ULP"


# ── fused_norm_rope ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512, 4096])
@pytest.mark.parametrize("index_interleave", [True, False])
@pytest.mark.parametrize("mla_fp8", [False, True])
def test_fused_norm_rope(num_tokens: int, index_interleave: bool, mla_fp8: bool):
    torch.manual_seed(0)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_c = torch.randn(num_tokens, Q_LORA, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, ROPE_DIM, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(Q_LORA, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(KV_LORA, device=dev, dtype=torch.bfloat16)
    ik = torch.randn(num_tokens, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
    ikw = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    ikb = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)

    mla_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)  # MLA k_pe: interleaved
    idx_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    bs = max_pos  # single block covering all tokens
    mla_dim = KV_LORA + ROPE_DIM
    if mla_fp8:
        mla_cache = torch.zeros(1, bs, mla_dim, device=dev, dtype=torch.uint8)
        mla_dtype = "fp8"
        mla_k_scale = torch.tensor([0.3], device=dev, dtype=torch.float32)
    else:
        mla_cache = torch.zeros(1, bs, mla_dim, device=dev, dtype=torch.bfloat16)
        mla_dtype = "auto"
        mla_k_scale = None
    idx_row = INDEX_HEAD_DIM + INDEX_HEAD_DIM // 128 * 4  # 132
    idx_cache = torch.zeros(1, bs, idx_row, device=dev, dtype=torch.uint8)
    slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    topk = torch.full((num_tokens, 2048), 7, device=dev, dtype=torch.int32)

    q_out = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        mla_cos_sin,
        ik,
        ikw,
        ikb,
        EPS,
        idx_cos_sin,
        topk,
        slot_mapping=slot,
        indexer_k_cache=idx_cache,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype=mla_dtype,
        mla_k_scale=mla_k_scale,
        has_indexer=True,
        index_rope_interleave=index_interleave,
    )

    # q_lora RMSNorm
    assert_bf16(q_out, rms_norm(q_c, qw), "q_c rmsnorm")

    # MLA cache: [kv_c_normed | k_pe_roped(interleaved)]
    kv_ref = rms_norm(kv_c, kvw)
    kpe_ref = rope(k_pe.float(), pos, mla_cos_sin, interleave=True)
    if mla_fp8:
        cache = mla_cache.view(FP8)[0, :num_tokens]
        s = mla_k_scale.item()
        assert_fp8(cache[:, :KV_LORA], (kv_ref / s).to(FP8), "MLA kv fp8")
        assert_fp8(cache[:, KV_LORA:], (kpe_ref / s).to(FP8), "MLA k_pe fp8")
    else:
        cache = mla_cache[0, :num_tokens]
        assert_bf16(cache[:, :KV_LORA], kv_ref, "MLA kv bf16")
        assert_bf16(cache[:, KV_LORA:], kpe_ref, "MLA k_pe bf16")

    # Indexer-K cache (packed [bs*head_dim fp8 | bs*4 fp32 scale]).
    ik_ref = layer_norm(ik, ikw, ikb)
    ik_ref = rope(ik_ref, pos, idx_cos_sin, interleave=index_interleave)
    q_ref, s_ref = ue8m0_quant(ik_ref)
    flat = idx_cache[0].reshape(-1)
    vals = flat[: bs * INDEX_HEAD_DIM].view(FP8).reshape(bs, INDEX_HEAD_DIM)
    scales = flat[bs * INDEX_HEAD_DIM :].view(torch.float32)
    assert_fp8(vals[:num_tokens], q_ref, "indexer-K fp8")
    torch.testing.assert_close(scales[:num_tokens], s_ref, rtol=0, atol=0)

    # Top-k buffer cleared to -1 on indexer layers.
    assert (topk == -1).all(), "topk buffer not cleared on indexer layer"


@pytest.mark.parametrize("num_tokens", [1, 17, 512])
def test_fused_norm_rope_no_indexer(num_tokens: int):
    """Shared (no-indexer) layer: q + kv/MLA only; top-k buffer untouched."""
    torch.manual_seed(1)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)

    q_c = torch.randn(num_tokens, Q_LORA, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, ROPE_DIM, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(Q_LORA, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(KV_LORA, device=dev, dtype=torch.bfloat16)
    mla_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    bs = max_pos
    mla_cache = torch.zeros(1, bs, KV_LORA + ROPE_DIM, device=dev, dtype=torch.bfloat16)
    slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    topk = torch.full((num_tokens, 2048), 7, device=dev, dtype=torch.int32)

    q_out = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        mla_cos_sin,
        None,
        None,
        None,
        EPS,
        None,
        topk,
        slot_mapping=slot,
        indexer_k_cache=None,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype="auto",
        mla_k_scale=None,
        has_indexer=False,
        index_rope_interleave=False,
    )

    assert_bf16(q_out, rms_norm(q_c, qw), "q_c rmsnorm (no-indexer)")
    cache = mla_cache[0, :num_tokens]
    assert_bf16(cache[:, :KV_LORA], rms_norm(kv_c, kvw), "MLA kv (no-indexer)")
    assert_bf16(
        cache[:, KV_LORA:],
        rope(k_pe.float(), pos, mla_cos_sin, interleave=True),
        "MLA k_pe (no-indexer)",
    )
    # Shared layers reuse the previous indexer's top-k: buffer must be untouched.
    assert (topk == 7).all(), "topk buffer should be untouched on shared layer"


@pytest.mark.parametrize("index_interleave", [True, False])
def test_fused_norm_rope_materializes_collective_cache_inputs(
    index_interleave: bool,
):
    """The PCP fallback prepares BF16 inputs without doing local cache writes."""
    torch.manual_seed(11)
    dev = "cuda"
    num_tokens = 5
    max_pos = 32
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)

    q_c = torch.randn(num_tokens, Q_LORA, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, ROPE_DIM, device=dev, dtype=torch.bfloat16)
    index_k = torch.randn(num_tokens, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
    q_input = q_c.clone()
    kv_input = kv_c.clone()
    kpe_input = k_pe.clone()
    index_input = index_k.clone()
    qw = torch.randn(Q_LORA, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(KV_LORA, device=dev, dtype=torch.bfloat16)
    index_w = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    index_b = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)
    topk = torch.full((num_tokens, 16), 7, device=dev, dtype=torch.int32)

    q_out = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        cos_sin,
        index_k,
        index_w,
        index_b,
        EPS,
        cos_sin,
        topk,
        slot_mapping=None,
        indexer_k_cache=None,
        mla_kv_cache=None,
        has_indexer=True,
        index_rope_interleave=index_interleave,
        materialize_cache_inputs=True,
    )

    assert_bf16(q_out, rms_norm(q_input, qw), "materialized q_c rmsnorm")
    assert_bf16(kv_c, rms_norm(kv_input, kvw), "materialized kv_c rmsnorm")
    assert_bf16(
        k_pe,
        rope(kpe_input.float(), pos, cos_sin, interleave=True),
        "materialized MLA k_pe",
    )
    index_ref = layer_norm(index_input, index_w, index_b)
    index_ref = rope(index_ref, pos, cos_sin, interleave=index_interleave)
    assert_bf16(index_k, index_ref, "materialized indexer k")
    assert (topk == -1).all(), "topk buffer not cleared on indexer layer"


@pytest.mark.parametrize("mla_dtype", ["fp8", "fp8_ds_mla"])
def test_fused_norm_rope_direct_pcp_fanout_uses_local_rank_slots(mla_dtype: str):
    """Direct PCP stores update every peer cache without gathering inputs."""
    torch.manual_seed(7)
    dev = "cuda"
    num_tokens = 4
    pcp_size = 2
    pcp_rank = 1
    max_pos = 32
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    q_c = torch.randn(num_tokens, Q_LORA, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, ROPE_DIM, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(Q_LORA, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(KV_LORA, device=dev, dtype=torch.bfloat16)
    index_k = torch.randn(num_tokens, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
    index_w = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    index_b = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    mla_dim = 656 if mla_dtype == "fp8_ds_mla" else KV_LORA + ROPE_DIM
    peer_mla = torch.zeros(pcp_size, 1, max_pos, mla_dim, device=dev, dtype=torch.uint8)
    index_row = INDEX_HEAD_DIM + 4
    peer_index = torch.zeros(
        pcp_size, 1, max_pos, index_row, device=dev, dtype=torch.uint8
    )
    # Rank 0 owns slots 0..3. This producer rank has two padding/masked rows
    # around slots 8..9, as happens for uneven and replicated-decode PCP rows.
    slot_mapping = torch.cat(
        (
            torch.arange(num_tokens, device=dev, dtype=torch.int64),
            torch.tensor([-1, 8, 9, -1], device=dev, dtype=torch.int64),
        )
    )
    topk = torch.full((num_tokens, 16), 7, device=dev, dtype=torch.int32)

    q_out = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        cos_sin,
        index_k,
        index_w,
        index_b,
        EPS,
        cos_sin,
        topk,
        slot_mapping=slot_mapping,
        indexer_k_cache=peer_index[pcp_rank],
        mla_kv_cache=peer_mla[pcp_rank],
        pcp_peer_indexer_k_cache=peer_index,
        pcp_peer_mla_kv_cache=peer_mla,
        pcp_rank=pcp_rank,
        pcp_size=pcp_size,
        mla_kv_cache_dtype=mla_dtype,
        mla_k_scale=(
            None if mla_dtype == "fp8_ds_mla" else torch.tensor([0.5], device=dev)
        ),
        has_indexer=True,
        index_rope_interleave=True,
    )

    torch.testing.assert_close(peer_mla[0], peer_mla[1])
    torch.testing.assert_close(peer_index[0], peer_index[1])
    assert_bf16(q_out, rms_norm(q_c, qw), "direct PCP q_c rmsnorm")
    assert not peer_mla[:, :, :num_tokens].any()
    assert peer_mla[:, :, 8:10].any()
    assert not peer_mla[:, :, 10:12].any()
    index_values = peer_index.view(pcp_size, -1)
    assert not index_values[:, : num_tokens * INDEX_HEAD_DIM].any()
    assert index_values[:, 8 * INDEX_HEAD_DIM : 10 * INDEX_HEAD_DIM].any()
    assert not index_values[:, 10 * INDEX_HEAD_DIM : 12 * INDEX_HEAD_DIM].any()


def _fused_norm_rope_replicated_vmm_worker(
    rank: int, world_size: int, port: int
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    mla_allocation = None
    index_allocation = None
    fence = None
    actual_mla = None
    actual_index = None
    expected_mla = None
    expected_index = None
    actual_guards = None
    actual_block_gaps = None
    direct_q_valid = None
    reference_q_valid = None
    expected_q_valid = None
    try:
        dev = torch.device(f"cuda:{rank}")
        # Each producer sees one replicated decode row followed by its four
        # DualChunkSwap prefill rows.  Only rank 0 owns the decode cache write;
        # the other ranks must still compute a valid Q for that PAD slot.
        num_tokens = 5
        num_blocks = 6
        block_size = 8
        mla_dim = KV_LORA + ROPE_DIM
        index_dim = INDEX_HEAD_DIM + 4

        # Production gives each rank a complete local cache and maps those
        # allocations rank-major. Construct real strided cache views over a
        # larger backing allocation: consecutive physical blocks are separated
        # by gaps which must not be mistaken for cache storage. Nonzero outer
        # offsets additionally validate mirroring a subview of the allocation.
        guard_bytes = 128
        mla_block_bytes = block_size * mla_dim
        index_block_bytes = block_size * index_dim
        mla_gap_bytes = 64
        index_gap_bytes = 32
        mla_block_stride = mla_block_bytes + mla_gap_bytes
        index_block_stride = index_block_bytes + index_gap_bytes
        mla_backing_bytes = (num_blocks - 1) * mla_block_stride + mla_block_bytes
        index_backing_bytes = (num_blocks - 1) * index_block_stride + index_block_bytes
        mla_allocation = create_rank_major_peer_view(
            (guard_bytes + mla_backing_bytes + guard_bytes,),
            dtype=torch.uint8,
            group=dist.group.WORLD,
            require_native_atomics=True,
            device=dev,
        )
        index_allocation = create_rank_major_peer_view(
            (guard_bytes + index_backing_bytes + guard_bytes,),
            dtype=torch.uint8,
            group=dist.group.WORLD,
            require_native_atomics=True,
            device=dev,
        )
        mla_allocation.local_view.zero_()
        index_allocation.local_view.zero_()
        local_mla_backing = mla_allocation.local_view.narrow(
            0, guard_bytes, mla_backing_bytes
        )
        local_index_backing = index_allocation.local_view.narrow(
            0, guard_bytes, index_backing_bytes
        )
        local_mla = torch.as_strided(
            local_mla_backing,
            size=(num_blocks, block_size, mla_dim),
            stride=(mla_block_stride, mla_dim, 1),
        )
        local_index = torch.as_strided(
            local_index_backing,
            size=(num_blocks, block_size, index_dim),
            stride=(index_block_stride, index_dim, 1),
        )
        assert local_mla.stride(0) == mla_block_stride > mla_block_bytes
        assert local_index.stride(0) == index_block_stride > index_block_bytes
        peer_mla = make_rank_major_tensor_view(mla_allocation, local_mla)
        peer_index = make_rank_major_tensor_view(index_allocation, local_index)
        assert peer_mla.shape == (world_size, *local_mla.shape)
        assert peer_index.shape == (world_size, *local_index.shape)
        fence = PCPPeerCacheFence(dist.group.WORLD, dev)

        # This is the exact DualChunkSwap partition for an uneven 13-token
        # prefill at PCP=4 (eight chunks of size two), prefixed by one decode
        # token replicated onto every rank. Padding gathers token 0 but carries
        # slot -1. Rank 0 exclusively owns the replicated decode cache write.
        #
        # Valid slots deliberately span six physical blocks and use varied
        # in-block offsets. This catches kernels that accidentally treat the
        # packed caches as dense rows or ignore their physical block stride.
        gather_rows = torch.tensor(
            [
                [13, 0, 1, 0, 0],
                [13, 2, 3, 12, 0],
                [13, 4, 5, 10, 11],
                [13, 6, 7, 8, 9],
            ],
            dtype=torch.int64,
            device=dev,
        )
        slot_rows = torch.tensor(
            [
                [37, 19, 26, -1, -1],
                [-1, 35, 4, 44, -1],
                [-1, 11, 21, 30, 47],
                [-1, 1, 15, 24, 40],
            ],
            dtype=torch.int64,
            device=dev,
        )
        valid_slots = slot_rows[slot_rows >= 0]
        assert valid_slots.unique().numel() == valid_slots.numel()
        assert slot_rows[0, 0] >= 0
        assert (slot_rows[1:, 0] == -1).all()
        assert (valid_slots // block_size).unique().numel() == num_blocks
        assert (valid_slots % block_size).unique().numel() > 4

        torch.manual_seed(2026)
        global_tokens = 14
        positions_global = torch.arange(global_tokens, device=dev, dtype=torch.int64)
        positions_global[-1] = 31
        q_global = torch.randn(global_tokens, Q_LORA, device=dev, dtype=torch.bfloat16)
        kv_global = torch.randn(
            global_tokens, KV_LORA, device=dev, dtype=torch.bfloat16
        )
        kpe_global = torch.randn(
            global_tokens, ROPE_DIM, device=dev, dtype=torch.bfloat16
        )
        index_global = torch.randn(
            global_tokens, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16
        )
        q_weight = torch.randn(Q_LORA, device=dev, dtype=torch.bfloat16)
        kv_weight = torch.randn(KV_LORA, device=dev, dtype=torch.bfloat16)
        index_weight = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
        index_bias = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
        cos_sin = make_cos_sin(32, ROPE_DIM, dev)
        local_gather = gather_rows[rank]
        positions = positions_global[local_gather]
        q_c = q_global[local_gather]
        kv_c = kv_global[local_gather]
        k_pe = kpe_global[local_gather]
        index_k = index_global[local_gather]
        mla_k_scale = torch.tensor([0.5], device=dev, dtype=torch.float32)

        direct_topk = torch.full((num_tokens, 16), 7, device=dev, dtype=torch.int32)
        direct_q = K.fused_norm_rope(
            positions,
            q_c,
            q_weight,
            EPS,
            kv_c,
            kv_weight,
            EPS,
            k_pe,
            cos_sin,
            index_k,
            index_weight,
            index_bias,
            EPS,
            cos_sin,
            direct_topk,
            slot_mapping=slot_rows.flatten(),
            indexer_k_cache=local_index,
            mla_kv_cache=local_mla,
            pcp_peer_indexer_k_cache=peer_index,
            pcp_peer_mla_kv_cache=peer_mla,
            pcp_rank=rank,
            pcp_size=world_size,
            mla_kv_cache_dtype="fp8",
            mla_k_scale=mla_k_scale,
            has_indexer=True,
            index_rope_interleave=True,
        )
        fence()

        # The ordinary single-rank path is the byte-level oracle for this
        # producer. Combining the disjoint local writes across ranks yields the
        # exact full replica that every direct-store cache must observe.
        reference_mla_backing = torch.zeros(
            mla_backing_bytes, device=dev, dtype=torch.uint8
        )
        reference_index_backing = torch.zeros(
            index_backing_bytes, device=dev, dtype=torch.uint8
        )
        reference_mla = torch.as_strided(
            reference_mla_backing,
            size=local_mla.shape,
            stride=local_mla.stride(),
        )
        reference_index = torch.as_strided(
            reference_index_backing,
            size=local_index.shape,
            stride=local_index.stride(),
        )
        reference_topk = torch.full_like(direct_topk, 7)
        reference_q = K.fused_norm_rope(
            positions,
            q_c,
            q_weight,
            EPS,
            kv_c,
            kv_weight,
            EPS,
            k_pe,
            cos_sin,
            index_k,
            index_weight,
            index_bias,
            EPS,
            cos_sin,
            reference_topk,
            slot_mapping=slot_rows[rank],
            indexer_k_cache=reference_index,
            mla_kv_cache=reference_mla,
            mla_kv_cache_dtype="fp8",
            mla_k_scale=mla_k_scale,
            has_indexer=True,
            index_rope_interleave=True,
        )

        local_reference = (reference_mla.cpu(), reference_index.cpu())
        gathered_references = [None] * world_size
        dist.all_gather_object(
            gathered_references, local_reference, group=dist.group.WORLD
        )
        expected_mla = torch.zeros_like(local_reference[0])
        expected_index = torch.zeros_like(local_reference[1])
        for reference in gathered_references:
            assert reference is not None
            reference_mla, reference_index = reference
            expected_mla.bitwise_or_(reference_mla)
            expected_index.bitwise_or_(reference_index)

        actual_mla = local_mla.cpu()
        actual_index = local_index.cpu()
        actual_guards = torch.cat(
            (
                mla_allocation.local_view[:guard_bytes],
                mla_allocation.local_view[guard_bytes + mla_backing_bytes :],
                index_allocation.local_view[:guard_bytes],
                index_allocation.local_view[guard_bytes + index_backing_bytes :],
            )
        ).cpu()
        actual_block_gaps = torch.cat(
            [
                local_mla_backing.narrow(
                    0, block * mla_block_stride + mla_block_bytes, mla_gap_bytes
                )
                for block in range(num_blocks - 1)
            ]
            + [
                local_index_backing.narrow(
                    0,
                    block * index_block_stride + index_block_bytes,
                    index_gap_bytes,
                )
                for block in range(num_blocks - 1)
            ]
        ).cpu()
        direct_q_valid = direct_q.cpu()
        reference_q_valid = reference_q.cpu()
        # Q production is independent of cache insertion. In particular, all
        # replicated-decode copies and uneven-prefill PAD rows must retain Q.
        # Save the evidence now and assert only after peer mappings are closed.
        expected_q_valid = rms_norm(q_c, q_weight).cpu()

        # Do not assert while peer mappings are live: all ranks first copy the
        # evidence, quiesce, and close in reverse allocation order.
        dist.barrier()
        fence.close()
        fence = None
        index_allocation.close()
        index_allocation = None
        mla_allocation.close()
        mla_allocation = None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    assert actual_mla is not None and expected_mla is not None
    assert actual_index is not None and expected_index is not None
    assert actual_guards is not None
    assert actual_block_gaps is not None
    assert direct_q_valid is not None and reference_q_valid is not None
    assert expected_q_valid is not None
    assert torch.equal(actual_mla, expected_mla), (
        int((actual_mla != expected_mla).sum()),
        int(actual_mla.count_nonzero()),
        int(expected_mla.count_nonzero()),
        actual_mla.count_nonzero(dim=(1, 2)).nonzero().flatten().tolist(),
        expected_mla.count_nonzero(dim=(1, 2)).nonzero().flatten().tolist(),
        actual_mla.flatten()[(actual_mla != expected_mla).flatten()][:16].tolist(),
        expected_mla.flatten()[(actual_mla != expected_mla).flatten()][:16].tolist(),
    )
    assert torch.equal(actual_index, expected_index), (
        int((actual_index != expected_index).sum()),
        int(actual_index.count_nonzero()),
        int(expected_index.count_nonzero()),
    )
    assert not actual_guards.any()
    assert not actual_block_gaps.any()
    assert torch.equal(direct_q_valid, reference_q_valid)
    assert_bf16(direct_q_valid, expected_q_valid, "direct PCP packed/mixed Q")


def test_fused_norm_rope_replicated_vmm_pcp4() -> None:
    world_size = 4
    if torch.cuda.device_count() < world_size or not all(
        source == destination or torch.cuda.can_device_access_peer(source, destination)
        for source in range(world_size)
        for destination in range(world_size)
    ):
        pytest.skip("replicated PCP fanout requires four peer-accessible CUDA GPUs")
    mp.spawn(
        _fused_norm_rope_replicated_vmm_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )


def test_fused_norm_rope_profile_without_cache_compiles():
    """The cache-free profiling path must still compile and produce Q."""
    torch.manual_seed(2)
    dev = "cuda"
    num_tokens = 4
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    q_c = torch.randn(num_tokens, Q_LORA, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, ROPE_DIM, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(Q_LORA, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(KV_LORA, device=dev, dtype=torch.bfloat16)
    index_k = torch.randn(num_tokens, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
    index_w = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    index_b = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    cos_sin = make_cos_sin(64, ROPE_DIM, dev)
    topk = torch.empty(num_tokens, 2048, device=dev, dtype=torch.int32)

    def profile_run(
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor,
    ) -> torch.Tensor:
        return K.fused_norm_rope(
            pos,
            q_c,
            qw,
            EPS,
            kv_c,
            kvw,
            EPS,
            k_pe,
            cos_sin,
            index_k,
            index_w,
            index_b,
            EPS,
            cos_sin,
            topk,
            slot_mapping=None,
            indexer_k_cache=None,
            mla_kv_cache=None,
            has_indexer=True,
            index_rope_interleave=False,
        )

    compiled = torch.compile(profile_run, fullgraph=True)
    q_out = compiled(q_c, kv_c, k_pe, index_k)
    assert_bf16(q_out, rms_norm(q_c, qw), "q_c rmsnorm (profiling)")


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512])
def test_fused_norm_rope_ds_mla(num_tokens: int):
    """fp8_ds_mla MLA cache layout (FlashMLA sparse, bf16-query path; SM90/SM100).

    Per-token 656-byte entry: 512 fp8 NoPE (4 per-128 tiles, dynamic float32
    scale) | 4 float32 scales | 64 bf16 (unquantized) RoPE.
    """
    torch.manual_seed(5)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_c = torch.randn(num_tokens, Q_LORA, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, ROPE_DIM, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(Q_LORA, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(KV_LORA, device=dev, dtype=torch.bfloat16)
    mla_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    bs = max_pos
    mla_cache = torch.zeros(1, bs, 656, device=dev, dtype=torch.uint8)
    slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    topk = torch.full((num_tokens, 2048), 7, device=dev, dtype=torch.int32)

    q_out = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        mla_cos_sin,
        None,
        None,
        None,
        EPS,
        None,
        topk,
        slot_mapping=slot,
        indexer_k_cache=None,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype="fp8_ds_mla",
        mla_k_scale=None,
        has_indexer=False,
        index_rope_interleave=False,
    )

    assert_bf16(q_out, rms_norm(q_c, qw), "q_c rmsnorm (ds_mla)")

    kv_ref = rms_norm(kv_c, kvw)  # [N, 512] fp32
    kpe_ref = rope(k_pe.float(), pos, mla_cos_sin, interleave=True)  # [N, 64]
    tiles = kv_ref.view(num_tokens, 4, 128)
    ref_scale = torch.clamp(tiles.abs().amax(dim=-1) / FP8_MAX, min=1.1754944e-38)
    ref_nope = (tiles / ref_scale[..., None]).reshape(num_tokens, KV_LORA).to(FP8)

    cache = mla_cache[0, :num_tokens]  # [N, 656] uint8
    nope = cache[:, :KV_LORA].view(FP8)
    scales = cache.view(torch.float32)[:, KV_LORA // 4 : KV_LORA // 4 + 4]
    rope_off = KV_LORA // 2 + 8
    rope_vals = cache.view(torch.bfloat16)[:, rope_off : rope_off + ROPE_DIM]

    torch.testing.assert_close(scales, ref_scale, rtol=1e-2, atol=1e-6)
    assert_fp8(nope, ref_nope, "ds_mla NoPE fp8")
    assert_bf16(rope_vals, kpe_ref, "ds_mla RoPE bf16")
    # No indexer on this call: top-k buffer must be untouched.
    assert (topk == 7).all(), "topk buffer should be untouched (no indexer)"


# ── fused_q ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512, 4096])
@pytest.mark.parametrize("index_interleave", [True, False])
def test_fused_q(num_tokens: int, index_interleave: bool):
    torch.manual_seed(2)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_pe = torch.randn(
        num_tokens, NUM_HEADS, ROPE_DIM, device=dev, dtype=torch.bfloat16
    )
    ql_nope = torch.randn(
        num_tokens, NUM_HEADS, KV_LORA, device=dev, dtype=torch.bfloat16
    )
    index_q = torch.randn(
        num_tokens, INDEX_HEADS, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16
    )
    index_w = torch.randn(num_tokens, INDEX_HEADS, device=dev, dtype=torch.float32)
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32)
    softmax_scale = INDEX_HEAD_DIM**-0.5
    head_scale = INDEX_HEADS**-0.5
    q_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)  # q_pe: interleaved
    idx_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    iq_fp8, iw_out, mqa = K.fused_q(
        pos,
        q_pe,
        q_cos_sin,
        index_q,
        idx_cos_sin,
        ql_nope,
        q_scale,
        index_w,
        softmax_scale,
        head_scale,
        has_indexer=True,
        index_rope_interleave=index_interleave,
    )

    s = q_scale.item()
    # MQA query: [ql_nope | q_pe RoPE'd (interleaved)], per-tensor fp8.
    mqa_nope_ref = (ql_nope.float() / s).to(FP8)
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, NUM_HEADS),
        q_cos_sin,
        interleave=True,
    )
    mqa_pe_ref = (qpe_ref / s).to(FP8)
    assert_fp8(mqa[:, :, :KV_LORA], mqa_nope_ref, "mqa ql_nope")
    assert_fp8(mqa[:, :, KV_LORA:], mqa_pe_ref, "mqa q_pe")

    # Indexer-Q: RoPE + UE8M0 fp8 quant; index weights fold in q-scale.
    iq_ref = rope(
        index_q.float(),
        pos.unsqueeze(-1).expand(num_tokens, INDEX_HEADS),
        idx_cos_sin,
        interleave=index_interleave,
    )
    q_ref, scale_ref = ue8m0_quant(iq_ref)
    assert_fp8(iq_fp8, q_ref, "indexer-Q fp8")
    iw_ref = index_w * scale_ref * softmax_scale * head_scale
    torch.testing.assert_close(iw_out, iw_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_tokens", [1, 17, 512])
def test_fused_q_no_indexer(num_tokens: int):
    torch.manual_seed(3)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    q_pe = torch.randn(
        num_tokens, NUM_HEADS, ROPE_DIM, device=dev, dtype=torch.bfloat16
    )
    ql_nope = torch.randn(
        num_tokens, NUM_HEADS, KV_LORA, device=dev, dtype=torch.bfloat16
    )
    q_scale = torch.tensor([0.5], device=dev, dtype=torch.float32)
    q_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    _, _, mqa = K.fused_q(
        pos,
        q_pe,
        q_cos_sin,
        None,
        None,
        ql_nope,
        q_scale,
        None,
        0.0,
        0.0,
        has_indexer=False,
        index_rope_interleave=False,
    )
    s = q_scale.item()
    assert_fp8(mqa[:, :, :KV_LORA], (ql_nope.float() / s).to(FP8), "mqa ql_nope")
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, NUM_HEADS),
        q_cos_sin,
        interleave=True,
    )
    assert_fp8(mqa[:, :, KV_LORA:], (qpe_ref / s).to(FP8), "mqa q_pe")


@pytest.mark.parametrize("num_tokens", [1, 17, 512])
@pytest.mark.parametrize("has_indexer", [True, False])
def test_fused_q_bf16_query(num_tokens: int, has_indexer: bool):
    """bf16-query path (FlashMLA sparse, SM90/SM100): only the RoPE'd q_pe is
    produced (bf16, unquantized); ql_nope is consumed directly by the caller."""
    torch.manual_seed(6)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_pe = torch.randn(
        num_tokens, NUM_HEADS, ROPE_DIM, device=dev, dtype=torch.bfloat16
    )
    ql_nope = torch.randn(
        num_tokens, NUM_HEADS, KV_LORA, device=dev, dtype=torch.bfloat16
    )
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32)
    q_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    index_q = index_w = idx_cos_sin = None
    if has_indexer:
        index_q = torch.randn(
            num_tokens, INDEX_HEADS, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16
        )
        index_w = torch.randn(num_tokens, INDEX_HEADS, device=dev, dtype=torch.float32)
        idx_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    iq_fp8, iw_out, q_pe_out = K.fused_q(
        pos,
        q_pe,
        q_cos_sin,
        index_q,
        idx_cos_sin,
        ql_nope,
        q_scale,
        index_w,
        INDEX_HEAD_DIM**-0.5,
        INDEX_HEADS**-0.5,
        has_indexer=has_indexer,
        index_rope_interleave=False,
        quantize_mqa=False,
    )

    # MQA query: only the RoPE'd q_pe, bf16, unquantized.
    assert q_pe_out.dtype == torch.bfloat16
    assert q_pe_out.shape == (num_tokens, NUM_HEADS, ROPE_DIM)
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, NUM_HEADS),
        q_cos_sin,
        interleave=True,
    )
    assert_bf16(q_pe_out, qpe_ref, "bf16 q_pe RoPE")

    # Indexer-Q is unchanged on this path (still UE8M0 fp8 + folded weights).
    if has_indexer:
        assert index_q is not None
        iq_ref = rope(
            index_q.float(),
            pos.unsqueeze(-1).expand(num_tokens, INDEX_HEADS),
            idx_cos_sin,
            interleave=False,
        )
        q_ref, scale_ref = ue8m0_quant(iq_ref)
        assert_fp8(iq_fp8, q_ref, "indexer-Q fp8 (bf16-query path)")
        iw_ref = index_w * scale_ref * (INDEX_HEAD_DIM**-0.5) * (INDEX_HEADS**-0.5)
        torch.testing.assert_close(iw_out, iw_ref, rtol=1e-3, atol=1e-3)


# ── fused_eh_norm (MTP) ──────────────────────────────────────────────────────


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512])
def test_fused_eh_norm(num_tokens: int):
    torch.manual_seed(4)
    dev = "cuda"
    # Mix in a position-0 token to exercise the embeds-zeroing branch.
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    pos[0] = 0
    embeds = torch.randn(num_tokens, HIDDEN, device=dev, dtype=torch.bfloat16)
    prev = torch.randn(num_tokens, HIDDEN, device=dev, dtype=torch.bfloat16)
    ew = torch.randn(HIDDEN, device=dev, dtype=torch.bfloat16)
    hw = torch.randn(HIDDEN, device=dev, dtype=torch.bfloat16)

    out = K.fused_eh_norm(pos, embeds, prev, ew, hw, EPS)

    masked = torch.where(pos.unsqueeze(-1) == 0, torch.zeros_like(embeds), embeds)
    ref = torch.cat([rms_norm(masked, ew), rms_norm(prev, hw)], dim=-1)
    assert out.shape == (num_tokens, 2 * HIDDEN)
    assert_bf16(out, ref, "eh_norm")
