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

from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.cuda_vmm import (
    create_rank_major_peer_view,
)
from vllm.model_executor.layers.attention.pcp_peer_cache import (
    PCPPeerCacheFence,
    make_rank_major_block_tensor_view,
    make_rank_major_tensor_view,
)
from vllm.models.deepseek_v32.nvidia import kernels as K
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.mla.sparse_utils import (
    build_rotated_dcp_peer_block_table,
)

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
def test_fused_norm_rope_owner_store_writes_once(mla_dtype: str):
    """Owner-sharded PCP stores each valid producer row to exactly one peer."""
    torch.manual_seed(17)
    dev = "cuda"
    num_tokens = 7
    pcp_size = 3
    pcp_rank = 1
    max_pos = 16
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
    # [owner peer rank, owner-local slot]. The middle row is PAD/poisoned, and
    # the final two rows carry malformed positive rank/slot values. None may
    # address either peer cache even though their Q rows remain valid.
    owner_slots = torch.tensor(
        [[2, 8], [0, 3], [-1, -1], [1, 11], [2, 4], [pcp_size, 2], [1, max_pos]],
        device=dev,
        dtype=torch.int64,
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
        slot_mapping=None,
        indexer_k_cache=peer_index[pcp_rank],
        mla_kv_cache=peer_mla[pcp_rank],
        pcp_peer_indexer_k_cache=peer_index,
        pcp_peer_mla_kv_cache=peer_mla,
        pcp_size=pcp_size,
        pcp_owner_slot_mapping=owner_slots,
        mla_kv_cache_dtype=mla_dtype,
        mla_k_scale=(
            None if mla_dtype == "fp8_ds_mla" else torch.tensor([0.5], device=dev)
        ),
        has_indexer=True,
        index_rope_interleave=True,
    )

    expected_occupancy = torch.zeros(pcp_size, max_pos, device=dev, dtype=torch.bool)
    for owner, slot in owner_slots.tolist():
        if 0 <= owner < pcp_size and 0 <= slot < max_pos:
            expected_occupancy[owner, slot] = True

    mla_occupancy = peer_mla[:, 0].bool().any(dim=-1)
    torch.testing.assert_close(mla_occupancy, expected_occupancy)
    index_values = peer_index.view(pcp_size, -1)[:, : max_pos * INDEX_HEAD_DIM]
    index_occupancy = (
        index_values.view(pcp_size, max_pos, INDEX_HEAD_DIM).bool().any(dim=-1)
    )
    torch.testing.assert_close(index_occupancy, expected_occupancy)
    assert int(mla_occupancy.sum()) == num_tokens - 3
    assert int(index_occupancy.sum()) == num_tokens - 3
    assert_bf16(q_out, rms_norm(q_c, qw), "owner-sharded PCP q_c rmsnorm")


def _fused_norm_rope_owner_vmm_worker(
    rank: int,
    world_size: int,
    port: int,
    mla_dtype: str,
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
    owner_q_valid = None
    reference_q_valid = None
    expected_q_valid = None
    try:
        dev = torch.device(f"cuda:{rank}")
        # Each producer sees one replicated decode row followed by its four
        # DualChunkSwap prefill rows.  Only rank 0 owns the decode cache write;
        # the other ranks must still compute a valid Q for that PAD slot.
        num_tokens = 8
        num_blocks = 6
        block_size = 8
        mla_dim = 656 if mla_dtype == "fp8_ds_mla" else KV_LORA + ROPE_DIM
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
                [13, 0, 1, 0, 0, 5, 6, 7],
                [13, 2, 3, 12, 0, 5, 6, 7],
                [13, 4, 5, 10, 11, 5, 6, 7],
                [13, 6, 7, 8, 9, 5, 6, 7],
            ],
            dtype=torch.int64,
            device=dev,
        )
        slot_rows = torch.tensor(
            [
                [37, 19, 26, -1, -1, -1, -1, -1],
                [-1, 35, 4, 44, -1, -1, -1, -1],
                [-1, 11, 21, 30, 47, -1, -1, -1],
                [-1, 1, 15, 24, 40, -1, -1, -1],
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
        mla_k_scale = (
            None
            if mla_dtype == "fp8_ds_mla"
            else torch.tensor([0.5], device=dev, dtype=torch.float32)
        )

        owner_topk = torch.full((num_tokens, 16), 7, device=dev, dtype=torch.int32)
        owner_slots = torch.full((num_tokens, 2), -1, device=dev, dtype=torch.int64)
        # Route valid producer rows across every owner, including remote
        # destinations. The third-to-last row is a deliberately poisoned
        # duplicate-owner mapping and remains [-1, -1], distinct from the
        # ordinary PAD rows created by DualChunkSwap. The final rows carry
        # malformed positive rank and slot values to exercise bounds masks
        # against guarded, strided VMM allocations.
        for row, slot in enumerate(slot_rows[rank].tolist()):
            if slot >= 0:
                owner_slots[row, 0] = (rank + row + 1) % world_size
                owner_slots[row, 1] = slot
        owner_slots[-2] = torch.tensor([world_size, 0], device=dev, dtype=torch.int64)
        owner_slots[-1] = torch.tensor(
            [0, num_blocks * block_size], device=dev, dtype=torch.int64
        )
        owner_q = K.fused_norm_rope(
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
            owner_topk,
            slot_mapping=None,
            indexer_k_cache=local_index,
            mla_kv_cache=local_mla,
            pcp_peer_indexer_k_cache=peer_index,
            pcp_peer_mla_kv_cache=peer_mla,
            pcp_size=world_size,
            pcp_owner_slot_mapping=owner_slots,
            mla_kv_cache_dtype=mla_dtype,
            mla_k_scale=mla_k_scale,
            has_indexer=True,
            index_rope_interleave=True,
        )
        fence()

        # The ordinary single-rank path is the byte-level oracle for each
        # producer row. Route those reference rows to the same owner-local
        # destinations and compare the resulting sharded cache.
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
        reference_topk = torch.full_like(owner_topk, 7)
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
            mla_kv_cache_dtype=mla_dtype,
            mla_k_scale=mla_k_scale,
            has_indexer=True,
            index_rope_interleave=True,
        )

        local_reference = (
            reference_mla.cpu(),
            reference_index.cpu(),
            owner_slots.cpu(),
            slot_rows[rank].cpu(),
        )
        gathered_references = [None] * world_size
        dist.all_gather_object(
            gathered_references, local_reference, group=dist.group.WORLD
        )
        expected_mla = torch.zeros_like(local_reference[0])
        expected_index = torch.zeros_like(local_reference[1])
        for reference in gathered_references:
            assert reference is not None
            reference_mla, reference_index, reference_owners, reference_slots = (
                reference
            )
            for row, slot in enumerate(reference_slots.tolist()):
                if slot >= 0 and int(reference_owners[row, 0]) == rank:
                    block, offset = divmod(slot, block_size)
                    expected_mla[block, offset].copy_(reference_mla[block, offset])
                    # The indexer cache is physically block-packed as all
                    # 128-byte values followed by all 4-byte scales. Its
                    # nominal [block_size, 132] tensor shape is only backing
                    # storage, so copying one apparent row would include
                    # four bytes from the next token and omit this token's
                    # scale.
                    expected_index_block = expected_index[block].reshape(-1)
                    reference_index_block = reference_index[block].reshape(-1)
                    value_start = offset * INDEX_HEAD_DIM
                    expected_index_block[
                        value_start : value_start + INDEX_HEAD_DIM
                    ].copy_(
                        reference_index_block[
                            value_start : value_start + INDEX_HEAD_DIM
                        ]
                    )
                    scale_start = block_size * INDEX_HEAD_DIM + offset * 4
                    expected_index_block[scale_start : scale_start + 4].copy_(
                        reference_index_block[scale_start : scale_start + 4]
                    )

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
        owner_q_valid = owner_q.cpu()
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
    assert owner_q_valid is not None and reference_q_valid is not None
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
    assert torch.equal(owner_q_valid, reference_q_valid)
    assert_bf16(owner_q_valid, expected_q_valid, "owner PCP packed/mixed Q")


def _owner_history_64k_vmm_materialization_worker(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    """Compose owner stores, publication, translation, and >64K gathers."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    allocation = None
    index_allocation = None
    fence = None
    translation_mismatches = None
    index_translation_mismatches = None
    source_owner_min = None
    local_cache_mismatches = None
    local_index_cache_mismatches = None
    padding_mismatches = None
    index_padding_mismatches = None
    nope_mismatches = None
    rope_mismatches = None
    boundary_mismatches = None
    index_value_mismatches = None
    index_scale_mismatches = None
    index_dequant_mismatches = None
    index_boundary_mismatches = None
    peer_block_stride_evidence = None
    index_peer_block_stride_evidence = None
    try:
        dev = torch.device(f"cuda:{rank}")
        block_size = 64
        entry_bytes = 656
        head_dim = KV_LORA + ROPE_DIM

        # Four pages beyond 64K exercise owner-local page 256 on every owner.
        owner_pages = 257
        num_logical_pages = owner_pages * world_size
        total_tokens = num_logical_pages * block_size
        assert total_tokens == 65_792

        # The last block is a decoy. The 257 blocks named by the request use a
        # permutation, so neither direct stores nor gathers can accidentally
        # succeed by treating logical and physical page IDs as identical.
        num_local_blocks = owner_pages + 1
        block_bytes = block_size * entry_bytes
        requested_bytes = num_local_blocks * block_bytes
        allocation = create_rank_major_peer_view(
            (requested_bytes,),
            dtype=torch.uint8,
            group=dist.group.WORLD,
            first_dim_multiple=block_bytes,
            require_native_atomics=True,
            device=dev,
        )
        allocation.local_view.fill_(0xA5)
        local_cache = allocation.local_view[:requested_bytes].view(
            num_local_blocks, block_size, entry_bytes
        )
        local_cache.zero_()
        peer_cache = make_rank_major_tensor_view(allocation, local_cache)
        flattened_peer_cache, peer_block_stride = make_rank_major_block_tensor_view(
            peer_cache
        )
        peer_block_stride_evidence = peer_block_stride

        index_entry_bytes = INDEX_HEAD_DIM + 4
        index_block_bytes = block_size * index_entry_bytes
        index_requested_bytes = num_local_blocks * index_block_bytes
        index_allocation = create_rank_major_peer_view(
            (index_requested_bytes,),
            dtype=torch.uint8,
            group=dist.group.WORLD,
            first_dim_multiple=index_block_bytes,
            require_native_atomics=True,
            device=dev,
        )
        index_allocation.local_view.fill_(0xA5)
        local_index_cache = index_allocation.local_view[:index_requested_bytes].view(
            num_local_blocks, block_size, index_entry_bytes
        )
        local_index_cache.zero_()
        peer_index_cache = make_rank_major_tensor_view(
            index_allocation, local_index_cache
        )
        flattened_peer_index_cache, index_peer_block_stride = (
            make_rank_major_block_tensor_view(peer_index_cache)
        )
        index_peer_block_stride_evidence = index_peer_block_stride
        fence = PCPPeerCacheFence(dist.group.WORLD, dev)

        owner_page_ids = torch.arange(owner_pages, dtype=torch.int64, device=dev)
        physical_blocks = ((owner_page_ids * 73 + 19) % owner_pages).to(torch.int32)
        request_block_table = physical_blocks.unsqueeze(0)
        owner_block_tables = request_block_table.unsqueeze(0).expand(world_size, -1, -1)
        translated_block_table = build_rotated_dcp_peer_block_table(
            owner_block_tables,
            # Production binds the canonical global rank-major VMM view.
            local_rank=0,
            peer_block_stride=peer_block_stride,
            cp_kv_cache_interleave_size=block_size,
            block_size=block_size,
        )
        translated_index_block_table = build_rotated_dcp_peer_block_table(
            owner_block_tables,
            local_rank=0,
            peer_block_stride=index_peer_block_stride,
            cp_kv_cache_interleave_size=block_size,
            block_size=block_size,
        )

        logical_pages = torch.arange(num_logical_pages, dtype=torch.int64, device=dev)
        owners_per_page = logical_pages % world_size
        owner_pages_per_page = logical_pages // world_size
        expected_peer_blocks = (
            owners_per_page * peer_block_stride
            + physical_blocks[owner_pages_per_page].to(torch.int64)
        ).to(torch.int32)
        translation_mismatches = int(
            (translated_block_table[0] != expected_peer_blocks).count_nonzero()
        )
        expected_index_peer_blocks = (
            owners_per_page * index_peer_block_stride
            + physical_blocks[owner_pages_per_page].to(torch.int64)
        ).to(torch.int32)
        index_translation_mismatches = int(
            (
                translated_index_block_table[0] != expected_index_peer_blocks
            ).count_nonzero()
        )

        # First, rotating interior, and last offsets cover every in-page offset
        # while retaining exact checks around the 64K page boundary.
        page_offsets = torch.stack(
            (
                torch.zeros_like(logical_pages),
                1 + logical_pages % (block_size - 2),
                torch.full_like(logical_pages, block_size - 1),
            ),
            dim=1,
        )
        selected_pages = logical_pages[:, None].expand_as(page_offsets).reshape(-1)
        selected_offsets = page_offsets.reshape(-1)
        selected_tokens = selected_pages * block_size + selected_offsets
        selected_owners = selected_pages % world_size
        selected_owner_pages = selected_pages // world_size
        selected_physical_blocks = physical_blocks[selected_owner_pages].to(torch.int64)
        selected_owner_slots = selected_physical_blocks * block_size + selected_offsets
        producer_ranks = (selected_owners + selected_owner_pages) % world_size
        source_owner_counts = torch.bincount(
            producer_ranks * world_size + selected_owners,
            minlength=world_size * world_size,
        ).view(world_size, world_size)
        source_owner_min = int(source_owner_counts.min())

        # Every rank constructs the same small set of source rows and the same
        # independent ordinary-cache oracle. Only 3/64 rows per logical page
        # are materialized, keeping this >64K test bounded.
        generator = torch.Generator(device=dev)
        generator.manual_seed(0xC0FFEE)
        num_selected = selected_tokens.numel()
        positions = selected_tokens % 4096
        q_c = torch.randn(
            num_selected,
            Q_LORA,
            dtype=torch.bfloat16,
            device=dev,
            generator=generator,
        )
        kv_c = torch.randn(
            num_selected,
            KV_LORA,
            dtype=torch.bfloat16,
            device=dev,
            generator=generator,
        )
        k_pe = torch.randn(
            num_selected,
            ROPE_DIM,
            dtype=torch.bfloat16,
            device=dev,
            generator=generator,
        )
        index_k = torch.randn(
            num_selected,
            INDEX_HEAD_DIM,
            dtype=torch.bfloat16,
            device=dev,
            generator=generator,
        )
        q_weight = torch.randn(
            Q_LORA, dtype=torch.bfloat16, device=dev, generator=generator
        )
        kv_weight = torch.randn(
            KV_LORA, dtype=torch.bfloat16, device=dev, generator=generator
        )
        index_weight = torch.randn(
            INDEX_HEAD_DIM, dtype=torch.float32, device=dev, generator=generator
        )
        index_bias = torch.randn(
            INDEX_HEAD_DIM, dtype=torch.float32, device=dev, generator=generator
        )
        cos_sin = make_cos_sin(4096, ROPE_DIM, dev)

        def run_fused_store(
            row_ids: torch.Tensor,
            *,
            destination_cache: torch.Tensor | None,
            destination_peer_cache: torch.Tensor | None,
            destination_index_cache: torch.Tensor | None,
            destination_peer_index_cache: torch.Tensor | None,
            slot_mapping: torch.Tensor | None,
            owner_slot_mapping: torch.Tensor | None,
        ) -> None:
            topk = torch.empty((row_ids.numel(), 1), dtype=torch.int32, device=dev)
            K.fused_norm_rope(
                positions[row_ids],
                q_c[row_ids],
                q_weight,
                EPS,
                kv_c[row_ids],
                kv_weight,
                EPS,
                k_pe[row_ids],
                cos_sin,
                index_k[row_ids],
                index_weight,
                index_bias,
                EPS,
                cos_sin,
                topk,
                slot_mapping=slot_mapping,
                indexer_k_cache=destination_index_cache,
                mla_kv_cache=destination_cache,
                pcp_peer_indexer_k_cache=destination_peer_index_cache,
                pcp_peer_mla_kv_cache=destination_peer_cache,
                pcp_size=(world_size if destination_peer_cache is not None else 1),
                pcp_owner_slot_mapping=owner_slot_mapping,
                mla_kv_cache_dtype="fp8_ds_mla",
                has_indexer=True,
                index_rope_interleave=True,
            )

        local_rows = (producer_ranks == rank).nonzero().flatten()
        local_owner_mapping = torch.stack(
            (
                selected_owners[local_rows],
                selected_owner_slots[local_rows],
            ),
            dim=1,
        )
        run_fused_store(
            local_rows,
            destination_cache=None,
            destination_peer_cache=peer_cache,
            destination_index_cache=None,
            destination_peer_index_cache=peer_index_cache,
            slot_mapping=None,
            owner_slot_mapping=local_owner_mapping,
        )
        fence()

        reference_blocks = (num_selected + block_size - 1) // block_size
        reference_cache = torch.zeros(
            reference_blocks,
            block_size,
            entry_bytes,
            dtype=torch.uint8,
            device=dev,
        )
        reference_index_cache = torch.zeros(
            reference_blocks,
            block_size,
            index_entry_bytes,
            dtype=torch.uint8,
            device=dev,
        )
        all_rows = torch.arange(num_selected, dtype=torch.int64, device=dev)
        run_fused_store(
            all_rows,
            destination_cache=reference_cache,
            destination_peer_cache=None,
            destination_index_cache=reference_index_cache,
            destination_peer_index_cache=None,
            slot_mapping=all_rows,
            owner_slot_mapping=None,
        )
        reference_entries = reference_cache.view(-1, entry_bytes)[:num_selected]

        expected_local_cache = torch.zeros_like(local_cache)
        owned_rows = (selected_owners == rank).nonzero().flatten()
        expected_local_cache[
            selected_physical_blocks[owned_rows],
            selected_offsets[owned_rows],
        ] = reference_entries[owned_rows]
        local_cache_mismatches = int(
            (local_cache != expected_local_cache).count_nonzero()
        )

        # Indexer blocks are packed as 64 value rows followed by 64 scales;
        # the nominal [64, 132] shape is backing storage rather than a
        # per-token struct. Copy the two byte ranges independently.
        expected_local_index_cache = torch.zeros_like(local_index_cache)
        reference_index_blocks = reference_index_cache.view(reference_blocks, -1)
        expected_index_blocks = expected_local_index_cache.view(num_local_blocks, -1)
        for row in owned_rows.tolist():
            reference_block, reference_offset = divmod(row, block_size)
            physical_block = int(selected_physical_blocks[row])
            physical_offset = int(selected_offsets[row])
            reference_value_start = reference_offset * INDEX_HEAD_DIM
            physical_value_start = physical_offset * INDEX_HEAD_DIM
            expected_index_blocks[
                physical_block,
                physical_value_start : physical_value_start + INDEX_HEAD_DIM,
            ].copy_(
                reference_index_blocks[
                    reference_block,
                    reference_value_start : reference_value_start + INDEX_HEAD_DIM,
                ]
            )
            reference_scale_start = block_size * INDEX_HEAD_DIM + reference_offset * 4
            physical_scale_start = block_size * INDEX_HEAD_DIM + physical_offset * 4
            expected_index_blocks[
                physical_block, physical_scale_start : physical_scale_start + 4
            ].copy_(
                reference_index_blocks[
                    reference_block,
                    reference_scale_start : reference_scale_start + 4,
                ]
            )
        local_index_cache_mismatches = int(
            (local_index_cache != expected_local_index_cache).count_nonzero()
        )
        padding_mismatches = int(
            (allocation.local_view[requested_bytes:] != 0xA5).count_nonzero()
        )
        index_padding_mismatches = int(
            (
                index_allocation.local_view[index_requested_bytes:] != 0xA5
            ).count_nonzero()
        )

        reference_workspace = torch.empty(
            num_selected, head_dim, dtype=torch.bfloat16, device=dev
        )
        reference_block_table = torch.arange(
            reference_blocks, dtype=torch.int32, device=dev
        ).unsqueeze(0)
        workspace_starts = torch.zeros(1, dtype=torch.int32, device=dev)
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            reference_cache,
            reference_workspace,
            reference_block_table,
            workspace_starts,
            1,
        )

        actual_workspace = torch.empty(
            total_tokens, head_dim, dtype=torch.bfloat16, device=dev
        )
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            flattened_peer_cache,
            actual_workspace,
            translated_block_table,
            workspace_starts,
            1,
        )
        expected_workspace = torch.zeros_like(actual_workspace)
        expected_workspace[selected_tokens] = reference_workspace

        nope_close = torch.isclose(
            actual_workspace[:, :KV_LORA],
            expected_workspace[:, :KV_LORA],
            atol=1e-3,
            rtol=1e-2,
        )
        nope_mismatches = int((~nope_close).count_nonzero())
        rope_mismatches = int(
            (
                actual_workspace[:, KV_LORA:] != expected_workspace[:, KV_LORA:]
            ).count_nonzero()
        )
        # Logical pages 1020..1023 are owner-local page 255 on ranks 0..3;
        # pages 1024..1027 are owner-local page 256. Their first and last
        # tokens also bracket logical page 1023 and the 65,536-token boundary.
        boundary_pages = torch.arange(1020, 1028, dtype=torch.int64, device=dev)
        boundary_tokens = (
            boundary_pages[:, None] * block_size
            + torch.tensor([0, block_size - 1], dtype=torch.int64, device=dev)
        ).flatten()
        boundary_mismatches = int(
            (
                actual_workspace[boundary_tokens] != expected_workspace[boundary_tokens]
            ).count_nonzero()
        )

        reference_index_values = torch.zeros(
            num_selected, INDEX_HEAD_DIM, dtype=torch.uint8, device=dev
        )
        reference_index_scales = torch.zeros(
            num_selected, 4, dtype=torch.uint8, device=dev
        )
        reference_index_cu_seq_lens = torch.tensor(
            [0, num_selected], dtype=torch.int32, device=dev
        )
        ops.cp_gather_indexer_k_quant_cache(
            reference_index_cache,
            reference_index_values,
            reference_index_scales,
            reference_block_table,
            reference_index_cu_seq_lens,
        )

        actual_index_values = torch.zeros(
            total_tokens, INDEX_HEAD_DIM, dtype=torch.uint8, device=dev
        )
        actual_index_scales = torch.zeros(
            total_tokens, 4, dtype=torch.uint8, device=dev
        )
        actual_index_cu_seq_lens = torch.tensor(
            [0, total_tokens], dtype=torch.int32, device=dev
        )
        ops.cp_gather_indexer_k_quant_cache(
            flattened_peer_index_cache,
            actual_index_values,
            actual_index_scales,
            translated_index_block_table,
            actual_index_cu_seq_lens,
        )
        expected_index_values = torch.zeros_like(actual_index_values)
        expected_index_scales = torch.zeros_like(actual_index_scales)
        expected_index_values[selected_tokens] = reference_index_values
        expected_index_scales[selected_tokens] = reference_index_scales
        index_value_mismatches = int(
            (actual_index_values != expected_index_values).count_nonzero()
        )
        index_scale_mismatches = int(
            (actual_index_scales != expected_index_scales).count_nonzero()
        )
        actual_index_dequant = actual_index_values.view(FP8).float() * (
            actual_index_scales.view(torch.float32)
        )
        expected_index_dequant = expected_index_values.view(FP8).float() * (
            expected_index_scales.view(torch.float32)
        )
        index_dequant_mismatches = int(
            (actual_index_dequant != expected_index_dequant).count_nonzero()
        )
        index_boundary_mismatches = int(
            (
                actual_index_dequant[boundary_tokens]
                != expected_index_dequant[boundary_tokens]
            ).count_nonzero()
        )

        # No rank may assert while another can still access its VMM allocation.
        torch.cuda.synchronize(dev)
        dist.barrier()
        fence.close()
        fence = None
        index_allocation.close()
        index_allocation = None
        allocation.close()
        allocation = None
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    assert translation_mismatches == 0
    assert index_translation_mismatches == 0
    assert source_owner_min is not None and source_owner_min > 0
    assert peer_block_stride_evidence is not None
    assert peer_block_stride_evidence >= num_local_blocks
    assert index_peer_block_stride_evidence is not None
    assert index_peer_block_stride_evidence >= num_local_blocks
    assert local_cache_mismatches == 0
    assert local_index_cache_mismatches == 0
    assert padding_mismatches == 0
    assert index_padding_mismatches == 0
    assert nope_mismatches == 0
    assert rope_mismatches == 0
    assert boundary_mismatches == 0
    assert index_value_mismatches == 0
    assert index_scale_mismatches == 0
    assert index_dequant_mismatches == 0
    assert index_boundary_mismatches == 0


@pytest.mark.distributed(num_gpus=4)
def test_fused_norm_rope_owner_vmm_pcp4() -> None:
    world_size = 4
    if torch.cuda.device_count() < world_size or not all(
        source == destination or torch.cuda.can_device_access_peer(source, destination)
        for source in range(world_size)
        for destination in range(world_size)
    ):
        pytest.skip("owner-sharded PCP publication requires four peer-accessible GPUs")
    mp.spawn(
        _fused_norm_rope_owner_vmm_worker,
        args=(world_size, get_open_port(), "fp8"),
        nprocs=world_size,
    )


@pytest.mark.distributed(num_gpus=4)
def test_fused_norm_rope_owner_vmm_pcp4_fp8_ds_mla() -> None:
    world_size = 4
    if torch.cuda.device_count() < world_size or not all(
        source == destination or torch.cuda.can_device_access_peer(source, destination)
        for source in range(world_size)
        for destination in range(world_size)
    ):
        pytest.skip("owner-sharded PCP publication requires four peer-accessible GPUs")
    mp.spawn(
        _fused_norm_rope_owner_vmm_worker,
        args=(world_size, get_open_port(), "fp8_ds_mla"),
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


@pytest.mark.distributed(num_gpus=4)
def test_owner_history_64k_vmm_store_publish_translate_and_materialize() -> None:
    world_size = 4
    if torch.cuda.device_count() < world_size or not all(
        source == destination or torch.cuda.can_device_access_peer(source, destination)
        for source in range(world_size)
        for destination in range(world_size)
    ):
        pytest.skip("owner-sharded 64K gather requires four peer-accessible CUDA GPUs")
    mp.spawn(
        _owner_history_64k_vmm_materialization_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )


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
