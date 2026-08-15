# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant 4bit MSE cache for DSA Indexer (Patch 08).

Store: L2-normalize K, Hadamard rotate (y = x_hat @ Pi), Lloyd-Max 4bit in rotated
space + eff_scale (norm correction). Decode/prefill score in rotated space; fold Pi
into indexer.wq_b at load (same strategy as MLA W_UK_T fold).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.triton_mla_tq import (
    _pack_bits_rows,
    _unpack_bits_rows,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_mla_decode import (
    _tq_mla_gather_centroids,
    _tq_mla_gather_centroids_1d,
)

logger = init_logger(__name__)

_BF16 = torch.bfloat16
_FP8_DTYPE = torch.float8_e4m3fn

INDEXER_HEAD_DIM = 128
INDEXER_MSE_BITS = 4
INDEXER_MSE_BYTES = 64
INDEXER_PACKED_BYTES = 66
INDEXER_FP8_SLOT_BYTES = 132

_INDEXER_TQ_BUFFERS: dict[str, dict] = {}
_INDEXER_TQ_FUSED_WARMED: set[str] = set()
_INDEXER_TQ_STORE_WARMED: set[str] = set()
_INDEXER_TQ_STORE_DISABLE_AUTOTUNE = (
    os.environ.get("VLLM_INDEXER_TQ_STORE_AUTOTUNE", "0") != "1"
)

_INDEXER_TQ_STORE_AUTOTUNE_CONFIGS = (
    [triton.Config({}, num_warps=4, num_stages=1)]
    if _INDEXER_TQ_STORE_DISABLE_AUTOTUNE
    else [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
    ]
)


def is_indexer_tq_4bit_enabled() -> bool:
    return os.environ.get("VLLM_INDEXER_KV_TQ_4BIT", "0") == "1"


def use_indexer_tq_hadamard() -> bool:
    """Hadamard @Pi before Lloyd-Max store; Q-side Pi fold in wq_b when enabled.

    Set VLLM_INDEXER_TQ_HADAMARD=0 to use legacy direct 4bit (norm -> bucketize).
    """
    return os.environ.get("VLLM_INDEXER_TQ_HADAMARD", "1") == "1"


def use_indexer_tq_fused_decode() -> bool:
    """Fused TQ4 paged logits (skip sync + DeepGEMM decode bridge)."""
    return os.environ.get("VLLM_INDEXER_TQ_FUSED_DECODE", "1") == "1"


INDEXER_DECODE_TILE = 64
# CUDA grid-Y cap is 65535; fixed launch width for long-context sync/decode.
INDEXER_SYNC_FIXED_GRID = 40960


def indexer_sync_fixed_grid() -> int:
    return max(
        1,
        int(
            os.environ.get(
                "VLLM_INDEXER_TQ_SYNC_FIXED_GRID", str(INDEXER_SYNC_FIXED_GRID)
            )
        ),
    )


def indexer_sync_inner_tile(max_model_len: int) -> int:
    """Tokens per program when grid-Y is fixed (e.g. 128k -> 4, 200k -> 8).

    Rounded up to a power of two because Triton ``tl.arange(0, TILE)`` requires it.
    """
    fg = indexer_sync_fixed_grid()
    needed = (max_model_len + fg - 1) // fg
    tile = 1
    while tile < needed:
        tile <<= 1
    return tile


def indexer_sync_num_chunks(max_model_len: int) -> int:
    fg = indexer_sync_fixed_grid()
    return (max_model_len + fg - 1) // fg


def indexer_decode_grid_n(max_model_len: int) -> int:
    """Fixed tile grid width for CUDA graph (40960 -> 640)."""
    return (max_model_len + INDEXER_DECODE_TILE - 1) // INDEXER_DECODE_TILE


# SM90 fp8_fp4_paged_mqa_logits (Hopper): BLOCK_KV=64, kNumMathWarpGroups=2.
INDEXER_DG_SYNC_BLOCK_KV = 64
INDEXER_DG_SYNC_BLOCKS_PER_SPLIT = 2


def indexer_tq_sync_mode() -> str:
    """TQ4 decode FP8 workspace sync: fixed | pos | tile | chunk | sm."""
    return os.environ.get("VLLM_INDEXER_TQ_SYNC_MODE", "fixed").lower()


def resolve_indexer_tq_sync_mode(max_model_len: int | None) -> str:
    """Pick sync launch mode; auto-fallback when pos grid would exceed CUDA limit."""
    mode = indexer_tq_sync_mode()
    if mode == "pos" and max_model_len is not None and max_model_len > 65535:
        return "fixed"
    return mode


_INDEXER_TQ_SYNC_LOGGED: set[str] = set()


def _log_indexer_tq_sync_config_once(mode: str, max_model_len: int | None) -> None:
    env_mode = indexer_tq_sync_mode()
    key = f"{mode}:{max_model_len}:{env_mode}"
    if key in _INDEXER_TQ_SYNC_LOGGED:
        return
    _INDEXER_TQ_SYNC_LOGGED.add(key)
    if mode == "fixed":
        fg = indexer_sync_fixed_grid()
        inner = indexer_sync_inner_tile(max_model_len or 0)
        logger.info_once(
            "Indexer TQ4 decode sync: mode=%s (env VLLM_INDEXER_TQ_SYNC_MODE=%s), "
            "grid=(batch,%d), inner_tile=%d, max_model_len=%s.",
            mode,
            env_mode,
            fg,
            inner,
            max_model_len,
        )
    elif mode == "chunk":
        fg = indexer_sync_fixed_grid()
        nc = indexer_sync_num_chunks(max_model_len or 0)
        logger.info_once(
            "Indexer TQ4 decode sync: mode=%s (env VLLM_INDEXER_TQ_SYNC_MODE=%s), "
            "%d launches x grid=(batch,%d), max_model_len=%s.",
            mode,
            env_mode,
            nc,
            fg,
            max_model_len,
        )
    elif mode == "tile":
        gn = indexer_decode_grid_n(max_model_len or 0)
        logger.info_once(
            "Indexer TQ4 decode sync: mode=%s (env VLLM_INDEXER_TQ_SYNC_MODE=%s), "
            "grid=(batch,%d), tile=%d, max_model_len=%s.",
            mode,
            env_mode,
            gn,
            INDEXER_DECODE_TILE,
            max_model_len,
        )
    else:
        logger.info_once(
            "Indexer TQ4 decode sync: mode=%s (env VLLM_INDEXER_TQ_SYNC_MODE=%s), "
            "max_model_len=%s.",
            mode,
            env_mode,
            max_model_len,
        )


def indexer_tq_sync_blocks_per_split() -> int:
    return max(
        1,
        int(
            os.environ.get(
                "VLLM_INDEXER_TQ_SYNC_DG_BLOCKS_PER_SPLIT",
                str(INDEXER_DG_SYNC_BLOCKS_PER_SPLIT),
            )
        ),
    )


def indexer_packed_head_dim() -> int:
    return (
        INDEXER_PACKED_BYTES if is_indexer_tq_4bit_enabled() else INDEXER_FP8_SLOT_BYTES
    )


def _get_indexer_tq_buffers(device: torch.device) -> dict:
    key = str(device)
    if key not in _INDEXER_TQ_BUFFERS:
        D = INDEXER_HEAD_DIM
        cents = get_centroids(D, INDEXER_MSE_BITS).to(
            device=device, dtype=torch.float32
        )
        c_sorted, _ = cents.sort()
        buf: dict = {
            "centroids_bf16": cents.to(_BF16),
            "centroids_fp32": cents,
            "midpoints": (c_sorted[:-1] + c_sorted[1:]) / 2,
        }
        if use_indexer_tq_hadamard():
            pi = _build_hadamard(D, key).to(device=device, dtype=torch.float32)
            buf["Pi"] = pi
            buf["PiT"] = pi
            buf["Pi_bf16"] = pi.to(_BF16)
            logger.info_once(
                "Indexer TurboQuant 4bit + Hadamard: %d B/token (FP8 legacy %d B).",
                INDEXER_PACKED_BYTES,
                INDEXER_FP8_SLOT_BYTES,
            )
        else:
            logger.info_once(
                "Indexer TurboQuant 4bit direct (no Hadamard): %d B/token.",
                INDEXER_PACKED_BYTES,
            )
        _INDEXER_TQ_BUFFERS[key] = buf
    return _INDEXER_TQ_BUFFERS[key]


def fold_indexer_pi_at_load(indexer) -> None:
    """Fold Pi into indexer.wq_b so Q is pre-rotated (decode + prefill score path).

    out_head = qr @ W_h^T; out_head_rot = out_head @ Pi = qr @ (Pi @ W_h)^T.
    """
    if getattr(indexer, "_indexer_tq_pi_folded", False):
        return
    if not is_indexer_tq_4bit_enabled() or not use_indexer_tq_hadamard():
        indexer._indexer_tq_pi_folded = True
        return

    w = indexer.wq_b.weight
    if w is None or w.numel() == 0:
        indexer._indexer_tq_pi_folded = False
        return

    device = w.device
    buf = _get_indexer_tq_buffers(device)
    pi = buf["Pi"].to(device=device, dtype=torch.float32)
    d = indexer.head_dim
    n_head = indexer.n_head
    w_f = w.data.to(torch.float32)
    w_new = w_f.clone()
    for h in range(n_head):
        sl = slice(h * d, (h + 1) * d)
        w_new[sl, :] = pi @ w_f[sl, :]
    indexer.wq_b.weight.data.copy_(w_new.to(dtype=w.dtype))
    indexer._indexer_tq_pi_folded = True


def _get_scores_workspace(
    device: torch.device,
    batch_size: int,
    max_model_len: int,
) -> torch.Tensor:
    """Persistent float32 scores buffer for fused decode+topk (CUDA-graph friendly)."""
    buf = _get_indexer_tq_buffers(device)
    ws = buf.get("scores_ws")
    if (
        ws is None
        or ws.shape[0] < batch_size
        or ws.shape[1] < max_model_len
        or ws.device != device
    ):
        ws = torch.empty(
            batch_size,
            max_model_len,
            device=device,
            dtype=torch.float32,
        )
        ws.fill_(float("-inf"))
        buf["scores_ws"] = ws
    return ws[:batch_size, :max_model_len]


def _launch_tq4_paged_mqa_scores(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_1d: torch.Tensor,
    scores: torch.Tensor,
    max_model_len: int,
) -> None:
    batch_size, num_heads, head_dim = q_fp8.shape
    assert head_dim == INDEXER_HEAD_DIM
    _, block_size, _ = packed_cache.shape
    buf = _get_indexer_tq_buffers(q_fp8.device)
    block_d = triton.next_power_of_2(INDEXER_HEAD_DIM)

    _tq4_paged_mqa_logits_kernel[(batch_size, max_model_len)](
        q_fp8,
        weights,
        packed_cache,
        block_table,
        seq_lens_1d,
        scores,
        buf["centroids_bf16"],
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights.stride(0),
        scores.stride(0),
        block_table.stride(0),
        packed_cache.stride(0),
        packed_cache.stride(1),
        block_size,
        INDEXER_HEAD_DIM,
        num_heads,
        INDEXER_MSE_BITS,
        INDEXER_MSE_BYTES,
        block_d,
    )


def pack_k_mse_torch(k: torch.Tensor, buf: dict) -> torch.Tensor:
    """bf16 k [N,128] -> uint8 [N,66]: norm -> [@Pi] -> Lloyd-Max + eff_scale."""
    k_f = k.to(torch.float32)
    norms = k_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_hat = k_f / norms
    if use_indexer_tq_hadamard():
        pi = buf["Pi"].to(device=k.device, dtype=torch.float32)
        y = x_hat @ pi
    else:
        y = x_hat
    idx = torch.bucketize(y.contiguous(), buf["midpoints"].to(k.device))
    idx = idx.clamp(max=(1 << INDEXER_MSE_BITS) - 1)
    y_hat = buf["centroids_bf16"][idx.to(torch.int64)].to(torch.float32)
    c_norm = y_hat.norm(dim=-1).clamp(min=1e-8)
    eff_scale = (norms.squeeze(-1) / c_norm).to(torch.float16)
    packed_idx = _pack_bits_rows(idx.to(torch.int32), bits=INDEXER_MSE_BITS)
    scale_bytes = eff_scale.view(torch.uint8).view(k.shape[0], 2)
    return torch.cat([packed_idx, scale_bytes], dim=-1)


def dequant_packed_rows_torch(packed: torch.Tensor, buf: dict) -> torch.Tensor:
    """uint8 [T,66] -> fp32 [T,128]."""
    idx = _unpack_bits_rows(
        packed[:, :INDEXER_MSE_BYTES].contiguous(),
        bits=INDEXER_MSE_BITS,
        D=INDEXER_HEAD_DIM,
    )
    y_hat = buf["centroids_bf16"][idx.to(torch.int64)].to(torch.float32)
    scale = (
        packed[:, INDEXER_MSE_BYTES:INDEXER_PACKED_BYTES]
        .contiguous()
        .view(torch.float16)
        .to(torch.float32)
    )
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1)
    return y_hat * scale


def _dequant_packed_slot_torch(
    packed_cache: torch.Tensor,
    block_id: int,
    slot_in_block: int,
    block_size: int,
    buf: dict,
) -> torch.Tensor:
    packed_base = block_id * block_size + slot_in_block
    packed = packed_cache.view(-1, INDEXER_PACKED_BYTES)[packed_base]
    return dequant_packed_rows_torch(packed.unsqueeze(0), buf).squeeze(0)


def tq4_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Reference: paged TQ4 cache -> logits (next_n=1 decode)."""
    batch_size, num_heads, head_dim = q_fp8.shape
    assert head_dim == INDEXER_HEAD_DIM
    block_size = packed_cache.shape[1]
    device = q_fp8.device
    buf = _get_indexer_tq_buffers(device)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)

    logits = torch.full(
        (batch_size, max_model_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    q_f = q_fp8.to(torch.float32)
    for batch_id in range(batch_size):
        seq_len = int(seq_lens[batch_id].item())
        if seq_len <= 0:
            continue
        q_h = q_f[batch_id]
        w_h = weights[batch_id]
        for pos in range(seq_len):
            block_table_id = pos // block_size
            slot_in_block = pos % block_size
            block_id = int(block_table[batch_id, block_table_id].item())
            k_f = _dequant_packed_slot_torch(
                packed_cache, block_id, slot_in_block, block_size, buf
            )
            scores = torch.relu(torch.matmul(q_h, k_f))
            logits[batch_id, pos] = (scores * w_h).sum()
    return logits


def _bf16_to_fp8_indexer(
    k_bf16: torch.Tensor,
    scale_fmt: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_f = k_bf16.to(torch.float32)
    amax = k_f.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-4)
    scale = amax / 448.0
    if scale_fmt == "ue8m0":
        scale = torch.exp2(torch.ceil(torch.log2(scale.clamp(min=1e-8))))
    k_fp8 = (k_f / scale).clamp(-448.0, 448.0).to(_FP8_DTYPE)
    k_scale = scale.squeeze(-1).to(torch.float32)
    return k_fp8, k_scale.view(torch.uint8).view(-1, 4)


@triton.autotune(
    configs=_INDEXER_TQ_STORE_AUTOTUNE_CONFIGS,
    key=["D", "MSE_BITS"],
)
@triton.jit
def _indexer_tq_fused_store_kernel(
    K_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    Midpoints_ptr,
    Centroids_ptr,
    Pi_ptr,
    stride_k,
    stride_cache_row,
    D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    USE_HADAMARD: tl.constexpr,
):
    """Fused Indexer TQ4 store: norm -> [@Pi] -> bucketize -> pack -> scatter."""
    tid = tl.program_id(0)
    slot = tl.load(Slot_mapping_ptr + tid)
    if slot < 0:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    k_base = K_ptr + tid * stride_k
    k_vec = tl.load(k_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    norm_sq = tl.sum(k_vec * k_vec, axis=0)
    norm = tl.sqrt(norm_sq)
    norm_safe = tl.maximum(norm, 1e-8)
    x_hat = k_vec / norm_safe

    if USE_HADAMARD:
        offs_i = tl.arange(0, BLOCK_D)
        offs_j = tl.arange(0, BLOCK_D)
        pi_mask = (offs_i[:, None] < D) & (offs_j[None, :] < D)
        pi_block = tl.load(
            Pi_ptr + offs_i[:, None] * D + offs_j[None, :],
            mask=pi_mask,
            other=0.0,
        ).to(tl.float32)
        y_row = tl.dot(x_hat[None, :], pi_block)
        y = tl.reshape(y_row, [BLOCK_D])
    else:
        y = x_hat

    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(y >= mid_val, mid + 1, lo)
        hi = tl.where(y >= mid_val, hi, mid)
    idx = tl.minimum(lo, N_CENTROIDS - 1)

    y_hat = _tq_mla_gather_centroids_1d(Centroids_ptr, idx, d_mask, BLOCK_D, MSE_BITS)
    c_norm_sq = tl.sum(y_hat * y_hat, axis=0)
    c_norm = tl.sqrt(tl.maximum(c_norm_sq, 1e-8))
    eff_scale = (norm / c_norm).to(tl.float16)
    scale_u16 = eff_scale.to(tl.uint16, bitcast=True)

    idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
    shifts_4 = tl.arange(0, 2) * 4
    packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
    mse_offs = tl.arange(0, BLOCK_D // 2)
    mse_mask = mse_offs < MSE_BYTES

    slot_base = slot.to(tl.int64) * stride_cache_row
    tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)
    tl.store(KV_cache_ptr + slot_base + MSE_BYTES, (scale_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + MSE_BYTES + 1,
        ((scale_u16 >> 8) & 0xFF).to(tl.uint8),
    )


def indexer_tq_store_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Triton fused TQ4 pack + scatter (replaces pack_k_mse_torch hot path)."""
    N = slot_mapping.shape[0]
    if N == 0:
        return
    buf = _get_indexer_tq_buffers(k.device)
    block_d = triton.next_power_of_2(INDEXER_HEAD_DIM)
    cache_flat = kv_cache.view(-1, INDEXER_PACKED_BYTES)
    slots = slot_mapping[:N].to(torch.int32)

    use_hadamard = use_indexer_tq_hadamard()
    pi_bf16 = buf.get("Pi_bf16")
    if pi_bf16 is None:
        pi_bf16 = torch.eye(INDEXER_HEAD_DIM, device=k.device, dtype=_BF16)

    grid = (N,)
    _indexer_tq_fused_store_kernel[grid](
        k[:N],
        cache_flat,
        slots,
        buf["midpoints"],
        buf["centroids_fp32"],
        pi_bf16,
        k.stride(0),
        INDEXER_PACKED_BYTES,
        INDEXER_HEAD_DIM,
        INDEXER_MSE_BITS,
        INDEXER_MSE_BYTES,
        INDEXER_PACKED_BYTES,
        block_d,
        1 << INDEXER_MSE_BITS,
        use_hadamard,
    )


def indexer_tq_store_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if k.numel() == 0:
        return
    N = slot_mapping.shape[0]
    if N == 0:
        return
    if k.is_cuda:
        indexer_tq_store_and_cache_triton(k, kv_cache, slot_mapping)
        return
    buf = _get_indexer_tq_buffers(k.device)
    k = k[:N]
    packed = pack_k_mse_torch(k, buf)
    cache_flat = kv_cache.view(-1, INDEXER_PACKED_BYTES)
    slots = slot_mapping[:N].to(torch.int64)
    valid = slots >= 0
    slots = slots.clamp(min=0)
    current = cache_flat.index_select(0, slots)
    new = torch.where(valid.view(-1, 1), packed, current)
    cache_flat.index_copy_(0, slots, new)


@triton.jit
def _tq4_prefill_gather_dequant_fp8_kernel(
    Packed_cache_ptr,
    Block_table_ptr,
    Token_to_seq_ptr,
    Cu_seq_lens_ptr,
    Dst_k_ptr,
    Dst_scale_ptr,
    Centroids_ptr,
    N,
    dst_k_stride_n,
    dst_scale_stride_n,
    block_table_stride_r,
    stride_packed_nb,
    stride_packed_bs,
    block_size: tl.constexpr,
    D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    """TQ4 paged K -> linear FP8 + per-token UE8M0 scale (DeepGEMM input format)."""
    n_j = tl.program_id(0)
    if n_j >= N:
        return
    b_id = tl.load(Token_to_seq_ptr + n_j)
    s_start = tl.load(Cu_seq_lens_ptr + b_id)
    pos = n_j - s_start
    bt_id = pos // block_size
    slot = pos % block_size
    blk = tl.load(Block_table_ptr + b_id * block_table_stride_r + bt_id)
    p_base = blk * stride_packed_nb + slot * stride_packed_bs

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    bit_off = d_offs * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    raw0 = tl.load(Packed_cache_ptr + p_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    raw1 = tl.load(Packed_cache_ptr + p_base + byte_idx + 1, mask=d_mask, other=0).to(
        tl.int32
    )
    raw16 = raw0 | (raw1 << 8)
    cidx = (raw16 >> bit_shift) & umask
    y_hat = _tq_mla_gather_centroids_1d(Centroids_ptr, cidx, d_mask, BLOCK_D, MSE_BITS)
    n_lo = tl.load(Packed_cache_ptr + p_base + MSE_BYTES).to(tl.uint16)
    n_hi = tl.load(Packed_cache_ptr + p_base + MSE_BYTES + 1).to(tl.uint16)
    token_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    k_f = y_hat.to(tl.float32) * token_scale

    amax = tl.max(tl.where(d_mask, tl.abs(k_f), 0.0), axis=0)
    amax = tl.maximum(amax, 1e-4)
    scale = amax / FP8_MAX
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    fp8_val = (k_f / scale).to(tl.float8e4nv)
    tl.store(Dst_k_ptr + n_j * dst_k_stride_n + d_offs, fp8_val, mask=d_mask)

    scale_u32 = scale.to(tl.uint32, bitcast=True)
    base = Dst_scale_ptr + n_j * dst_scale_stride_n
    tl.store(base + 0, ((scale_u32 >> 0) & 0xFF).to(tl.uint8))
    tl.store(base + 1, ((scale_u32 >> 8) & 0xFF).to(tl.uint8))
    tl.store(base + 2, ((scale_u32 >> 16) & 0xFF).to(tl.uint8))
    tl.store(base + 3, ((scale_u32 >> 24) & 0xFF).to(tl.uint8))


def indexer_tq_cp_gather_dequant_fp8(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    block_size: int,
    scale_fmt: str | None,
) -> None:
    """Prefill gather: TQ paged cache -> linear FP8 (Triton, cp_gather replacement)."""
    total_tokens = dst_k.shape[0]
    if total_tokens == 0:
        return
    device = kv_cache.device
    buf = _get_indexer_tq_buffers(device)

    # token_to_seq[n] = batch index for token n (precompute via searchsorted on GPU)
    token_to_seq = torch.searchsorted(
        cu_seq_lens[1:].contiguous(),
        torch.arange(total_tokens, device=device, dtype=cu_seq_lens.dtype),
        right=True,
    ).to(torch.int32)

    block_d = triton.next_power_of_2(INDEXER_HEAD_DIM)
    _tq4_prefill_gather_dequant_fp8_kernel[(total_tokens,)](
        kv_cache,
        block_table,
        token_to_seq,
        cu_seq_lens,
        dst_k,
        dst_scale,
        buf["centroids_bf16"],
        total_tokens,
        dst_k.stride(0),
        dst_scale.stride(0),
        block_table.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        block_size,
        INDEXER_HEAD_DIM,
        INDEXER_MSE_BITS,
        INDEXER_MSE_BYTES,
        block_d,
        448.0,
        scale_fmt == "ue8m0",
        num_warps=4,
    )


@triton.jit
def _indexer_tq_sync_decode_kernel(
    Packed_cache_ptr,
    Fp8_ws_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_ptr,
    block_table_stride,
    stride_packed_nb,
    stride_packed_bs,
    stride_ws_nb,
    stride_ws_bs,
    block_size: tl.constexpr,
    D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    FP8_SLOT_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    KV_OFFSET: tl.constexpr,
):
    batch_id = tl.program_id(0)
    pos = tl.program_id(1) + KV_OFFSET
    seq_len = tl.load(Seq_lens_ptr + batch_id)
    if pos >= seq_len:
        return

    block_table_id = pos // block_size
    slot_in_block = pos % block_size
    block_id = tl.load(Block_table_ptr + batch_id * block_table_stride + block_table_id)

    packed_base = block_id * stride_packed_nb + slot_in_block * stride_packed_bs
    fp8_base = block_id * stride_ws_nb + slot_in_block * stride_ws_bs

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    bit_off = d_offs * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    raw0 = tl.load(Packed_cache_ptr + packed_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    raw1 = tl.load(
        Packed_cache_ptr + packed_base + byte_idx + 1, mask=d_mask, other=0
    ).to(tl.int32)
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask

    y_hat = _tq_mla_gather_centroids_1d(Centroids_ptr, idx, d_mask, BLOCK_D, MSE_BITS)
    n_lo = tl.load(Packed_cache_ptr + packed_base + MSE_BYTES).to(tl.uint16)
    n_hi = tl.load(Packed_cache_ptr + packed_base + MSE_BYTES + 1).to(tl.uint16)
    token_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    k_f = y_hat.to(tl.float32) * token_scale

    amax = tl.max(tl.where(d_mask, tl.abs(k_f), 0.0), axis=0)
    amax = tl.maximum(amax, 1e-4)
    scale = amax / FP8_MAX
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    fp8_val = (k_f / scale).to(tl.float8e4nv)

    fp8_u8 = fp8_val.to(tl.uint8, bitcast=True)
    tl.store(Fp8_ws_ptr + fp8_base + d_offs, fp8_u8, mask=d_mask)

    scale_u32 = scale.to(tl.uint32, bitcast=True)
    tl.store(Fp8_ws_ptr + fp8_base + D + 0, ((scale_u32 >> 0) & 0xFF).to(tl.uint8))
    tl.store(Fp8_ws_ptr + fp8_base + D + 1, ((scale_u32 >> 8) & 0xFF).to(tl.uint8))
    tl.store(Fp8_ws_ptr + fp8_base + D + 2, ((scale_u32 >> 16) & 0xFF).to(tl.uint8))
    tl.store(Fp8_ws_ptr + fp8_base + D + 3, ((scale_u32 >> 24) & 0xFF).to(tl.uint8))


@triton.jit
def _indexer_tq_sync_block_vectorized(
    Packed_cache_ptr,
    Fp8_ws_ptr,
    Centroids_ptr,
    block_id,
    stride_packed_nb,
    stride_packed_bs,
    stride_ws_nb,
    stride_ws_bs,
    slot_offs,
    slot_mask,
    D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    """Sync up to BLOCK_KV slots in one physical page block (vectorized)."""
    d_offs = tl.arange(0, BLOCK_D)[:, None]
    n_offs = slot_offs[None, :]
    d_mask = d_offs < D
    full_mask = d_mask & slot_mask[None, :]

    packed_base = block_id * stride_packed_nb + n_offs * stride_packed_bs
    fp8_base = block_id * stride_ws_nb + n_offs * stride_ws_bs

    bit_off = d_offs * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    raw0 = tl.load(
        Packed_cache_ptr + packed_base + byte_idx, mask=full_mask, other=0
    ).to(tl.int32)
    raw1 = tl.load(
        Packed_cache_ptr + packed_base + byte_idx + 1, mask=full_mask, other=0
    ).to(tl.int32)
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask
    y_hat = _tq_mla_gather_centroids(
        Centroids_ptr, idx, full_mask, BLOCK_D, BLOCK_KV, MSE_BITS
    )
    n_lo = tl.load(
        Packed_cache_ptr + packed_base + MSE_BYTES, mask=slot_mask, other=0
    ).to(tl.uint16)
    n_hi = tl.load(
        Packed_cache_ptr + packed_base + MSE_BYTES + 1, mask=slot_mask, other=0
    ).to(tl.uint16)
    token_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    k_f = y_hat.to(tl.float32) * token_scale

    amax = tl.max(tl.where(d_mask, tl.abs(k_f), 0.0), axis=0)
    amax = tl.maximum(amax, 1e-4)
    scale = amax / FP8_MAX
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    fp8_val = (k_f / scale).to(tl.float8e4nv)
    fp8_u8 = fp8_val.to(tl.uint8, bitcast=True)
    tl.store(Fp8_ws_ptr + fp8_base + d_offs, fp8_u8, mask=full_mask)

    scale_u32 = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Fp8_ws_ptr + fp8_base + D + 0,
        ((scale_u32 >> 0) & 0xFF).to(tl.uint8),
        mask=slot_mask,
    )
    tl.store(
        Fp8_ws_ptr + fp8_base + D + 1,
        ((scale_u32 >> 8) & 0xFF).to(tl.uint8),
        mask=slot_mask,
    )
    tl.store(
        Fp8_ws_ptr + fp8_base + D + 2,
        ((scale_u32 >> 16) & 0xFF).to(tl.uint8),
        mask=slot_mask,
    )
    tl.store(
        Fp8_ws_ptr + fp8_base + D + 3,
        ((scale_u32 >> 24) & 0xFF).to(tl.uint8),
        mask=slot_mask,
    )


@triton.jit
def _indexer_tq_sync_decode_tile_kernel(
    Packed_cache_ptr,
    Fp8_ws_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_ptr,
    block_table_stride,
    stride_packed_nb,
    stride_packed_bs,
    stride_ws_nb,
    stride_ws_bs,
    block_size: tl.constexpr,
    D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    batch_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    seq_len = tl.load(Seq_lens_ptr + batch_id)
    tile_start = tile_id * TILE
    if tile_start >= seq_len:
        return

    n_pos = tile_start + tl.arange(0, TILE)
    pos_ok = n_pos < seq_len
    block_table_id = n_pos // block_size
    slot_in_block = n_pos % block_size
    block_id = tl.load(
        Block_table_ptr + batch_id * block_table_stride + block_table_id,
        mask=pos_ok,
        other=0,
    )

    d_offs = tl.arange(0, BLOCK_D)[:, None]
    tl.arange(0, TILE)[None, :]
    d_mask = d_offs < D
    full_mask = d_mask & pos_ok[None, :]
    bit_off = d_offs * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    packed_base = block_id * stride_packed_nb + slot_in_block * stride_packed_bs
    fp8_base = block_id * stride_ws_nb + slot_in_block * stride_ws_bs

    raw0 = tl.load(
        Packed_cache_ptr + packed_base + byte_idx, mask=full_mask, other=0
    ).to(tl.int32)
    raw1 = tl.load(
        Packed_cache_ptr + packed_base + byte_idx + 1, mask=full_mask, other=0
    ).to(tl.int32)
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask
    y_hat = _tq_mla_gather_centroids(
        Centroids_ptr, idx, full_mask, BLOCK_D, TILE, MSE_BITS
    )
    n_lo = tl.load(Packed_cache_ptr + packed_base + MSE_BYTES, mask=pos_ok, other=0).to(
        tl.uint16
    )
    n_hi = tl.load(
        Packed_cache_ptr + packed_base + MSE_BYTES + 1, mask=pos_ok, other=0
    ).to(tl.uint16)
    token_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    k_f = y_hat.to(tl.float32) * token_scale

    amax = tl.max(tl.where(d_mask, tl.abs(k_f), 0.0), axis=0)
    amax = tl.maximum(amax, 1e-4)
    scale = amax / FP8_MAX
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    fp8_val = (k_f / scale).to(tl.float8e4nv)
    fp8_u8 = fp8_val.to(tl.uint8, bitcast=True)
    tl.store(Fp8_ws_ptr + fp8_base + d_offs, fp8_u8, mask=full_mask)

    scale_u32 = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Fp8_ws_ptr + fp8_base + D + 0,
        ((scale_u32 >> 0) & 0xFF).to(tl.uint8),
        mask=pos_ok,
    )
    tl.store(
        Fp8_ws_ptr + fp8_base + D + 1,
        ((scale_u32 >> 8) & 0xFF).to(tl.uint8),
        mask=pos_ok,
    )
    tl.store(
        Fp8_ws_ptr + fp8_base + D + 2,
        ((scale_u32 >> 16) & 0xFF).to(tl.uint8),
        mask=pos_ok,
    )
    tl.store(
        Fp8_ws_ptr + fp8_base + D + 3,
        ((scale_u32 >> 24) & 0xFF).to(tl.uint8),
        mask=pos_ok,
    )


@triton.jit
def _indexer_tq_sync_decode_sm_kernel(
    Packed_cache_ptr,
    Fp8_ws_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_ptr,
    Schedule_ptr,
    block_table_stride,
    stride_packed_nb,
    stride_packed_bs,
    stride_ws_nb,
    stride_ws_bs,
    block_size: tl.constexpr,
    D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    NUM_BLOCKS_PER_SPLIT: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    """Sync KV slots assigned to this SM (same schedule_meta as DeepGEMM)."""
    sm_idx = tl.program_id(0)
    start_q = tl.load(Schedule_ptr + sm_idx * 2)
    start_kv_s = tl.load(Schedule_ptr + sm_idx * 2 + 1)
    end_q = tl.load(Schedule_ptr + (sm_idx + 1) * 2)
    end_kv_s = tl.load(Schedule_ptr + (sm_idx + 1) * 2 + 1)

    cur_q = start_q
    cur_kv = start_kv_s * NUM_BLOCKS_PER_SPLIT
    end_kv = end_kv_s * NUM_BLOCKS_PER_SPLIT

    slot_offs = tl.arange(0, BLOCK_KV)

    while cur_q < end_q or (cur_q == end_q and cur_kv < end_kv):
        q_row = cur_q
        seq_len = tl.load(Seq_lens_ptr + q_row)
        num_kv = (seq_len + BLOCK_KV - 1) // BLOCK_KV

        for kv_b in tl.static_range(NUM_BLOCKS_PER_SPLIT):
            kv_idx = cur_kv + kv_b
            if kv_idx < num_kv:
                block_id = tl.load(
                    Block_table_ptr + q_row * block_table_stride + kv_idx
                )
                token_base = kv_idx * BLOCK_KV
                slot_mask = slot_offs + token_base < seq_len
                _indexer_tq_sync_block_vectorized(
                    Packed_cache_ptr,
                    Fp8_ws_ptr,
                    Centroids_ptr,
                    block_id,
                    stride_packed_nb,
                    stride_packed_bs,
                    stride_ws_nb,
                    stride_ws_bs,
                    slot_offs,
                    slot_mask,
                    D,
                    MSE_BITS,
                    MSE_BYTES,
                    BLOCK_D,
                    BLOCK_KV,
                    FP8_MAX,
                    USE_UE8M0,
                )

        cur_kv += NUM_BLOCKS_PER_SPLIT
        if cur_kv >= num_kv:
            cur_kv = 0
            cur_q += 1


def sync_fp8_workspace_for_decode(
    packed_cache: torch.Tensor,
    fp8_workspace: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale_fmt: str | None,
    max_model_len: int | None = None,
    schedule_metadata: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fill FP8-shaped workspace slots used by decode paged DeepGEMM."""
    _, block_size, _ = packed_cache.shape
    if seq_lens.dim() == 2:
        seq_lens_1d = seq_lens.max(dim=-1).values.to(torch.int32)
    else:
        seq_lens_1d = seq_lens.to(torch.int32)
    batch_size = seq_lens_1d.shape[0]
    if batch_size == 0:
        return fp8_workspace

    buf = _get_indexer_tq_buffers(packed_cache.device)
    block_d = triton.next_power_of_2(INDEXER_HEAD_DIM)
    use_ue8m0 = scale_fmt == "ue8m0"
    if max_model_len is None:
        max_seq_for_mode = block_table.shape[1] * block_size
    else:
        max_seq_for_mode = max_model_len
    mode = resolve_indexer_tq_sync_mode(max_seq_for_mode)
    _log_indexer_tq_sync_config_once(mode, max_seq_for_mode)
    blocks_per_split = indexer_tq_sync_blocks_per_split()

    stride_args = (
        block_table.stride(0),
        packed_cache.stride(0),
        packed_cache.stride(1),
        fp8_workspace.stride(0),
        fp8_workspace.stride(1),
    )
    ptr_args = (
        packed_cache,
        fp8_workspace,
        block_table,
        seq_lens_1d,
        buf["centroids_bf16"],
    )

    if mode == "sm":
        if schedule_metadata is None:
            from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

            num_sms = torch.cuda.get_device_properties(
                packed_cache.device
            ).multi_processor_count
            sl_for_meta = seq_lens if seq_lens.dim() == 2 else seq_lens.unsqueeze(-1)
            schedule_metadata = get_paged_mqa_logits_metadata(
                sl_for_meta, block_size, num_sms
            )
        num_sms = schedule_metadata.shape[0] - 1
        _indexer_tq_sync_decode_sm_kernel[(num_sms,)](
            *ptr_args,
            schedule_metadata,
            *stride_args,
            block_size,
            INDEXER_HEAD_DIM,
            INDEXER_MSE_BITS,
            INDEXER_MSE_BYTES,
            block_d,
            INDEXER_DG_SYNC_BLOCK_KV,
            blocks_per_split,
            448.0,
            use_ue8m0,
            num_warps=4,
        )
    elif mode == "tile":
        if max_model_len is None:
            max_seq = block_table.shape[1] * block_size
        else:
            max_seq = max_model_len
        grid_n = (max_seq + INDEXER_DECODE_TILE - 1) // INDEXER_DECODE_TILE
        _indexer_tq_sync_decode_tile_kernel[(batch_size, grid_n)](
            *ptr_args,
            *stride_args,
            block_size,
            INDEXER_HEAD_DIM,
            INDEXER_MSE_BITS,
            INDEXER_MSE_BYTES,
            block_d,
            INDEXER_DECODE_TILE,
            448.0,
            use_ue8m0,
            num_warps=4,
        )
    elif mode == "chunk":
        if max_model_len is None:
            max_seq = block_table.shape[1] * block_size
        else:
            max_seq = max_model_len
        if max_seq == 0:
            return fp8_workspace
        fixed_grid = indexer_sync_fixed_grid()
        num_chunks = indexer_sync_num_chunks(max_seq)
        kernel_tail = (
            block_size,
            INDEXER_HEAD_DIM,
            INDEXER_MSE_BITS,
            INDEXER_MSE_BYTES,
            INDEXER_PACKED_BYTES,
            INDEXER_FP8_SLOT_BYTES,
            block_d,
            448.0,
            use_ue8m0,
        )
        for ci in range(num_chunks):
            kv_offset = ci * fixed_grid
            remaining = max_seq - kv_offset
            if remaining <= 0:
                break
            grid_y = min(fixed_grid, remaining)
            _indexer_tq_sync_decode_kernel[(batch_size, grid_y)](
                packed_cache,
                fp8_workspace,
                block_table,
                seq_lens_1d,
                buf["centroids_bf16"],
                block_table.stride(0),
                packed_cache.stride(0),
                packed_cache.stride(1),
                fp8_workspace.stride(0),
                fp8_workspace.stride(1),
                *kernel_tail,
                KV_OFFSET=kv_offset,
            )
    elif mode == "fixed":
        if max_model_len is None:
            max_seq = block_table.shape[1] * block_size
        else:
            max_seq = max_model_len
        if max_seq == 0:
            return fp8_workspace
        fixed_grid = indexer_sync_fixed_grid()
        inner_tile = indexer_sync_inner_tile(max_seq)
        _indexer_tq_sync_decode_tile_kernel[(batch_size, fixed_grid)](
            *ptr_args,
            *stride_args,
            block_size,
            INDEXER_HEAD_DIM,
            INDEXER_MSE_BITS,
            INDEXER_MSE_BYTES,
            block_d,
            inner_tile,
            448.0,
            use_ue8m0,
            num_warps=4,
        )
    else:
        if max_model_len is None:
            max_seq = block_table.shape[1] * block_size
        else:
            max_seq = max_model_len
        if max_seq == 0:
            return fp8_workspace
        _indexer_tq_sync_decode_kernel[(batch_size, max_seq)](
            packed_cache,
            fp8_workspace,
            block_table,
            seq_lens_1d,
            buf["centroids_bf16"],
            block_table.stride(0),
            packed_cache.stride(0),
            packed_cache.stride(1),
            fp8_workspace.stride(0),
            fp8_workspace.stride(1),
            block_size,
            INDEXER_HEAD_DIM,
            INDEXER_MSE_BITS,
            INDEXER_MSE_BYTES,
            INDEXER_PACKED_BYTES,
            INDEXER_FP8_SLOT_BYTES,
            block_d,
            448.0,
            use_ue8m0,
            KV_OFFSET=0,
        )
    return fp8_workspace


@triton.jit
def _tq4_paged_mqa_logits_kernel(
    Q_ptr,
    Weights_ptr,
    Packed_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Logits_ptr,
    Centroids_ptr,
    q_stride_b,
    q_stride_h,
    weights_stride_b,
    logits_stride,
    block_table_stride,
    stride_packed_nb,
    stride_packed_bs,
    block_size: tl.constexpr,
    D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_id = tl.program_id(0)
    pos = tl.program_id(1)
    seq_len = tl.load(Seq_lens_ptr + batch_id)
    if pos >= seq_len:
        return

    block_table_id = pos // block_size
    slot_in_block = pos % block_size
    block_id = tl.load(Block_table_ptr + batch_id * block_table_stride + block_table_id)

    packed_base = block_id * stride_packed_nb + slot_in_block * stride_packed_bs

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    bit_off = d_offs * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    raw0 = tl.load(Packed_cache_ptr + packed_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    raw1 = tl.load(
        Packed_cache_ptr + packed_base + byte_idx + 1, mask=d_mask, other=0
    ).to(tl.int32)
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask

    y_hat = _tq_mla_gather_centroids_1d(Centroids_ptr, idx, d_mask, BLOCK_D, MSE_BITS)
    n_lo = tl.load(Packed_cache_ptr + packed_base + MSE_BYTES).to(tl.uint16)
    n_hi = tl.load(Packed_cache_ptr + packed_base + MSE_BYTES + 1).to(tl.uint16)
    token_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    k_f = y_hat.to(tl.float32) * token_scale

    acc = tl.zeros((), dtype=tl.float32)
    for h in tl.static_range(NUM_HEADS):
        q_base = Q_ptr + batch_id * q_stride_b + h * q_stride_h
        q_val = tl.load(q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        dot = tl.sum(q_val * k_f, axis=0)
        w = tl.load(Weights_ptr + batch_id * weights_stride_b + h).to(tl.float32)
        acc += tl.maximum(dot, 0.0) * w

    tl.store(Logits_ptr + batch_id * logits_stride + pos, acc)


def tq4_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    packed_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Fused TQ4 paged gather + dequant + MQA logits (next_n=1 decode)."""
    batch_size, num_heads, head_dim = q_fp8.shape
    assert head_dim == INDEXER_HEAD_DIM
    if batch_size == 0:
        return torch.empty((0, max_model_len), device=q_fp8.device, dtype=torch.float32)

    if seq_lens.dim() > 1:
        seq_lens_1d = seq_lens.max(dim=-1).values.to(torch.int32)
    else:
        seq_lens_1d = seq_lens.to(torch.int32)

    # Reuse persistent workspace so FULL CUDA graph replay and Python
    # consumers (persistent_topk) see the same buffer address each step.
    logits = _get_scores_workspace(q_fp8.device, batch_size, max_model_len)
    logits.fill_(float("-inf"))
    _launch_tq4_paged_mqa_scores(
        q_fp8,
        weights,
        packed_cache,
        block_table,
        seq_lens_1d,
        logits,
        max_model_len,
    )
    return logits


def warmup_indexer_tq_store(
    device: torch.device | int,
    block_size: int = 64,
) -> None:
    """Compile fused store Triton kernel before CUDA graph capture."""
    if not is_indexer_tq_4bit_enabled():
        return
    if not torch.cuda.is_available():
        return
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    key = str(device)
    if key in _INDEXER_TQ_STORE_WARMED:
        return

    _get_indexer_tq_buffers(device)
    k = torch.zeros(1, INDEXER_HEAD_DIM, device=device, dtype=_BF16)
    kv_cache = torch.zeros(
        1,
        block_size,
        INDEXER_PACKED_BYTES,
        device=device,
        dtype=torch.uint8,
    )
    slot_mapping = torch.zeros(1, device=device, dtype=torch.int32)
    for _ in range(8):
        indexer_tq_store_and_cache_triton(k, kv_cache, slot_mapping)
    torch.accelerator.synchronize(device)
    _INDEXER_TQ_STORE_WARMED.add(key)
    logger.info_once("Indexer TQ4 fused store kernel warmed up.")


def warmup_indexer_tq_kernels(
    device: torch.device | int,
    max_model_len: int,
    num_heads: int,
    block_size: int = 64,
) -> None:
    """Warm up Indexer TQ4 Triton kernels for CUDA graph capture."""
    warmup_indexer_tq_store(device, block_size)
    warmup_indexer_tq_fused_decode(device, max_model_len, num_heads, block_size)


def warmup_indexer_tq_fused_decode(
    device: torch.device | int,
    max_model_len: int,
    num_heads: int,
    block_size: int = 64,
) -> None:
    """Compile fused decode Triton kernel before CUDA graph capture."""
    if not is_indexer_tq_4bit_enabled() or not use_indexer_tq_fused_decode():
        return
    if not torch.cuda.is_available():
        return
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    key = str(device)
    if key in _INDEXER_TQ_FUSED_WARMED:
        return

    _get_indexer_tq_buffers(device)
    batch_size = 1
    q_fp8 = torch.zeros(
        batch_size,
        num_heads,
        INDEXER_HEAD_DIM,
        device=device,
        dtype=_FP8_DTYPE,
    )
    weights = torch.zeros(batch_size, num_heads, device=device, dtype=torch.float32)
    packed_cache = torch.zeros(
        1,
        block_size,
        INDEXER_PACKED_BYTES,
        device=device,
        dtype=torch.uint8,
    )
    num_blocks = max(1, (max_model_len + block_size - 1) // block_size)
    block_table = torch.zeros(
        batch_size,
        num_blocks,
        device=device,
        dtype=torch.int32,
    )
    seq_lens = torch.ones(batch_size, device=device, dtype=torch.int32)
    tq4_paged_mqa_logits_triton(
        q_fp8,
        weights,
        packed_cache,
        block_table,
        seq_lens,
        max_model_len,
    )
    torch.accelerator.synchronize(device)
    _INDEXER_TQ_FUSED_WARMED.add(key)
    logger.info_once("Indexer TQ4 fused decode logits kernel warmed up.")
