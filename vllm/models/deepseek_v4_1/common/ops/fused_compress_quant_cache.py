# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 state saving/compression and independently schedulable cache insertion."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX950
else:
    _ON_GFX950 = False

# The ring and raw rows a ratio-2 request program pools are not adjacent, so
# reading them as one tile means joining two pointers. ROCm's Triton fails to
# legalize `tt.join` on pointers, so there the two rows are loaded separately.
_JOIN_ROW_PTRS = not current_platform.is_rocm()


def fused_save_compress_norm(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    latent_out: torch.Tensor,
) -> None:
    """Pool each closed group into a normalized BF16 latent; save FP32 states.

    The latent feeds the main-cache insert and the indexer K path, which the
    attention layer schedules on separate streams.

    Ratio 2 keeps one ring block per request holding the open group's rows:
    position ``p`` lives in row ``p % capacity`` and ``slot_mapping`` encodes
    ``block * capacity + p % capacity``. The grid has one program per request
    followed by one per pair of packed tokens. A request program handles the
    group that the chunk's first token closes with its predecessor's ring row,
    then stores the chunk's last ``capacity`` rows to the ring; because the
    same program does both, ring reads and writes never race. A pair program
    handles the group that ends inside its pair, reading both rows from the
    raw input. Ratio 1 has no ring and one program per token; ``slot_mapping``
    then only marks valid tokens.

    Args:
        kv_score: FP32 [tokens, 512] for CR1, [tokens, 1024] for CR2.
        positions: Absolute positions of the packed request tokens.
        state_cache: Ring FP32 [blocks, capacity, 1024] KV/score states (CR2).
        slot_mapping: Ring slots (CR2) or main-cache slots (CR1).
        query_start_loc: [num_reqs + 1] token offsets of each request's chunk.
        token_to_req_indices: Request indices for the packed token rows.
        rms_norm_weight: BF16 [512] normalization weight.
        rms_norm_eps: RMSNorm epsilon.
        compress_ratio: Group size, either 1 or 2.
        latent_out: BF16 [tokens, 512], written only at valid group boundaries.
    """
    assert compress_ratio in (1, 2)
    assert kv_score.dtype == torch.float32
    assert kv_score.shape[1] == 512 * compress_ratio and kv_score.stride(1) == 1
    assert latent_out.shape == (kv_score.shape[0], 512)
    assert latent_out.is_contiguous() and latent_out.dtype == torch.bfloat16
    assert positions.is_contiguous() and slot_mapping.is_contiguous()
    # Rows stay 64-byte aligned, as the kernel's tl.multiple_of hint promises.
    assert kv_score.stride(0) % 16 == 0
    if compress_ratio == 2:
        assert state_cache is not None and query_start_loc is not None
        assert token_to_req_indices is not None
        assert query_start_loc.is_contiguous()
        assert token_to_req_indices.is_contiguous()
        assert state_cache.dtype == torch.float32
        assert state_cache.shape[2] == 1024 and state_cache.stride(2) == 1
        assert state_cache.stride(1) % 16 == 0
        state_stride, state_row_stride, state_block = (
            state_cache.stride(0),
            state_cache.stride(1),
            state_cache.shape[1],
        )
        num_reqs = query_start_loc.numel() - 1
    else:
        state_cache = query_start_loc = token_to_req_indices = None
        state_stride = state_row_stride = state_block = 1
        num_reqs = 0
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(kv_score.shape[0], positions.numel())
    if num_tokens == 0:
        return
    grid = num_reqs + triton.cdiv(num_tokens, compress_ratio)
    _fused_save_compress_norm_kernel[(grid,)](
        kv_score,
        positions,
        state_cache,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        rms_norm_weight,
        latent_out,
        num_tokens,
        num_reqs,
        RAW_STRIDE=kv_score.stride(0),
        STATE_STRIDE=state_stride,
        STATE_ROW_STRIDE=state_row_stride,
        STATE_BLOCK=state_block,
        COMPRESS_RATIO=compress_ratio,
        EPS=rms_norm_eps,
        JOIN_ROW_PTRS=_JOIN_ROW_PTRS,
        num_warps=4,
        **({"launch_pdl": False} if current_platform.is_cuda() else {}),
    )


@triton.jit(do_not_specialize=["num_tokens", "num_reqs"])
def _fused_save_compress_norm_kernel(
    raw,
    positions,
    state,
    state_slots,
    query_start_loc,
    req_ids,
    norm_weight,
    latent,
    num_tokens,
    num_reqs,
    RAW_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    EPS: tl.constexpr,
    JOIN_ROW_PTRS: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, 512)

    if COMPRESS_RATIO == 2:  # noqa: SIM102 (constexpr branch, then runtime)
        if pid < num_reqs:
            # Request program: the group closed by the chunk's first token,
            # then the ring tail store.
            start = tl.load(query_start_loc + pid)
            end = tl.load(query_start_loc + pid + 1)
            if start >= end:
                return
            state_slot = tl.load(state_slots + start)
            if state_slot < 0:
                return
            position = tl.load(positions + start)
            if (position + 1) % 2 == 0:
                ring = state + (state_slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                prev = ring + ((position - 1) % STATE_BLOCK) * STATE_ROW_STRIDE
                current = raw + start.to(tl.int64) * RAW_STRIDE
                if JOIN_ROW_PTRS:
                    # Joining pointers hides row alignment from Triton; restate it.
                    rows = tl.multiple_of(tl.join(prev, current)[:, None], (16, 16))
                    kv = tl.load(rows + d[None, :])
                    score = tl.load(rows + 512 + d[None, :])
                    # Every lane has read the ring before the tail store below
                    # writes it.
                    tl.debug_barrier()
                    pooled = tl.sum(kv * tl.softmax(score, 0), 0)
                else:
                    kv_prev = tl.load(prev + d)
                    score_prev = tl.load(prev + 512 + d)
                    kv_current = tl.load(current + d)
                    score_current = tl.load(current + 512 + d)
                    # Every lane has read the ring before the tail store below
                    # writes it.
                    tl.debug_barrier()
                    peak = tl.maximum(score_prev, score_current)
                    weight_prev = tl.exp(score_prev - peak)
                    weight_current = tl.exp(score_current - peak)
                    pooled = (kv_prev * weight_prev + kv_current * weight_current) / (
                        weight_prev + weight_current
                    )
                _store_latent(pooled, start, norm_weight, latent, EPS)
            num_rows = tl.minimum(end - start, STATE_BLOCK)
            for k in tl.range(0, num_rows):
                token = end - num_rows + k
                slot = tl.load(state_slots + token)
                if slot >= 0:
                    row = (
                        state
                        + (slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                        + (slot % STATE_BLOCK) * STATE_ROW_STRIDE
                    )
                    src = raw + token.to(tl.int64) * RAW_STRIDE
                    tl.store(row + d, tl.load(src + d))
                    tl.store(row + 512 + d, tl.load(src + 512 + d))
            return

    # Group program: one token (ratio 1) or the token of the pair that ends
    # a group (ratio 2). A chunk's first token belongs to its request program.
    t = (pid - num_reqs) * COMPRESS_RATIO
    if COMPRESS_RATIO == 2:
        t += (tl.load(positions + t) + 1) % 2
    state_slot = tl.load(state_slots + t, t < num_tokens, other=-1)
    if state_slot < 0:
        return
    if COMPRESS_RATIO == 1:
        pooled = tl.load(raw + t.to(tl.int64) * RAW_STRIDE + d)
    else:
        if t == tl.load(query_start_loc + tl.load(req_ids + t)):
            return
        rows = raw + (t.to(tl.int64) - 1 + tl.arange(0, 2)[:, None]) * RAW_STRIDE
        kv = tl.load(rows + d[None, :])
        score = tl.load(rows + 512 + d[None, :])
        pooled = tl.sum(kv * tl.softmax(score, 0), 0)
    _store_latent(pooled, t, norm_weight, latent, EPS)


@triton.jit
def _store_latent(pooled, t, norm_weight, latent, EPS: tl.constexpr):
    d = tl.arange(0, 512)
    weight = tl.load(norm_weight + d).to(tl.float32)
    variance = tl.sum(pooled * pooled, 0) / 512
    normed = pooled * tl.rsqrt(variance + EPS) * weight
    tl.store(latent + t.to(tl.int64) * 512 + d, normed.to(tl.bfloat16))


def rope_quant_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    """Apply GPT-J RoPE and publish a latent to the compressed KV cache.

    The BF16 latent supplies both NoPE quantization and RoPE input. It is read
    only for valid slots at group boundaries. The cache dtype selects the
    layout: ``uint8`` is the fp8_ds_mla paged layout (576 value bytes and eight
    segregated UE8M0 scale bytes per token, including one zero padding scale);
    ``bfloat16`` and ``float8_e4m3fn`` are the plain [448 NoPE | 64 RoPE] rows
    read by FlashInfer, the latter scaled by the per-tensor ``fp8_scale``.
    """
    assert compress_ratio in (1, 2)
    assert latent.shape[1] == 512 and latent.dtype == torch.bfloat16
    assert latent.is_contiguous()
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(latent.shape[0], positions.numel())
    if num_tokens == 0:
        return
    launch_kwargs = {"launch_pdl": False} if current_platform.is_cuda() else {}
    if kv_cache.dtype == torch.uint8:
        assert kv_cache.shape[-1] == 584
        _rope_quant_insert_kernel[(num_tokens,)](
            latent,
            positions,
            cos_sin_cache,
            kv_cache,
            slot_mapping,
            COS_STRIDE=cos_sin_cache.stride(0),
            CACHE_STRIDE=kv_cache.stride(0),
            CACHE_BLOCK=kv_cache.shape[1],
            COMPRESS_RATIO=compress_ratio,
            SANITIZE_CACHE_NANS=_ON_GFX950,
            num_warps=4,
            **launch_kwargs,
        )
        return

    assert kv_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    assert kv_cache.shape[-1] == 512 and kv_cache.stride(-1) == 1
    store_fp8 = kv_cache.dtype == torch.float8_e4m3fn
    if store_fp8:
        assert fp8_scale is not None and fp8_scale.numel() == 1
        assert fp8_scale.dtype == torch.float32
    _rope_plain_insert_kernel[(num_tokens,)](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        fp8_scale if store_fp8 else None,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        ROW_STRIDE=kv_cache.stride(1),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        STORE_FP8=store_fp8,
        num_warps=4,
        **launch_kwargs,
    )


@triton.jit
def _rope_quant_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot % CACHE_BLOCK) * 576
    scales = page + CACHE_BLOCK * 576 + (slot % CACHE_BLOCK) * 8

    quant = tl.reshape(normed, (8, 64))
    amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
    scaled = quant * tl.reshape(tl.exp2(-exponent), (8, 1))
    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    packed = tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,))
    tl.store(values + d, packed, d < 448)
    s = tl.arange(0, 8)
    max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), max_encoded)
    tl.store(scales + s, encoded.to(tl.uint8), s < 7)
    tl.store(scales + 7, tl.full((), 0, tl.uint8))

    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
    if SANITIZE_CACHE_NANS:
        rotated = tl.where(rotated == rotated, rotated, 0.0)
    rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + d - 448, rotated.to(tl.bfloat16), d >= 448)


@triton.jit
def _rope_plain_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    fp8_scale,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STORE_FP8: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)

    # NoPE pairs load (cos, sin) = (1, 0), so the rotation is the identity there.
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    row = tl.interleave(even * c - odd * s, odd * c + even * s).to(tl.bfloat16)

    dst = (
        cache
        + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
        + (slot % CACHE_BLOCK) * ROW_STRIDE
    )
    if STORE_FP8:
        scaled = row.to(tl.float32) * (1.0 / tl.load(fp8_scale))
        tl.store(dst + d, tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv))
    else:
        tl.store(dst + d, row)
