# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM103 ReplaySSM with an FP16 recurrence checkpoint and quantized reads.

The native kernels support one token, 16 replay entries, 64 head dimensions,
128 state dimensions, and 8 or 16 heads per group. Slot zero is reserved for padding.
Physical cache pages may have padding between slots.
"""

import math
from collections.abc import Sequence

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _software_sr_half(value, random):
    magnitude = tl.minimum(tl.abs(value), 65504.0)
    lower = magnitude.to(tl.float16, fp_downcast_rounding="rtz")
    upper = (lower.to(tl.uint16, bitcast=True) + 1).to(tl.float16, bitcast=True)
    probability = (magnitude - lower.to(tl.float32)) / (
        upper.to(tl.float32) - lower.to(tl.float32)
    )
    result = tl.where(
        (random & 8191).to(tl.float32) / 8192.0 < probability, upper, lower
    ).to(tl.uint16, bitcast=True)
    sign = (value.to(tl.uint32, bitcast=True) >> 16) & 32768
    return (result | sign).to(tl.uint16)


@triton.jit
def _encode_e2m1(x):
    a = tl.abs(x)
    q = tl.where(a > 0.25, 1, 0)
    q = tl.where(a >= 0.75, 2, q)
    q = tl.where(a > 1.25, 3, q)
    q = tl.where(a >= 1.75, 4, q)
    q = tl.where(a > 2.5, 5, q)
    q = tl.where(a >= 3.5, 6, q)
    q = tl.where(a > 5.0, 7, q)
    sign = x.to(tl.int32, bitcast=True) < 0
    return (q | tl.where(sign, 8, 0)).to(tl.uint8)


@triton.jit
def _initialize_rows(
    src,
    ids,
    anchor,
    read,
    scales,
    group_max,
    global_scale,
    meta,
    parity,
    SRC_STRIDE: tl.constexpr,
    ANCHOR_STRIDE: tl.constexpr,
    READ_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    META_STRIDE: tl.constexpr,
    PARITY_STRIDE: tl.constexpr,
    MAX_STRIDE: tl.constexpr,
    GLOBAL_STRIDE: tl.constexpr,
    SLOTS: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    NVFP4: tl.constexpr,
    SEED: tl.constexpr,
    SOFTWARE_SR: tl.constexpr,
):
    batch = tl.program_id(0)
    row = tl.program_id(1)
    slot = tl.load(ids + batch).to(tl.int64)
    if (slot > 0) & (slot < SLOTS):
        pair = tl.arange(0, 64)
        even = tl.load(src + batch * SRC_STRIDE + row * 128 + pair * 2)
        odd = tl.load(src + batch * SRC_STRIDE + row * 128 + pair * 2 + 1)
        bits = tl.randint(SEED, slot * HEADS * 4096 + row * 64 + pair, n_rounds=7)
        if SOFTWARE_SR:
            packed = _software_sr_half(even, bits).to(tl.uint32) | (
                _software_sr_half(odd, bits >> 16).to(tl.uint32) << 16
            )
        else:
            packed = tl.inline_asm_elementwise(
                "cvt.rs.satfinite.f16x2.f32 $0, $2, $1, $3;",
                constraints="=r,f,f,r",
                args=[even, odd, bits],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
        tl.store(anchor + slot * ANCHOR_STRIDE + row * 64 + pair, packed)
        if NVFP4:
            # Eight blocks per row. The payload and scales use native layouts.
            gscale = tl.load(global_scale + batch * GLOBAL_STRIDE)
            e = tl.reshape(even, (8, 8))
            o = tl.reshape(odd, (8, 8))
            amax = tl.max(tl.maximum(tl.abs(e), tl.abs(o)), axis=1)
            sf = (amax / (6.0 * gscale)).to(tl.float8e4nv)
            sf32 = sf.to(tl.float32)
            inv = tl.where(sf32 > 0, 1.0 / sf32, 1.0) / gscale
            q = _encode_e2m1(e * inv[:, None]) | (_encode_e2m1(o * inv[:, None]) << 4)
            tl.store(read + slot * READ_STRIDE + row * 64 + pair, tl.reshape(q, (64,)))
            block = tl.arange(0, 8)
            pair_head = row // 128
            pair_row = row % 128
            off = (
                pair_head * 1024
                + (block // 4) * 512
                + (pair_row % 32) * 16
                + (pair_row // 32) * 4
                + block % 4
            )
            tl.store(scales + slot * SCALE_STRIDE + off, sf.to(tl.uint8, bitcast=True))
            if row % ((HEADS // GROUPS) * 64) == 0:
                group = row // ((HEADS // GROUPS) * 64)
                maximum = tl.load(group_max + batch * MAX_STRIDE + group)
                tl.store(meta + slot * META_STRIDE + group, maximum)
                tl.store(meta + slot * META_STRIDE + GROUPS + group, maximum)
                tl.store(meta + slot * META_STRIDE + 2 * GROUPS + group, gscale)
                tl.store(parity + slot * PARITY_STRIDE + group, 0)
        else:
            amax = tl.max(tl.maximum(tl.abs(even), tl.abs(odd)), axis=0)
            scale = tl.where(amax > 0, amax / 448.0, 1.0)
            tl.store(scales + slot * SCALE_STRIDE + row, scale)
            tl.store(
                read + slot * READ_STRIDE + row * 128 + pair * 2,
                (even / scale).to(tl.float8e4nv),
            )
            tl.store(
                read + slot * READ_STRIDE + row * 128 + pair * 2 + 1,
                (odd / scale).to(tl.float8e4nv),
            )


@triton.jit
def _clear_slots(
    ids,
    x,
    dt,
    b,
    event,
    XS: tl.constexpr,
    DS: tl.constexpr,
    BS: tl.constexpr,
    ES: tl.constexpr,
    XN: tl.constexpr,
    DN: tl.constexpr,
    BN: tl.constexpr,
    EVENT_WIDTH: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.load(ids + tl.program_id(0)).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (slot > 0) & (slot < SLOTS)
    tl.store(x + slot * XS + i, 0, valid & (i < XN))
    tl.store(dt + slot * DS + i, 0, valid & (i < DN))
    tl.store(b + slot * BS + i, 0, valid & (i < BN))
    if tl.program_id(1) == 0:
        tl.store(event + slot * ES, 0, valid)
        if EVENT_WIDTH == 2:
            tl.store(event + slot * ES + 1, 0, valid)


@triton.jit
def _prepare_decode(
    ids,
    event,
    native_ids,
    tags,
    positions,
    flush,
    out,
    SLOTS: tl.constexpr,
    EVENT_STRIDE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    slot = tl.load(ids + batch).to(tl.int64)
    valid = (slot > 0) & (slot < SLOTS)
    count = tl.load(event + slot * EVENT_STRIDE, valid, 0)
    tl.store(native_ids + batch, tl.where(valid, slot, 0).to(tl.int32))
    tl.store(tags + batch, count)
    tl.store(positions + batch, (count & 15).to(tl.int32))
    tl.store(flush + batch, ((count & 15) == 15) & valid)
    tl.store(event + slot * EVENT_STRIDE, count + 1, valid)
    for start in range(tl.cdiv(OUTPUT_SIZE, BLOCK)):
        offset = start * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + batch * OUTPUT_SIZE + offset, 0, offset < OUTPUT_SIZE)


@triton.jit
def _prepare_pending(
    ids,
    accepted,
    cu,
    event,
    meta,
    parity,
    native_ids,
    control,
    SLOTS: tl.constexpr,
    ES: tl.constexpr,
    MS: tl.constexpr,
    PS: tl.constexpr,
    GROUPS: tl.constexpr,
    G_BLOCK: tl.constexpr,
    NVFP4: tl.constexpr,
):
    batch = tl.program_id(0)
    slot = tl.load(ids + batch).to(tl.int64)
    valid = (slot > 0) & (slot < SLOTS)
    total = tl.load(event + slot * ES, valid, 0)
    pending = tl.load(event + slot * ES + 1, valid, 0)
    count = tl.minimum(tl.maximum(tl.load(accepted + batch), 0), pending)
    count = tl.where(valid, count, 0)
    old = total % 16
    crossed = old + count >= 16
    remain = (old + count) % 16
    tl.store(native_ids + batch, tl.where(valid, slot, 0).to(tl.int32))
    tl.store(control + batch * 5, old)
    tl.store(control + batch * 5 + 1, count)
    tl.store(control + batch * 5 + 2, crossed)
    tl.store(control + batch * 5 + 3, remain)
    tl.store(control + batch * 5 + 4, total + count)
    current = tl.load(cu + batch + 1) - tl.load(cu + batch)
    tl.store(event + slot * ES, total + count, valid)
    tl.store(event + slot * ES + 1, current, valid)
    if NVFP4:
        group = tl.arange(0, G_BLOCK)
        active = valid & crossed & (group < GROUPS)
        plane = tl.load(parity + slot * PS + group, active, 0)
        previous_max = tl.load(meta + slot * MS + plane * GROUPS + group, active, 0)
        global_scale = tl.max(previous_max, 0) / (6.0 * 448.0)
        global_scale = tl.where(global_scale > 0.0, global_scale, 1.0)
        target = plane ^ 1
        tl.store(meta + slot * MS + target * GROUPS + group, 0.0, active)
        tl.store(meta + slot * MS + 2 * GROUPS + group, global_scale, active)
        tl.store(parity + slot * PS + group, target, active)


@triton.jit
def _stage_pending(
    ids,
    cu,
    control,
    x,
    dt,
    b,
    bias,
    xc,
    dc,
    bc,
    out,
    XS: tl.constexpr,
    DS: tl.constexpr,
    BS: tl.constexpr,
    X_STRIDES: tl.constexpr,
    DT_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    LENGTH: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(ids + batch).to(tl.int64)
    begin = tl.load(cu + batch)
    count = tl.load(cu + batch + 1) - begin
    old = tl.load(control + batch * 5)
    accepted = tl.load(control + batch * 5 + 1)
    crossed = tl.load(control + batch * 5 + 2) != 0
    t = tl.arange(0, BLOCK_T)
    dim = tl.arange(0, 128)
    if slot > 0:
        # Load the accepted pending tail before replacing candidate entries.
        old_x = tl.load(
            xc + slot * XS + head * LENGTH * 64 + (16 + t[:, None]) * 64 + dim[None, :],
            (t[:, None] < accepted) & (dim[None, :] < 64),
            0,
        )
        old_dt = tl.load(dc + slot * DS + head * LENGTH + 16 + t, t < accepted, 0)
        dst = tl.where(crossed, t - (16 - old), old + t)
        retain = (t < accepted) & (dst >= 0) & (dst < 16)
        tl.store(
            xc + slot * XS + head * LENGTH * 64 + dst[:, None] * 64 + dim[None, :],
            old_x,
            retain[:, None] & (dim[None, :] < 64),
        )
        tl.store(dc + slot * DS + head * LENGTH + dst, old_dt, retain)
        new_x = tl.load(
            x
            + (begin + t[:, None]) * X_STRIDES[0]
            + head * X_STRIDES[1]
            + dim[None, :] * X_STRIDES[2],
            (t[:, None] < count) & (dim[None, :] < 64),
            0,
        )
        new_dt = tl.load(
            dt + (begin + t) * DT_STRIDES[0] + head * DT_STRIDES[1], t < count, 0
        ).to(tl.float32) + tl.load(bias + head * BIAS_STRIDE).to(tl.float32)
        new_dt = tl.where(new_dt <= 20.0, tl.log(1.0 + tl.exp(new_dt)), new_dt)
        tl.store(
            xc + slot * XS + head * LENGTH * 64 + (16 + t[:, None]) * 64 + dim[None, :],
            new_x,
            (t[:, None] < count) & (dim[None, :] < 64),
        )
        tl.store(dc + slot * DS + head * LENGTH + 16 + t, new_dt, t < count)
        if head < GROUPS:
            old_b = tl.load(
                bc
                + slot * BS
                + head * LENGTH * 128
                + (16 + t[:, None]) * 128
                + dim[None, :],
                t[:, None] < accepted,
                0,
            )
            tl.store(
                bc
                + slot * BS
                + head * LENGTH * 128
                + dst[:, None] * 128
                + dim[None, :],
                old_b,
                retain[:, None],
            )
            new_b = tl.load(
                b
                + (begin + t[:, None]) * B_STRIDES[0]
                + head * B_STRIDES[1]
                + dim[None, :] * B_STRIDES[2],
                t[:, None] < count,
                0,
            )
            tl.store(
                bc
                + slot * BS
                + head * LENGTH * 128
                + (16 + t[:, None]) * 128
                + dim[None, :],
                new_b,
                t[:, None] < count,
            )
    else:
        tl.store(
            out + (begin + t[:, None]) * HEADS * 64 + head * 64 + dim[None, :],
            0,
            (t[:, None] < count) & (dim[None, :] < 64),
        )


def _pending_state_update(
    cache,
    x,
    dt,
    a,
    b,
    c,
    d,
    bias,
    ids,
    out,
    *,
    nvfp4,
    seed,
    num_accepted_tokens,
    cu_seqlens,
    max_seqlen,
):
    slots, heads = cache[1].shape[:2]
    groups, length = cache[4].shape[1:3]
    batch = ids.numel()
    if not batch:
        return out
    if not 1 <= max_seqlen <= 8 or length != 16 + max_seqlen:
        raise ValueError(
            "Pending ReplaySSM requires matching ring capacity for 1..8 candidates"
        )
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Slot indices must be int32 or int64")
    if any(t.device != cache[1].device for t in (x, dt, a, b, c, d, bias, ids, out)):
        raise ValueError("Pending decode tensors must use the cache device")
    if any(t.dtype != torch.bfloat16 for t in (x, b, c, out)):
        raise ValueError("Pending ReplaySSM requires BF16 x, B, C, and output")
    tokens = x.numel() // (heads * 64)
    if out.numel() != tokens * heads * 64 or not out.is_contiguous():
        raise ValueError("Pending output must be contiguous [tokens, heads, 64]")
    if cu_seqlens is None:
        if tokens != batch or max_seqlen != 1:
            raise ValueError(
                "Multi-token ReplaySSM requires cumulative sequence lengths"
            )
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=x.device)
    if cu_seqlens.shape != (batch + 1,) or cu_seqlens.device != x.device:
        raise ValueError("Cumulative lengths must have one entry per request plus one")
    if num_accepted_tokens is None:
        if max_seqlen != 1:
            raise ValueError("Speculative ReplaySSM requires previous accepted counts")
        num_accepted_tokens = torch.ones(batch, dtype=torch.int32, device=x.device)
    if num_accepted_tokens.numel() != batch or num_accepted_tokens.device != x.device:
        raise ValueError("Previous accepted counts must have one entry per request")
    ids = ids.reshape(-1).contiguous()
    cu = cu_seqlens.to(torch.int32).contiguous()
    accepted = num_accepted_tokens.reshape(-1).to(torch.int32).contiguous()
    if any(t.dtype not in (torch.float32, torch.bfloat16) for t in (dt, a, d, bias)):
        raise ValueError("Pending dt, A, D, and bias must use FP32 or BF16")
    x = x.reshape(tokens, heads, 64)
    dt = dt.reshape(tokens, heads, -1)[:, :, 0]
    a = a.reshape(heads, -1)[:, 0]
    bias = bias.reshape(heads, -1)[:, 0]
    d = d.reshape(heads, -1).expand(heads, 64)
    b = b.reshape(tokens, groups, 128)
    c = c.reshape(tokens, groups, 128)
    native_ids = torch.empty(batch, dtype=torch.int32, device=x.device)
    control = torch.empty((batch, 5), dtype=torch.int64, device=x.device)
    meta, parity = cache[7:9] if nvfp4 else (cache[6], cache[6])
    _prepare_pending[(batch,)](
        ids,
        accepted,
        cu,
        cache[-1],
        meta,
        parity,
        native_ids,
        control,
        slots,
        cache[-1].stride(0),
        meta.stride(0),
        parity.stride(0),
        groups,
        triton.next_power_of_2(groups),
        nvfp4,
    )
    from vllm.third_party.replayssm.mamba2.pending import run_pending_kernels

    run_pending_kernels(
        cache,
        x,
        dt,
        a,
        b,
        c,
        d,
        bias,
        native_ids,
        cu,
        control,
        out,
        nvfp4=nvfp4,
        max_tokens=max_seqlen,
        seed=seed,
    )
    _stage_pending[(batch, heads)](
        native_ids,
        cu,
        control,
        x,
        dt,
        b,
        bias,
        *cache[2:5],
        out,
        cache[2].stride(0),
        cache[3].stride(0),
        cache[4].stride(0),
        x.stride(),
        dt.stride(),
        b.stride(),
        bias.stride(0),
        heads,
        groups,
        length,
        max_seqlen,
        triton.next_power_of_2(max_seqlen),
    )
    return out


def _nvfp4(state_dtype: str) -> bool:
    if state_dtype not in ("fp8", "fp8_e4m3", "fp8_e4m3fn", "nvfp4"):
        raise ValueError("Separate checkpoints require FP8 E4M3 or NVFP4")
    return state_dtype == "nvfp4"


def _check_cache(cache: Sequence[torch.Tensor], nvfp4: bool) -> tuple[int, int, int]:
    if len(cache) != (10 if nvfp4 else 8):
        raise ValueError("Unexpected separate ReplaySSM cache layout")
    state = cache[1]
    if state.dtype != torch.float16 or state.shape[2:] != (64, 128):
        raise ValueError(
            "The recurrence checkpoint must be FP16 [slots, heads, 64, 128]"
        )
    slots, heads = state.shape[:2]
    groups = cache[4].shape[1]
    pending = cache[-1].shape == (slots, 2)
    length = cache[2].shape[2]
    if (pending and not 17 <= length <= 24) or (not pending and length != 16):
        raise ValueError(
            "ReplaySSM requires 16 history entries and at most 8 pending entries"
        )
    if heads // groups not in (8, 16) or heads % groups != 0:
        raise ValueError("Separate ReplaySSM requires 8 or 16 heads per group")
    expected = [
        ((slots, heads, 64, 128), torch.float16),
        ((slots, heads, length, 64), torch.bfloat16),
        ((slots, heads, length), torch.float32),
        ((slots, groups, length, 128), torch.bfloat16),
    ]
    if nvfp4:
        expected += [
            ((slots, heads // 2, 128, 64), torch.uint8),
            ((slots, heads // 2, 4, 256), torch.uint8),
            ((slots, 3, groups), torch.float32),
            ((slots, groups), torch.int32),
        ]
    else:
        expected += [
            ((slots, heads, 64, 128), torch.float8_e4m3fn),
            ((slots, heads, 64), torch.float32),
        ]
    for index, (tensor, (shape, dtype)) in enumerate(zip(cache[1:-1], expected), 1):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"Expected cache shape {shape} and dtype {dtype}")
        inner_strides = tuple(math.prod(shape[i + 1 :]) for i in range(1, len(shape)))
        if tuple(tensor.stride()[1:]) != inner_strides:
            raise ValueError(
                "Cache dimensions within a physical slot must be contiguous"
            )
        if tensor.stride(0) < math.prod(shape[1:]):
            raise ValueError("Cache slots must not overlap")
        if tensor.device != state.device or not tensor.is_cuda:
            raise ValueError("All cache tensors must use the same CUDA device")
        if index <= 6 and (
            tensor.data_ptr() % 16 or tensor.stride(0) * tensor.element_size() % 16
        ):
            raise ValueError("Cache pages must have 16-byte alignment")
    if (
        cache[-1].shape not in ((slots,), (slots, 1), (slots, 2))
        or cache[-1].dtype != torch.int64
    ):
        raise ValueError("Event positions must be int64 with one value per slot")
    if cache[-1].device != state.device:
        raise ValueError("Event positions must use the cache device")
    capability = torch.cuda.get_device_capability(state.device)
    if capability != ((12, 1) if pending else (10, 3)):
        raise ValueError(
            "Separate ReplaySSM requires SM103 for immediate state "
            "or SM121 for pending state"
        )
    return slots, heads, groups


def initialize_replayssm_state(
    cache: Sequence[torch.Tensor],
    states: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    state_dtype: str,
    seed: int = 0,
) -> None:
    """Initialize both checkpoints independently from FP32 prefill states.

    This operation also clears replay rings and event positions for reused slots.
    Nonpositive and out-of-range slot indices are ignored.
    """
    nvfp4 = _nvfp4(state_dtype)
    slots, heads, groups = _check_cache(cache, nvfp4)
    if states.dtype != torch.float32 or states.shape[1:] != (heads, 64, 128):
        raise ValueError("Prefill states must be FP32 [batch, heads, 64, 128]")
    if states.device != cache[1].device or state_indices.device != states.device:
        raise ValueError("Prefill states and slot indices must use the cache device")
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Slot indices must be int32 or int64")
    ids = state_indices.reshape(-1).contiguous()
    if ids.numel() != states.shape[0]:
        raise ValueError("Each prefill state requires one slot index")
    if ids.numel() == 0:
        return
    if states.stride()[1:] != (64 * 128, 128, 1):
        states = states.contiguous()
    if nvfp4:
        # Warmup can broadcast one zero state over all request slots.
        source = states[:1] if states.stride(0) == 0 else states
        group_max = (
            source.abs().reshape(-1, groups, (heads // groups) * 64 * 128).amax(dim=-1)
        )
        global_scale = group_max.amax(dim=-1) / (6.0 * 448.0)
        global_scale = torch.where(global_scale > 0, global_scale, 1.0)
        max_stride = 0 if states.stride(0) == 0 else group_max.stride(0)
        global_stride = 0 if states.stride(0) == 0 else global_scale.stride(0)
        meta, parity = cache[7:9]
    else:
        group_max = global_scale = meta = parity = cache[6]
        max_stride = global_stride = 0
    _initialize_rows[(ids.numel(), heads * 64)](
        states,
        ids,
        cache[1].view(torch.int32),
        cache[5],
        cache[6],
        group_max,
        global_scale,
        meta,
        parity,
        states.stride(0),
        cache[1].stride(0) // 2,
        cache[5].stride(0),
        cache[6].stride(0),
        meta.stride(0),
        parity.stride(0),
        max_stride,
        global_stride,
        slots,
        heads,
        groups,
        nvfp4,
        seed,
        cache[-1].shape == (slots, 2),
        enable_fp_fusion=False,
    )
    counts = [math.prod(t.shape[1:]) for t in cache[2:5]]
    _clear_slots[(ids.numel(), triton.cdiv(max(counts), 256))](
        ids,
        cache[2],
        cache[3],
        cache[4],
        cache[-1],
        cache[2].stride(0),
        cache[3].stride(0),
        cache[4].stride(0),
        cache[-1].stride(0),
        *counts,
        2 if cache[-1].shape == (slots, 2) else 1,
        slots,
        256,
    )


def selective_state_update_replayssm(
    cache: Sequence[torch.Tensor],
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    state_dtype: str,
    seed: int = 0,
    num_accepted_tokens: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int = 1,
) -> torch.Tensor:
    """Decode one token and advance device-resident positions.

    An eager call compiles both flush paths before CUDA graph capture. Calls
    apply softplus to dt plus dt_bias. Active slot indices must be unique.
    """
    nvfp4 = _nvfp4(state_dtype)
    slots, heads, groups = _check_cache(cache, nvfp4)
    if cache[-1].shape == (slots, 2):
        return _pending_state_update(
            cache,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            state_indices,
            out,
            nvfp4=nvfp4,
            seed=seed,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
    if max_seqlen != 1:
        raise ValueError("SM103 separate ReplaySSM supports one token")
    batch = state_indices.numel()
    if batch == 0:
        return out
    if (
        x.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or C.dtype != torch.bfloat16
    ):
        raise ValueError("Separate ReplaySSM requires BF16 x, B, and C")
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Slot indices must be int32 or int64")
    if (
        out.dtype != torch.bfloat16
        or out.numel() != batch * heads * 64
        or not out.is_contiguous()
    ):
        raise ValueError("Output must be contiguous BF16 [batch, heads, 64]")
    tensors = (x, dt, A, B, C, D, dt_bias, state_indices, out)
    if any(t.device != cache[1].device for t in tensors):
        raise ValueError("Decode tensors must use the cache device")
    x = x.reshape(batch, heads, 64).contiguous()
    B = B.reshape(batch, groups, 128).contiguous()
    C = C.reshape(batch, groups, 128).contiguous()
    # The serving layer may supply tied head dimensions as stride-zero views.
    dt = dt.reshape(batch, heads, -1)[:, :, 0].float().contiguous()
    A = A.reshape(heads, -1)[:, 0].float().contiguous()
    dt_bias = dt_bias.reshape(heads, -1)[:, 0].float().contiguous()
    D = D.reshape(heads, -1).expand(heads, 64).float().contiguous()
    ids = state_indices.reshape(-1).contiguous()
    native_ids = torch.empty(batch, device=x.device, dtype=torch.int32)
    tags = torch.empty(batch, device=x.device, dtype=torch.int64)
    positions = torch.empty(batch, device=x.device, dtype=torch.int32)
    flush = torch.empty(batch, device=x.device, dtype=torch.int8)
    native_out = out.view(batch, heads, 64)
    _prepare_decode[(batch,)](
        ids,
        cache[-1],
        native_ids,
        tags,
        positions,
        flush,
        native_out,
        slots,
        cache[-1].stride(0),
        heads * 64,
        256,
    )
    common = (x, dt, A, B, C, D, dt_bias, cache[2], cache[3], cache[4], native_ids)
    if nvfp4:
        from vllm.third_party.replayssm.mamba2.nvfp4_flush import mamba2_nvfp4_flush
        from vllm.third_party.replayssm.mamba2.nvfp4_nonflush import (
            mamba2_nvfp4_nonflush,
        )

        mamba2_nvfp4_nonflush(
            *common,
            positions,
            flush,
            native_out,
            state_lo=cache[5],
            sf_lo=cache[6],
            sg_meta=cache[7],
        )
        mamba2_nvfp4_flush(
            cache[1],
            *common,
            tags,
            flush,
            native_out,
            state_lo=cache[5],
            sf_lo=cache[6],
            sg_meta=cache[7],
            sg_parity=cache[8],
            stochastic_rounding_seed=seed,
        )
    else:
        from vllm.third_party.replayssm.mamba2.fp8_flush import mamba2_fp8_flush
        from vllm.third_party.replayssm.mamba2.fp8_nonflush import mamba2_fp8_nonflush

        mamba2_fp8_nonflush(
            *common,
            positions,
            flush,
            native_out,
            state_lo=cache[5],
            state_lo_scale=cache[6],
        )
        mamba2_fp8_flush(
            cache[1],
            *common,
            tags,
            flush,
            native_out,
            state_lo=cache[5],
            state_lo_scale=cache[6],
            stochastic_rounding_seed=seed,
        )
    return out
