# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental complete K1 pipelines under the existing residency contract."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.attention import kda as k
from vllm.models.kimi_k3.amd.mono.common.ops import memrealtime, traced
from vllm.models.kimi_k3.amd.mono.common.sync import publish


def pack_projection_weights(weight):
    """Pack immutable BF16 projection weights for contiguous wave loads."""
    assert weight.shape == (k.NPROJ, k.HIDDEN)
    return (
        weight.view(k.GEMV_TASKS, k.ROWS, k.PER_WAVE, k.WAVES, 4, 4, 8)
        .permute(0, 2, 3, 4, 5, 1, 6)
        .contiguous()
        .view_as(weight)
    )


@traced
def cache_input(c, start, width):
    words = c["S"] * width // 8
    count = (words + k.THREADS - 1) // k.THREADS
    for first in range_constexpr(0, count, 2):
        values = []
        for i in range_constexpr(first, min(first + 2, count)):
            e = fx.min(c["tid"] + k.THREADS * i, words - 1)
            token, col = e // (width // 8), e % (width // 8) * 8
            value = fx.Vector(
                bo.buffer_load(
                    k.rsrc(c["xbuf"]),
                    (token * k.HIDDEN + start + col) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=k.CM_DEV,
                )
            )
            values.append((token, col, value))
        for i in range_constexpr(len(values)):
            token, col, value = values[i]
            if c["tid"] + k.THREADS * (first + i) < words:
                fx.ptr_store(
                    value, c["xl"] + (token * (c["XCOLS"] + k.LDS_PAD) + col) // 2
                )
    gpu.barrier()


def weight_offset(c, group, vector):
    if c["pipeline"] & 2097152:
        return group * k.THREADS * 16 + vector * k.THREADS * 4 + c["tid"] * 4
    return (group * k.THREADS + c["tid"]) * 16 + vector * 4


def cache_tail_math(c, acc0, acc1):
    token = fx.min(c["lane"] % k.ROWS, c["S"] - 1)
    col = c["wave"] * k.KCH + c["lane"] // k.ROWS * 8
    for q in range(4):
        xb = fx.Vector(
            fx.ptr_load(
                c["xl"] + (token * (c["XCOLS"] + k.LDS_PAD) + col + 32 * q) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        ).bitcast(fx.BFloat16)
        w0 = fx.Vector(
            fx.ptr_load(
                c["wl"] + weight_offset(c, 0, q),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        ).bitcast(fx.BFloat16)
        acc0 = k.mfma_bf16(w0, xb, acc0)
        w1 = fx.Vector(
            fx.ptr_load(
                c["wl"] + weight_offset(c, 1, q),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        ).bitcast(fx.BFloat16)
        acc1 = k.mfma_bf16(w1, xb, acc1)
    return acc0, acc1


@traced
def cached_projection(contexts, rg0, rg1):
    """Keep the last weight chunk in LDS to bound register lifetime."""
    c0 = contexts[0]
    for group in range_constexpr(2):
        tail = k.gemv_loads(c0, rg0 if group == 0 else rg1, [6])
        for q in range_constexpr(4):
            fx.ptr_store(tail[0][q], c0["wl"] + weight_offset(c0, group, q))
    if const_expr(c0["pipeline"] & 16384):
        for group in range_constexpr(2):
            extra = k.gemv_loads(c0, rg0 if group == 0 else rg1, [5], first_column=3)
            fx.ptr_store(
                extra[0][0],
                c0["wl"] + 2 * k.THREADS * 16 + (group * k.THREADS + c0["tid"]) * 4,
            )
        ops0 = k.gemv_loads(c0, rg0, range(5)) + k.gemv_loads(c0, rg0, [5], columns=3)
        ops1 = k.gemv_loads(c0, rg1, range(5)) + k.gemv_loads(c0, rg1, [5], columns=3)
    else:
        ops0 = k.gemv_loads(c0, rg0, range(6))
        ops1 = k.gemv_loads(c0, rg1, range(6))
    gpu.barrier()
    for tile in range_constexpr(len(contexts)):
        c = contexts[tile]
        if c["tid"] < k.NA:
            c["poll"]([(c["xrdy"], c["tid"], 1)])
        gpu.barrier()
        acc0, acc1 = (
            fx.Vector.filled(4, 0.0, fx.Float32),
            fx.Vector.filled(4, 0.0, fx.Float32),
        )
        for start in range_constexpr(0, 7, 2):
            count = min(2, 7 - start)
            event(c, c["event_init_base"] + c["bid"], start)
            cache_input(c, start * k.WAVES * k.KCH, count * k.WAVES * k.KCH)
            if const_expr(start < 6):
                acc0, acc1 = chunk_math(
                    c,
                    count,
                    ops0[start : start + count],
                    ops1[start : start + count],
                    acc0,
                    acc1,
                    start=start,
                )
            else:
                acc0, acc1 = cache_tail_math(c, acc0, acc1)
            gpu.barrier()
            event(c, c["event_init_base"] + c["bid"], start + 1)
        k.projection_out(c, rg0, acc0)
        k.projection_out(c, rg1, acc1)


def worker_weights(c, group, part):
    lane, wave = c["lane"], fx.max(c["wave"] - 1, 0)
    row = fx.min(group, k.GEMV_TASKS - 1) * k.ROWS + lane % k.ROWS
    size = (c["wave"] > 0).select(fx.Int32(k.NPROJ * k.HIDDEN * 2), fx.Int32(0))
    resource = k.rsrc(c["args"]["w_in"], nbytes=size)
    return [
        [
            fx.Vector(
                bo.buffer_load(
                    resource,
                    (
                        row * k.HIDDEN
                        + (wave + 7 * (part + i)) * k.KCH
                        + lane // k.ROWS * 8
                    )
                    // 2
                    + 16 * q,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=k.CM_NT,
                )
            )
            for q in range(4)
        ]
        for i in range(2)
    ]


@traced
def loader_tile(c, part, bank):
    """Wave zero fills the unused bank while the other waves compute."""
    if c["lane"] < k.NA:
        c["poll"]([(c["xrdy"], c["lane"], 1)])
    width, stride = 14 * k.KCH, c["XCOLS"] + k.LDS_PAD
    words = c["S"] * width // 8
    count = (words + 63) // 64
    for first in range_constexpr(0, count, 8):
        loads = []
        for i in range_constexpr(first, min(first + 8, count)):
            e = fx.min(c["lane"] + 64 * i, words - 1)
            token, col = e // (width // 8), e % (width // 8) * 8
            v = fx.Vector(
                bo.buffer_load(
                    k.rsrc(c["xbuf"]),
                    (token * k.HIDDEN + part * 7 * k.KCH + col) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=k.CM_DEV,
                )
            )
            loads.append((token, col, v))
        for i in range_constexpr(len(loads)):
            token, col, v = loads[i]
            if c["lane"] + 64 * (first + i) < words:
                fx.ptr_store(
                    v,
                    c["xl"] + (bank * c["XROWS"] * stride + token * stride + col) // 2,
                )


def worker_math(c, bank, ops0, ops1, acc0, acc1):
    token = fx.min(c["lane"] % k.ROWS, c["S"] - 1)
    stride = c["XCOLS"] + k.LDS_PAD
    wave = c["wave"] - 1
    for i in range(2):
        col = (wave + 7 * i) * k.KCH + c["lane"] // k.ROWS * 8
        for q in range(4):
            xb = fx.Vector(
                fx.ptr_load(
                    c["xl"]
                    + (bank * c["XROWS"] * stride + token * stride + col + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            ).bitcast(fx.BFloat16)
            acc0 = k.mfma_bf16(ops0[i][q].bitcast(fx.BFloat16), xb, acc0)
            acc1 = k.mfma_bf16(ops1[i][q].bitcast(fx.BFloat16), xb, acc1)
    return acc0, acc1


@traced
def wave_projection(contexts, rg0, rg1):
    """One loader wave and seven compute waves share two LDS banks."""
    c0 = contexts[0]
    if c0["wave"] == 0:
        loader_tile(c0, 0, 0)
    ops0, ops1 = worker_weights(c0, rg0, 0), worker_weights(c0, rg1, 0)
    gpu.barrier()
    accs0 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in contexts]
    accs1 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in contexts]
    for part in range_constexpr(0, 8, 2):
        for tile in range_constexpr(len(contexts)):
            c = contexts[tile]
            stage = part // 2 * len(contexts) + tile
            event(c, c["event_init_base"] + c["bid"], part)
            if const_expr(tile + 1 == len(contexts) and part + 2 < 8):
                next0 = worker_weights(c0, rg0, part + 2)
                next1 = worker_weights(c0, rg1, part + 2)
            if const_expr(stage + 1 < 4 * len(contexts)):  # noqa: SIM102
                if c["wave"] == 0:
                    next_tile = (tile + 1) % len(contexts)
                    next_part = part + (2 if next_tile == 0 else 0)
                    loader_tile(contexts[next_tile], next_part, (stage + 1) % 2)
            acc0, acc1 = accs0[tile], accs1[tile]
            if c["wave"] > 0:
                acc0, acc1 = worker_math(c, stage % 2, ops0, ops1, acc0, acc1)
            accs0[tile], accs1[tile] = acc0, acc1
            gpu.barrier()
            event(c, c["event_init_base"] + c["bid"], part + 1)
        if const_expr(part + 2 < 8):
            ops0, ops1 = next0, next1
    for tile in range_constexpr(len(contexts)):
        k.projection_out(contexts[tile], rg0, accs0[tile])
        k.projection_out(contexts[tile], rg1, accs1[tile])


def prefetch_state(c, first):
    task0 = k.first_task(c["bid"], first)
    count = c["S"] // c["L"] * k.NH * c["NVCH"]
    task = fx.min(task0, count - 1)
    req, head, chunk = (
        task // (k.NH * c["NVCH"]),
        task // c["NVCH"] % k.NH,
        task % c["NVCH"],
    )
    a = c["args"]
    accepted = k.uniform(k.ld_i32(a["num_acc"], req)) - 1
    slot = k.uniform(k.ld_i32(a["st_idx"], req * a["st_stride"] + accepted))
    base = a["rstate"] + fx.Int64(fx.max(slot, 0)) * fx.Int64(a["rs_seq"]) * 4
    size = (task0 < count).select(a["rs_seq"] * 4, fx.Int32(0))
    resource = k.rsrc(base, nbytes=size)
    offset = (head * k.HD + chunk * c["VCH"] + c["tid"] // 32) * k.HD + c[
        "tid"
    ] % 32 * 4
    return [
        fx.Vector(
            bo.buffer_load(resource, offset + r * 16 * k.HD, vec_width=4, dtype=T.f32)
        )
        for r in range(c["VCH"] // 16)
    ]


@traced
def wave_gates(c, staged, req, head):
    """Four waves compute gates while four normalize Q/K, with local outputs."""
    L, tid, lane, wave = c["L"], c["tid"], c["lane"], c["wave"]
    if wave < 4:
        token = fx.min(lane % 16, L - 1)
        col = lane // 16 * 32
        got = c["poll"](
            [
                (c["proj"], (req * L + token) * k.NPROJ + k.FA_0 + col + e, 1)
                for e in range(32)
            ]
        )
        operands = [
            fx.Vector.from_elements(
                [k.f32_of(got[q * 8 + e][0]) for e in range(8)], fx.Float32
            ).to(fx.BFloat16)
            for q in range(4)
        ]
        for half in range_constexpr(2):
            weights = staged["fb"] if half == 0 else staged["fb_hi"]
            bias = staged["dtb"] if half == 0 else staged["dtb_hi"]
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for q in range_constexpr(4):
                acc = k.mfma_bf16(weights[q].bitcast(fx.BFloat16), operands[q], acc)
            for i in range_constexpr(4):
                col_out = (wave + half * 4) * 16 + 4 * (lane // 16) + i
                gate = fx.Float32(c["lb"]) * k.sigmoid(
                    staged["aa"] * (k.bf16r(acc[i]) + bias[i])
                )
                if lane % 16 < L:
                    fx.ptr_store(
                        k.hw_exp2(gate * fx.Float32(k.LOG2E)),
                        c["decay"] + token * k.HD + col_out,
                    )
        if tid < L:
            beta = c["poll"](
                [(c["proj"], (req * L + tid) * k.NPROJ + k.BETA_0 + head, 1)]
            )[0][0]
            fx.ptr_store(k.sigmoid(k.f32_of(beta)), c["beta"] + tid)
    else:
        for first in range_constexpr(0, 2 * L, 4):
            row = wave - 4 + first
            if row < 2 * L:
                got = c["poll"](
                    [
                        (
                            c["qk"],
                            (req * k.NH + head) * L * 2 * k.HD + row * k.HD + lane + d,
                            1,
                        )
                        for d in (0, 64)
                    ]
                )
                x0, x1 = k.f32_of(got[0][0]), k.f32_of(got[1][0])
                ss = k.butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
                scale = k.hw_rsq(ss + fx.Float32(1e-6))
                scale = (row % 2 == 0).select(scale * fx.Float32(k.HD**-0.5), scale)
                fx.ptr_store(x0 * scale, c["qkl"] + row * k.HD + lane)
                fx.ptr_store(x1 * scale, c["qkl"] + row * k.HD + lane + 64)
    gpu.barrier()


@traced
def resident_projection(contexts, rg0, rg1):
    """Keep all weights in registers and publish a complete tile at a time."""
    ops0 = k.gemv_loads(contexts[0], rg0, range(k.PER_WAVE))
    ops1 = k.gemv_loads(contexts[0], rg1, range(k.PER_WAVE))
    for tile in range_constexpr(len(contexts)):
        c = contexts[tile]
        acc0, acc1 = (
            fx.Vector.filled(4, 0.0, fx.Float32),
            fx.Vector.filled(4, 0.0, fx.Float32),
        )
        for half in range_constexpr(2):
            event(c, c["event_init_base"] + c["bid"], 2 * half)
            lo, hi = half * 4, min(half * 4 + 4, k.PER_WAVE)
            acc0, acc1 = k.projection_chunk(
                c, half, ops0[lo:hi], ops1[lo:hi], acc0, acc1
            )
            event(c, c["event_init_base"] + c["bid"], 2 * half + 1)
        k.projection_out(c, rg0, acc0)
        k.projection_out(c, rg1, acc1)


@traced
def await_inputs(contexts):
    """Only wave zero polls, before that wave issues any weight requests."""
    c = contexts[0]
    if c["tid"] < k.NA:
        for tile in range_constexpr(len(contexts)):
            ct = contexts[tile]
            ct["poll"]([(ct["xrdy"], c["tid"], 1)])


def chunk_loads(c, start, width):
    words, tid = c["S"] * width // 8, c["tid"]
    values = []
    for i in range((words + k.THREADS - 1) // k.THREADS):
        e = fx.min(tid + k.THREADS * i, words - 1)
        token, col = e // (width // 8), e % (width // 8) * 8
        value = fx.Vector(
            bo.buffer_load(
                k.rsrc(c["xbuf"]),
                (token * k.HIDDEN + start + col) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=k.CM_DEV,
            )
        )
        values.append((token, col, value))
    return values


@traced
def chunk_store(c, width, values):
    words = c["S"] * width // 8
    for i in range_constexpr(len(values)):
        token, col, value = values[i]
        if c["tid"] + k.THREADS * i < words:
            fx.ptr_store(value, c["xl"] + (token * (c["XCOLS"] + k.LDS_PAD) + col) // 2)
    gpu.barrier()


def chunk_math(c, count, ops0, ops1, acc0, acc1, start=0):
    token = fx.min(c["lane"] % k.ROWS, c["S"] - 1)
    stride = c["XCOLS"] + k.LDS_PAD
    for i in range(count):
        col = (c["wave"] + k.WAVES * i) * k.KCH + c["lane"] // k.ROWS * 8
        for q in range(4):
            xb = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (token * stride + col + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            ).bitcast(fx.BFloat16)
            if c["pipeline"] & 16384 and start + i == 5 and q == 3:
                off = 2 * k.THREADS * 16 + c["tid"] * 4
                w0 = fx.Vector(
                    fx.ptr_load(
                        c["wl"] + off, result_type=fx.Vector.make_type(4, fx.Int32)
                    )
                )
                w1 = fx.Vector(
                    fx.ptr_load(
                        c["wl"] + off + k.THREADS * 4,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                acc0 = k.mfma_bf16(w0.bitcast(fx.BFloat16), xb, acc0)
                acc1 = k.mfma_bf16(w1.bitcast(fx.BFloat16), xb, acc1)
            else:
                acc0 = k.mfma_bf16(ops0[i][q].bitcast(fx.BFloat16), xb, acc0)
                acc1 = k.mfma_bf16(ops1[i][q].bitcast(fx.BFloat16), xb, acc1)
    return acc0, acc1


def operand_loads(c, part):
    token = fx.min(c["lane"] % k.ROWS, c["S"] - 1)
    col = (c["wave"] + k.WAVES * part) * k.KCH + c["lane"] // k.ROWS * 8
    return [
        fx.Vector(
            bo.buffer_load(
                k.rsrc(c["xbuf"]),
                (token * k.HIDDEN + col + 32 * q) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=k.CM_DEV,
            )
        )
        for q in range(4)
    ]


@traced
def register_projection(contexts, rg0, rg1):
    """Prefetch MFMA operands directly; keep weights across token tiles."""
    c0 = contexts[0]
    ops0 = k.gemv_loads(c0, rg0, [0])
    ops1 = k.gemv_loads(c0, rg1, [0])
    gpu.barrier()
    values = operand_loads(c0, 0)
    acc0 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in contexts]
    acc1 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in contexts]
    for tile in range_constexpr(len(contexts)):
        c = contexts[tile]
        event(c, c["event_init_base"] + c["bid"], 0)
    for part in range_constexpr(k.PER_WAVE):
        for tile in range_constexpr(len(contexts)):
            current = values
            if const_expr(tile + 1 < len(contexts)):
                values = operand_loads(contexts[tile + 1], part)
            elif const_expr(part + 1 < k.PER_WAVE):
                values = operand_loads(c0, part + 1)
                next0 = k.gemv_loads(c0, rg0, [part + 1])
                next1 = k.gemv_loads(c0, rg1, [part + 1])
            for q in range_constexpr(4):
                xb = current[q].bitcast(fx.BFloat16)
                acc0[tile] = k.mfma_bf16(
                    ops0[0][q].bitcast(fx.BFloat16), xb, acc0[tile]
                )
                acc1[tile] = k.mfma_bf16(
                    ops1[0][q].bitcast(fx.BFloat16), xb, acc1[tile]
                )
        if const_expr(part + 1 < k.PER_WAVE):
            ops0, ops1 = next0, next1
    for tile in range_constexpr(len(contexts)):
        c = contexts[tile]
        event(c, c["event_init_base"] + c["bid"], 1)
        k.projection_out(c, rg0, acc0[tile])
        k.projection_out(c, rg1, acc1[tile])


@traced
def prefetched_projection(contexts, rg0, rg1):
    """Reuse each weight chunk over all token tiles; prefetch the next task."""
    c0 = contexts[0]
    chunks = c0["projection_chunks"]
    count0 = min(chunks, k.PER_WAVE)
    ops0 = k.gemv_loads(c0, rg0, range(count0))
    ops1 = k.gemv_loads(c0, rg1, range(count0))
    # Wave zero has finished polling; release the remaining waves to read x.
    gpu.barrier()
    values = chunk_loads(c0, 0, count0 * k.WAVES * k.KCH)
    acc0 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in contexts]
    acc1 = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in contexts]
    for start in range_constexpr(0, k.PER_WAVE, chunks):
        count = min(chunks, k.PER_WAVE - start)
        width = count * k.WAVES * k.KCH
        for tile in range_constexpr(len(contexts)):
            c = contexts[tile]
            event(c, c["event_init_base"] + c["bid"], 2 * (start // chunks))
            chunk_store(c, width, values)
            if const_expr(tile + 1 < len(contexts)):
                values = chunk_loads(contexts[tile + 1], start * k.WAVES * k.KCH, width)
            elif const_expr(start + count < k.PER_WAVE):
                next_start = start + count
                next_count = min(chunks, k.PER_WAVE - next_start)
                values = chunk_loads(
                    c0, next_start * k.WAVES * k.KCH, next_count * k.WAVES * k.KCH
                )
            if const_expr(
                start + count < k.PER_WAVE
                and tile == (0 if c0["pipeline"] & 32 else len(contexts) - 1)
            ):
                next_start = start + count
                next_count = min(chunks, k.PER_WAVE - next_start)
                next0 = k.gemv_loads(
                    c0, rg0, range(next_start, next_start + next_count)
                )
                next1 = k.gemv_loads(
                    c0, rg1, range(next_start, next_start + next_count)
                )
            acc0[tile], acc1[tile] = chunk_math(
                c, count, ops0, ops1, acc0[tile], acc1[tile]
            )
            # All reads from the current LDS tile precede the next overwrite.
            gpu.barrier()
            event(c, c["event_init_base"] + c["bid"], 2 * (start // chunks) + 1)
        if const_expr(start + count < k.PER_WAVE):
            ops0, ops1 = next0, next1
    for tile in range_constexpr(len(contexts)):
        k.projection_out(contexts[tile], rg0, acc0[tile])
        k.projection_out(contexts[tile], rg1, acc1[tile])


@traced
def event(c, record, point):
    if const_expr(c["on"]):  # noqa: SIM102
        if c["tid"] == 0:
            address = c["events"] + fx.Int64(record * c["event_stride"] + point) * 8
            fx.generic_store(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                    address,
                ),
                memrealtime(),
            )


@traced
def load_x_batched(c, start, width):
    words = c["S"] * width // 8
    tid = c["tid"]
    count = (words + k.THREADS - 1) // k.THREADS
    for first in range_constexpr(0, count, 8):
        staged = []
        for i in range_constexpr(first, min(first + 8, count)):
            e = fx.min(tid + k.THREADS * i, words - 1)
            row, col = e // (width // 8), e % (width // 8) * 8
            value = fx.Vector(
                bo.buffer_load(
                    k.rsrc(c["xbuf"]),
                    (row * k.HIDDEN + start + col) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=k.CM_DEV,
                )
            )
            staged.append((row, col, value))
        for j in range_constexpr(len(staged)):
            row, col, value = staged[j]
            if tid + k.THREADS * (first + j) < words:
                fx.ptr_store(
                    value, c["xl"] + (row * (c["XCOLS"] + k.LDS_PAD) + col) // 2
                )


def x_loads(c, part):
    width, tid = k.WAVES * k.KCH, c["tid"]
    words = c["S"] * width // 8
    loads = []
    for i in range((words + k.THREADS - 1) // k.THREADS):
        e = fx.min(tid + k.THREADS * i, words - 1)
        row, col = e // (width // 8), e % (width // 8) * 8
        value = fx.Vector(
            bo.buffer_load(
                k.rsrc(c["xbuf"]),
                (row * k.HIDDEN + part * width + col) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=k.CM_DEV,
            )
        )
        loads.append((e, value))
    return loads


@traced
def x_store(c, loads, bank):
    width, stride = k.WAVES * k.KCH, c["XCOLS"] + k.LDS_PAD
    words = c["S"] * width // 8
    for i in range_constexpr(len(loads)):
        e, value = loads[i]
        row, col = e // (width // 8), e % (width // 8) * 8
        if c["tid"] + k.THREADS * i < words:
            fx.ptr_store(
                value, c["xl"] + (bank * c["S"] * stride + row * stride + col) // 2
            )
    gpu.barrier()


def projection_math(c, bank, ops0, ops1, acc0, acc1):
    row = c["XCOLS"] + k.LDS_PAD
    token = fx.min(c["lane"] % k.ROWS, c["S"] - 1)
    col = c["wave"] * k.KCH + c["lane"] // k.ROWS * 8
    for q in range(4):
        xb = fx.Vector(
            fx.ptr_load(
                c["xl"] + (bank * c["S"] * row + token * row + col + 32 * q) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        ).bitcast(fx.BFloat16)
        acc0 = k.mfma_bf16(ops0[0][q].bitcast(fx.BFloat16), xb, acc0)
        acc1 = k.mfma_bf16(ops1[0][q].bitcast(fx.BFloat16), xb, acc1)
    return acc0, acc1


@traced
def projection(c, rg0, rg1):
    event(c, c["event_init_base"] + c["bid"], 0)
    if c["tid"] < k.NA:
        c["poll"]([(c["xrdy"], c["tid"], 1)])
    gpu.barrier()
    event(c, c["event_init_base"] + c["bid"], 1)
    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc1 = fx.Vector.filled(4, 0.0, fx.Float32)
    ops0, ops1 = k.gemv_loads(c, rg0, [0]), k.gemv_loads(c, rg1, [0])
    x_store(c, x_loads(c, 0), 0)
    for part in range_constexpr(k.PER_WAVE):
        if const_expr(part + 1 < k.PER_WAVE):
            next0 = k.gemv_loads(c, rg0, [part + 1])
            next1 = k.gemv_loads(c, rg1, [part + 1])
            next_x = x_loads(c, part + 1)
        acc0, acc1 = projection_math(c, part % 2, ops0, ops1, acc0, acc1)
        event(c, c["event_init_base"] + c["bid"], 2 + part)
        if const_expr(part + 1 < k.PER_WAVE):
            x_store(c, next_x, (part + 1) % 2)
            ops0, ops1 = next0, next1
    k.projection_out(c, rg0, acc0)
    k.projection_out(c, rg1, acc1)
    event(c, c["event_init_base"] + c["bid"], 9)


@traced
def gates(c, staged, req, head):
    L, tid, lane, wave = c["L"], c["tid"], c["lane"], c["wave"]
    for i in range_constexpr((L * k.HD + k.THREADS - 1) // k.THREADS):
        e = fx.min(tid + k.THREADS * i, L * k.HD - 1)
        value = c["poll"](
            [(c["proj"], (req * L + e // k.HD) * k.NPROJ + k.FA_0 + e % k.HD, 1)]
        )[0][0]
        if tid + k.THREADS * i < L * k.HD:
            fx.ptr_store(k.f32_of(value), c["fa"] + e)
    if tid < L:
        beta = c["poll"]([(c["proj"], (req * L + tid) * k.NPROJ + k.BETA_0 + head, 1)])[
            0
        ][0]
        fx.ptr_store(k.sigmoid(k.f32_of(beta)), c["beta"] + tid)
    gpu.barrier()
    token = fx.min(lane % 16, L - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        kb = (lane // 16) * 32 + 8 * q
        xs = [fx.ptr_load(c["fa"] + token * k.HD + kb + e) for e in range(8)]
        xb = fx.Vector.from_elements(xs, fx.Float32).to(fx.BFloat16)
        acc = k.mfma_bf16(staged["fb"][q].bitcast(fx.BFloat16), xb, acc)
    for i in range_constexpr(4):
        col = wave * 16 + 4 * (lane // 16) + i
        g = k.bf16r(acc[i]) + staged["dtb"][i]
        gate = fx.Float32(c["lb"]) * k.sigmoid(staged["aa"] * g)
        if lane % 16 < L:
            fx.ptr_store(
                k.hw_exp2(gate * fx.Float32(k.LOG2E)), c["decay"] + token * k.HD + col
            )
    for i in range_constexpr((2 * L * k.HD + k.THREADS - 1) // k.THREADS):
        e = fx.min(tid + k.THREADS * i, 2 * L * k.HD - 1)
        value = c["poll"]([(c["qk"], (req * k.NH + head) * L * 2 * k.HD + e, 1)])[0][0]
        if tid + k.THREADS * i < 2 * L * k.HD:
            fx.ptr_store(k.f32_of(value), c["qkl"] + e)
    gpu.barrier()
    for rr in range_constexpr((2 * L + k.WAVES - 1) // k.WAVES):
        row = fx.min(wave + k.WAVES * rr, 2 * L - 1)
        x0 = fx.ptr_load(c["qkl"] + row * k.HD + lane)
        x1 = fx.ptr_load(c["qkl"] + row * k.HD + 64 + lane)
        ss = k.butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        scale = k.hw_rsq(ss + fx.Float32(1e-6))
        scale = (row % 2 == 0).select(scale * fx.Float32(k.HD**-0.5), scale)
        if wave + k.WAVES * rr < 2 * L:
            fx.ptr_store(x0 * scale, c["qkl"] + row * k.HD + lane)
            fx.ptr_store(x1 * scale, c["qkl"] + row * k.HD + 64 + lane)
    gpu.barrier()


@traced
def head_prepare(c, head_task):
    req, head = head_task // k.NH, head_task % k.NH
    event(c, c["event_head_base"] + head_task, 0)
    staged = k.kda_stage(c, req, head, fx.Int32(0))
    gates(c, staged, req, head)
    event(c, c["event_head_base"] + head_task, 1)
    L, tid = c["L"], c["tid"]
    stride = 3 * L * k.HD + 16
    for name, start, count in (
        ("qkl", 0, 2 * L * k.HD),
        ("decay", 2 * L * k.HD, L * k.HD),
        ("beta", 3 * L * k.HD, L),
    ):
        for i in range_constexpr((count + k.THREADS - 1) // k.THREADS):
            e = fx.min(tid + k.THREADS * i, count - 1)
            value = fx.ptr_load(c[name] + e)
            if tid + k.THREADS * i < count:
                bo.buffer_store(
                    value,
                    k.rsrc(c["head_data"]),
                    head_task * stride + start + e,
                    cache_modifier=k.CM_DEV,
                )
    publish(c["put"], c["head_ready"], head_task, fx.Int32(1), tid == 0)
    event(c, c["event_head_base"] + head_task, 2)


@traced
def head_load(c, req, head):
    L, tid = c["L"], c["tid"]
    task, stride = req * k.NH + head, 3 * L * k.HD + 16
    if tid == 0:
        c["poll"]([(c["head_ready"], task, 1)])
    gpu.barrier()
    for name, start, count in (
        ("qkl", 0, 2 * L * k.HD),
        ("decay", 2 * L * k.HD, L * k.HD),
        ("beta", 3 * L * k.HD, L),
    ):
        for i in range_constexpr((count + k.THREADS - 1) // k.THREADS):
            e = fx.min(tid + k.THREADS * i, count - 1)
            value = bo.buffer_load(
                k.rsrc(c["head_data"]),
                task * stride + start + e,
                vec_width=1,
                dtype=T.f32,
                cache_modifier=k.CM_DEV,
            )
            if tid + k.THREADS * i < count:
                fx.ptr_store(fx.Float32(value), c[name] + e)
    gpu.barrier()


@traced
def recur(c, staged, req, head, chunk, task):
    L, tid, lane, a = c["L"], c["tid"], c["lane"], c["args"]
    v, k0 = tid // 32, (tid % 32) * 4
    live = staged["slot0"] > 0
    state = [
        [live.select(staged["st"][r][e], fx.Float32(0.0)) for e in range(4)]
        for r in range(c["VCH"] // 16)
    ]
    for j in range_constexpr(L):
        event(c, c["event_prod_base"] + task, 4 + 2 * j)
        qv = [fx.ptr_load(c["qkl"] + 2 * j * k.HD + k0 + e) for e in range(4)]
        kv = [fx.ptr_load(c["qkl"] + (2 * j + 1) * k.HD + k0 + e) for e in range(4)]
        decay = [fx.ptr_load(c["decay"] + j * k.HD + k0 + e) for e in range(4)]
        beta = fx.ptr_load(c["beta"] + j)
        slot = staged["slots"][j]
        for r in range_constexpr(c["VCH"] // 16):
            state[r] = [state[r][e] * decay[e] for e in range(4)]
            p = sum(state[r][e] * kv[e] for e in range(4))
            p = k.butterfly(p, (16, 8, 4, 2, 1))
            correction = (fx.ptr_load(c["vloc"] + j * c["VCH"] + v + 16 * r) - p) * beta
            state[r] = [state[r][e] + correction * kv[e] for e in range(4)]
            output = sum(state[r][e] * qv[e] for e in range(4))
            output = live.select(
                k.bf16r(k.butterfly(output, (16, 8, 4, 2, 1))), fx.Float32(0.0)
            )
            if (slot > 0) & live:
                dst = k.rsrc(a["rstate"] + fx.Int64(slot) * fx.Int64(a["rs_seq"]) * 4)
                bo.buffer_store(
                    fx.Vector.from_elements(state[r], fx.Float32),
                    dst,
                    staged["head_off"] + r * 16 * k.HD,
                )
            if lane % 32 == 0:
                fx.ptr_store(output, c["ol"] + j * c["VCH"] + v + 16 * r)
                at = (
                    (req * L + j) * k.PROJ + head * k.HD + chunk * c["VCH"] + v + 16 * r
                )
                if const_expr(c["pipeline"] & k.MAILBOX_OUTPUT):
                    c["put"](c["output_values"], at, output)
                else:
                    bo.buffer_store(
                        output, k.rsrc(c["ybuf"]), at, cache_modifier=k.CM_DEV
                    )
        gpu.barrier()
        if tid == 0:
            ss = fx.Float32(0.0)
            for e in range_constexpr(c["VCH"]):
                x = fx.ptr_load(c["ol"] + j * c["VCH"] + e)
                ss = ss + x * x
            if const_expr(c["pipeline"] & k.MAILBOX_OUTPUT):
                c["put"](
                    c["onrm"],
                    ((req * k.NH + head) * L + j) * c["NVCH"] + chunk,
                    ss,
                )
            else:
                fx.ptr_store(ss, c["red"])
        if const_expr(not (c["pipeline"] & k.MAILBOX_OUTPUT)):
            publish(
                c["put"],
                c["onrm"],
                ((req * k.NH + head) * L + j) * c["NVCH"] + chunk,
                fx.ptr_load(c["red"]),
                tid == 0,
            )
        event(c, c["event_prod_base"] + task, 5 + 2 * j)


@traced
def output(c, task):
    L, tid = c["L"], c["tid"]
    req, head, token = task // (k.NH * L), task // L % k.NH, task % L
    event(c, c["event_out_base"] + task, 0)
    if tid < k.HD:
        gate = c["poll"](
            [(c["proj"], (req * L + token) * k.NPROJ + k.G2_0 + head * k.HD + tid, 1)]
        )[0][0]
        scale_weight = k.ld_bf(c["args"]["on_w"], tid)
        gate = k.sigmoid(k.f32_of(gate))
        norms = c["poll"](
            [
                (c["onrm"], ((req * k.NH + head) * L + token) * c["NVCH"] + q, 1)
                for q in range(c["NVCH"])
            ]
        )
        ss = k.sum_polled(norms)
        scale = k.hw_rsq(ss * fx.Float32(1.0 / k.HD) + fx.Float32(c["onorm_eps"]))
        at = (req * L + token) * k.PROJ + head * k.HD + tid
        if const_expr(c["pipeline"] & k.MAILBOX_OUTPUT):
            value = k.f32_of(c["poll"]([(c["output_values"], at, 1)])[0][0])
        else:
            value = fx.Float32(
                bo.buffer_load(
                    k.rsrc(c["ybuf"]),
                    at,
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=k.CM_DEV,
                )
            )
        y = value * scale * scale_weight * gate
        bo.buffer_store(y.to(fx.BFloat16), k.rsrc(c["args"]["core_out"]), at)
    gpu.barrier()
    event(c, c["event_out_base"] + task, 1)
