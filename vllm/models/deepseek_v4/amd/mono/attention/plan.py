# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometry of the attention stages (``front``, ``back``): the KV record, the
split attention's chunks and the scratch (mailbox) layout; the model's widths
and a TP rank's share of them are in ``..dims``."""

from vllm.models.deepseek_v4.amd.mono.dims import (
    HEAD_DIM,
    HIDDEN,
    Q_RANK,
    QKV_ROWS,
    WQKV_KPARTS,
    X_GROUPS,
    Dims,
)

# KV record (V4 fp8_ds_mla): 576 B data rows (448 e4m3 | 64 bf16), then 8 B of
# scales a token (7 UE8M0 + pad) after the block's rows
DATA = 576
SCALE_BYTES = 8

# attention: a split is 16 keys
BK = 16
COLS = 32  # PV columns a group (one MX group of the output)
GROUPS = HEAD_DIM // COLS  # 16

PAIR = 8  # a mailbox (value, tag) pair


def cdiv(a, b):
    return -(-a // b)


def round_up(a, b):
    return cdiv(a, b) * b


def pair_layout(items, start=0, align=256):
    out, off = {}, start
    for name, pairs in items:
        off = round_up(off, align)
        out[name] = (off, pairs * PAIR)
        off += pairs * PAIR
    return out


def byte_layout(items, start=0, align=256):
    out, off = {}, start
    for name, nbytes in items:
        off = round_up(off, align)
        out[name] = (off, nbytes)
        off += nbytes
    return out


def front_scratch(s: int, d: Dims) -> dict:
    """K1's scratch regions -> (byte offset, bytes): published data (plain
    words behind a flag) and mailbox pairs."""
    return byte_layout(
        [
            ("x8", s * HIDDEN),  # e4m3 words (S > 16: quantized once, shared)
            ("x8s", s * X_GROUPS * 4),  # an i32 code a group
            ("xrdy", WQKV_KPARTS * s * PAIR),  # a flag a (token, K half)
            ("qkvp", WQKV_KPARTS * s * QKV_ROWS * PAIR),
            ("qx8", s * Q_RANK),  # e4m3 words
            ("qx8s", s * Q_RANK // 32 * 4),  # an i32 code a group
            ("qrdy", s * PAIR),
        ]
    )


def back_scratch(s: int, d: Dims, start: int = 0) -> dict:
    """K2's scratch regions, laid out past ``start``: published data (plain)
    and flag pairs."""
    ck = (
        64 if d.head_tiles == 2 and s * cdiv(d.keys, 64) <= 256 else 128
    )  # back.chunk_keys
    nchunk = cdiv(d.keys, ck)
    return byte_layout(
        [
            ("part", s * nchunk * d.heads * HEAD_DIM * 2),  # bf16 partials
            ("pstat", s * nchunk * d.heads * 8),  # (max, sum) f32
            ("srdy", s * nchunk * PAIR),
            ("xo", s * d.q_rows),  # e4m3 words
            ("xos", s * d.q_rows // 32 * 4),  # an i32 code a group
            ("crdy", s * d.heads // 8 * PAIR),
            ("x8b", s * d.o_rows),
            ("x8bs", s * d.o_rows // 32 * 4),
            ("ardy", d.woa_tasks * PAIR),
        ],
        start=start,
    )


def front_end(s: int, d: Dims) -> int:
    """Where K2's regions start: past K1's, which share the launch tag."""
    return round_up(max(o + n for o, n in front_scratch(s, d).values()), 256)


def scratch_bytes(s: int, d: Dims) -> int:
    b = back_scratch(s, d, start=front_end(s, d))
    return round_up(max(o + n for o, n in b.values()), 4096)
