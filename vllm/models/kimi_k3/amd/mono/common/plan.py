# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch geometry and the workspace layout of the mono MoE launch.

The workspace is one device allocation: the control words, then the sorted
routes, the MXFP4 intermediate and its scales, and the shared expert's K-split
partials and activation. Every offset is a compile-time constant, so the
kernel reaches each buffer from the control-word pointer.
"""

from aiter.ops.flydsl.kernels.mxfp4_gemm_common import kas_per_chunk_dw_for

BM = 16
WAVE = 64
THREADS = 256
N_WAVES = THREADS // WAVE
SPIN_SLEEP = 8
ROUTE_MARKS = 16

# Each control word sits on its own 128-byte line: words that different
# workgroups poll and bump never share a line.
LINE_WORDS = 32
W_TICKET = 0
W_ROUTED = LINE_WORDS
W_EPOCH = 2 * LINE_WORDS
W_TOPK = 3 * LINE_WORDS
W_MBLOCK = 4 * LINE_WORDS
MB_STRIDE = LINE_WORDS


def max_m_blocks(m: int, topk: int) -> int:
    """An expert gets ceil(routes / BM) m-blocks, so at most M * TOPK of them."""
    return m * topk


def sh_pairs(sh_inter: int) -> int:
    """Shared-expert gate/up column pairs: 16 gate and the matching 16 up columns."""
    return sh_inter // 16


def ctrl_words(m_max: int, topk: int, sh_inter: int) -> int:
    """Lines: ticket, routed, epoch, top-k, the m-block counters, one counter per
    shared-expert gate/up pair and the pairs-done counter."""
    return LINE_WORDS * (4 + max_m_blocks(m_max, topk) + sh_pairs(sh_inter) + 1)


def sh_pair_word(m_max: int, topk: int) -> int:
    return W_MBLOCK + max_m_blocks(m_max, topk) * MB_STRIDE


def sh_done_word(m_max: int, topk: int, sh_inter: int) -> int:
    return sh_pair_word(m_max, topk) + sh_pairs(sh_inter) * MB_STRIDE


def ws_layout(
    m_max: int, topk: int, inter: int, sh_inter: int, sh_ks: int
) -> tuple[dict[str, int], int]:
    """Byte offsets of the workspace buffers and the total size.

    sh_part holds the shared expert's gate/up K-split partials (fp32, one slice
    per wave of each K split, BM rows), sh_h its bf16 activation.
    """
    max_sorted = m_max * topk * BM
    sizes = dict(
        ctrl=ctrl_words(m_max, topk, sh_inter) * 4,
        stids=max_sorted * 4,
        sw=max_sorted * 4,
        eids=max_m_blocks(m_max, topk) * 4,
        cumsum=8,
        mind=max_sorted * 4,
        inter=max_sorted * (inter // 2),
        inter_scale=max(
            max_sorted * 64, max_sorted // BM * kas_per_chunk_dw_for(inter) * 4
        ),
        sh_part=sh_ks * N_WAVES * BM * sh_pairs(sh_inter) * 32 * 4,
        sh_h=BM * sh_inter * 2,
    )
    offs, total = {}, 0
    for k, size in sizes.items():
        offs[k] = total
        total += (size + 255) // 256 * 256
    return offs, total
