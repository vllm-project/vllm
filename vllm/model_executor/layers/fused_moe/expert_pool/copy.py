# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-grid row copies between the pinned host source (UVA view) and the
VRAM bank: one launch per step reads the device-side pair count, so an
empty step costs a few programs and nothing synchronizes. Two launch
shapes: "stripes" (program = (tensor, stripe) streaming its columns of
every row) and "chunks" (programs grid-stride over (row, chunk) pairs).
Ported from the lab expert tier (promote.py).
"""

from __future__ import annotations

from typing import Any

from vllm.model_executor.layers.fused_moe.expert_pool.tables import TENSORS

_KERNELS: dict[str, Any] = {}
COPY_PROGRAMS_PER_BANK = 32
COPY_WORDS = 4096  # int32 words (16 KiB) per program iteration
COPY_SHAPES = ("stripe", "chunks")
_COPY_SHAPE = "stripe"
COPY_CHUNK_PROGRAMS_PER_BANK = 8


def copy_rows_reference(source, destination, pairs):
    """Copy (src row, dst row) pairs for every bank tensor, in order."""
    for src, dst in pairs:
        for name in TENSORS:
            destination[name][dst].copy_(source[name][src])


def configure_copy(shape, programs=None, words=None):
    """Select the copy launch shape and its grid (runtime settings)."""
    global _COPY_SHAPE, COPY_PROGRAMS_PER_BANK, COPY_CHUNK_PROGRAMS_PER_BANK, COPY_WORDS
    if shape not in COPY_SHAPES:
        raise ValueError(f"Copy shape must be one of {COPY_SHAPES}")
    if programs is not None:
        if int(programs) < 1:
            raise ValueError("Copy programs per bank must be positive")
        COPY_PROGRAMS_PER_BANK = COPY_CHUNK_PROGRAMS_PER_BANK = int(programs)
    if words is not None:
        if int(words) < 32 or int(words) & (int(words) - 1):
            raise ValueError("Copy words per iteration must be a power of two >= 32")
        COPY_WORDS = int(words)
    _COPY_SHAPE = shape


def copy_rows(source, destination, src_rows, dst_rows, count):
    """Copy `count` (src, dst) row pairs of every bank tensor; device count.

    One launch of a fixed small grid reading `count` on the device, in one
    of two shapes (see COPY_SHAPES); an empty step costs a few programs.
    """
    src_device = source[TENSORS[0]].device
    if src_device.type != "cuda":
        n = int(count.reshape(-1)[0].item())
        pairs = [(int(src_rows[i]), int(dst_rows[i])) for i in range(n)]
        copy_rows_reference(source, destination, pairs)
        return
    srcs = [_word_rows(source[name]) for name in TENSORS]
    dsts = [_word_rows(destination[name]) for name in TENSORS]
    for name, src, dst in zip(TENSORS, srcs, dsts):
        if src.shape[1] != dst.shape[1]:
            raise ValueError(f"{name}: destination row size differs from the source")
    if _COPY_SHAPE == "chunks":
        grid = (len(TENSORS) * COPY_CHUNK_PROGRAMS_PER_BANK,)
        _copy_chunks_kernel()[grid](
            *srcs,
            *dsts,
            src_rows,
            dst_rows,
            count,
            *(dst.shape[1] for dst in dsts),
            *(src.stride(0) for src in srcs),
            *(dst.stride(0) for dst in dsts),
            PROGRAMS=COPY_CHUNK_PROGRAMS_PER_BANK,
            BLOCK=COPY_WORDS,
            num_warps=32,
        )
        return
    grid = (len(TENSORS) * COPY_PROGRAMS_PER_BANK,)
    _copy_kernel()[grid](
        *srcs,
        *dsts,
        src_rows,
        dst_rows,
        count,
        *(dst.shape[1] for dst in dsts),
        *(src.stride(0) for src in srcs),
        *(dst.stride(0) for dst in dsts),
        PROGRAMS=COPY_PROGRAMS_PER_BANK,
        BLOCK=COPY_WORDS,
        num_warps=4,
    )


def _word_rows(tensor):
    """View a [rows, ...] contiguous tensor as [rows, int32 words]."""
    import torch

    if not tensor.is_contiguous():
        raise ValueError("Copies require contiguous bank rows")
    rows = tensor.shape[0]
    return tensor.view(torch.uint8).reshape(rows, -1).view(torch.int32)


def _byte_rows(tensor):
    import torch

    if not tensor.is_contiguous():
        raise ValueError("Promote copies require contiguous bank rows")
    return tensor.view(torch.uint8).reshape(tensor.shape[0], -1)


def _copy_kernel():
    """Fixed grid: program (bank, stripe) streams its columns of every row."""
    if "copy" in _KERNELS:
        return _KERNELS["copy"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _stripe(
        src,
        dst,
        src_rows_ptr,
        dst_rows_ptr,
        count,
        words,
        sstride,
        dstride,
        stripe,
        PROGRAMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        for lane in range(0, count):
            src_row = tl.load(src_rows_ptr + lane).to(tl.int64)
            dst_row = tl.load(dst_rows_ptr + lane).to(tl.int64)
            src_base = src + src_row * sstride
            dst_base = dst + dst_row * dstride
            for start in range(stripe * BLOCK, words, PROGRAMS * BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                mask = offsets < words
                values = tl.load(src_base + offsets, mask=mask)
                tl.store(dst_base + offsets, values, mask=mask)

    @triton.jit
    def promote_copy(
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        dst0,
        dst1,
        dst2,
        dst3,
        dst4,
        dst5,
        src_rows_ptr,
        dst_rows_ptr,
        count_ptr,
        words0,
        words1,
        words2,
        words3,
        words4,
        words5,
        sstride0,
        sstride1,
        sstride2,
        sstride3,
        sstride4,
        sstride5,
        dstride0,
        dstride1,
        dstride2,
        dstride3,
        dstride4,
        dstride5,
        PROGRAMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        which = tl.program_id(0) // PROGRAMS
        stripe = tl.program_id(0) % PROGRAMS
        count = tl.load(count_ptr)
        if which == 0:
            _stripe(
                src0,
                dst0,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words0,
                sstride0,
                dstride0,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        elif which == 1:
            _stripe(
                src1,
                dst1,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words1,
                sstride1,
                dstride1,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        elif which == 2:
            _stripe(
                src2,
                dst2,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words2,
                sstride2,
                dstride2,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        elif which == 3:
            _stripe(
                src3,
                dst3,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words3,
                sstride3,
                dstride3,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        elif which == 4:
            _stripe(
                src4,
                dst4,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words4,
                sstride4,
                dstride4,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        else:
            _stripe(
                src5,
                dst5,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words5,
                sstride5,
                dstride5,
                stripe,
                PROGRAMS,
                BLOCK,
            )

    _KERNELS["copy"] = promote_copy
    return promote_copy


def _copy_chunks_kernel():
    """FreeToken-shaped copy: programs grid-stride over (row, chunk) pairs."""
    if "copy_chunks" in _KERNELS:
        return _KERNELS["copy_chunks"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _chunks(
        src,
        dst,
        src_rows_ptr,
        dst_rows_ptr,
        count,
        words,
        sstride,
        dstride,
        program,
        PROGRAMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        chunks_per_row = tl.cdiv(words, BLOCK)
        total = count * chunks_per_row
        for c in range(program, total, PROGRAMS):
            row = c // chunks_per_row
            chunk = c - row * chunks_per_row
            src_row = tl.load(src_rows_ptr + row).to(tl.int64)
            dst_row = tl.load(dst_rows_ptr + row).to(tl.int64)
            offsets = chunk * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < words
            values = tl.load(src + src_row * sstride + offsets, mask=mask)
            tl.store(dst + dst_row * dstride + offsets, values, mask=mask)

    @triton.jit
    def promote_copy_chunks(
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        dst0,
        dst1,
        dst2,
        dst3,
        dst4,
        dst5,
        src_rows_ptr,
        dst_rows_ptr,
        count_ptr,
        words0,
        words1,
        words2,
        words3,
        words4,
        words5,
        sstride0,
        sstride1,
        sstride2,
        sstride3,
        sstride4,
        sstride5,
        dstride0,
        dstride1,
        dstride2,
        dstride3,
        dstride4,
        dstride5,
        PROGRAMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        which = tl.program_id(0) // PROGRAMS
        program = tl.program_id(0) % PROGRAMS
        count = tl.load(count_ptr)
        if which == 0:
            _chunks(
                src0,
                dst0,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words0,
                sstride0,
                dstride0,
                program,
                PROGRAMS,
                BLOCK,
            )
        elif which == 1:
            _chunks(
                src1,
                dst1,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words1,
                sstride1,
                dstride1,
                program,
                PROGRAMS,
                BLOCK,
            )
        elif which == 2:
            _chunks(
                src2,
                dst2,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words2,
                sstride2,
                dstride2,
                program,
                PROGRAMS,
                BLOCK,
            )
        elif which == 3:
            _chunks(
                src3,
                dst3,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words3,
                sstride3,
                dstride3,
                program,
                PROGRAMS,
                BLOCK,
            )
        elif which == 4:
            _chunks(
                src4,
                dst4,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words4,
                sstride4,
                dstride4,
                program,
                PROGRAMS,
                BLOCK,
            )
        else:
            _chunks(
                src5,
                dst5,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words5,
                sstride5,
                dstride5,
                program,
                PROGRAMS,
                BLOCK,
            )

    _KERNELS["copy_chunks"] = promote_copy_chunks
    return promote_copy_chunks
