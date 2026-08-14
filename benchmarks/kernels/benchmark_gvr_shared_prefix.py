# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark exact shared-prefix scheduling for DeepSeek indexer logits."""

import argparse
from functools import partial
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton
from vllm.utils import deep_gemm

_HEADS = 32
_HEAD_DIM = 128
_PAGE_SIZE = 64


@triton.jit
def _indexer_tail_kernel(
    q,
    kv,
    scales,
    weights,
    block_table,
    seq_lens,
    logits,
    q_stride,
    weight_stride,
    block_table_stride,
    logits_stride,
    page_stride,
    scale_page_stride,
    common_tokens: tl.constexpr,
    token_block: tl.constexpr,
):
    row = tl.program_id(0)
    token_block_id = tl.program_id(1)
    head = tl.arange(0, 32)
    dim = tl.arange(0, 128)
    token = token_block_id * token_block + tl.arange(0, token_block)

    q_values = tl.load(q + row * q_stride + head[:, None] * 128 + dim[None, :])
    page = tl.load(block_table + row * block_table_stride + common_tokens // 64)
    k_values = tl.load(kv + page * page_stride + token[:, None] * 128 + dim[None, :])
    accum = tl.dot(q_values, tl.trans(k_values))
    head_weights = tl.load(weights + row * weight_stride + head)
    reduced = tl.sum(tl.maximum(accum, 0.0) * head_weights[:, None], axis=0)
    scale = tl.load(scales + page * scale_page_stride + token)
    seq_len = tl.load(seq_lens + row)
    valid = common_tokens + token < seq_len
    tl.store(
        logits + row * logits_stride + common_tokens + token,
        reduced * scale,
        mask=valid,
    )


@triton.jit
def _copy_tail_kernel(
    prefix,
    tail,
    tail_lens,
    prefix_stride,
    tail_stride,
    common_tokens: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.program_id(1) * 256 + tl.arange(0, 256)
    valid = token < tl.load(tail_lens + row)
    values = tl.load(tail + row * tail_stride + token, mask=valid)
    tl.store(
        prefix + row * prefix_stride + common_tokens + token,
        values,
        mask=valid,
    )


def _time(launch, repeats: int) -> float:
    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def _full_logits(
    dg,
    q,
    kv_pages,
    weights,
    seq_lens,
    block_table,
    metadata,
    indices,
):
    return dg.fp8_fp4_paged_mqa_logits(
        (q, None),
        kv_pages,
        weights,
        seq_lens,
        block_table,
        metadata,
        100032,
        False,
        torch.float32,
        indices,
    )


def _shared_prefix_logits(
    dg,
    q,
    kv_pages,
    kv_values,
    kv_scales,
    weights,
    seq_lens,
    shared_lens,
    block_table,
    metadata,
    indices,
    batch,
    common_tokens,
    token_block,
):
    output = _full_logits(
        dg,
        q,
        kv_pages,
        weights,
        shared_lens,
        block_table,
        metadata,
        indices,
    )
    _indexer_tail_kernel[(batch, triton.cdiv(64, token_block))](
        q,
        kv_values,
        kv_scales,
        weights,
        block_table,
        seq_lens,
        output,
        q.stride(0),
        weights.stride(0),
        block_table.stride(0),
        output.stride(0),
        kv_values.stride(0),
        kv_scales.stride(0),
        common_tokens=common_tokens,
        token_block=token_block,
    )
    return output


def _shared_prefix_segmented_logits(
    dg,
    q,
    kv_pages,
    weights,
    shared_lens,
    tail_lens,
    block_table,
    tail_block_table,
    shared_metadata,
    tail_metadata,
    grouped_indices,
    unique_indices,
    tail_width,
):
    prefix = _full_logits(
        dg,
        q,
        kv_pages,
        weights,
        shared_lens,
        block_table,
        shared_metadata,
        grouped_indices,
    )
    tail = dg.fp8_fp4_paged_mqa_logits(
        (q, None),
        kv_pages,
        weights,
        tail_lens,
        tail_block_table,
        tail_metadata,
        tail_width,
        False,
        torch.float32,
        unique_indices,
    )
    return prefix, tail


def _shared_prefix_stitched_logits(
    dg,
    q,
    kv_pages,
    weights,
    shared_lens,
    tail_lens,
    block_table,
    tail_block_table,
    shared_metadata,
    tail_metadata,
    grouped_indices,
    unique_indices,
    batch,
    common_tokens,
    tail_width,
):
    prefix, tail = _shared_prefix_segmented_logits(
        dg,
        q,
        kv_pages,
        weights,
        shared_lens,
        tail_lens,
        block_table,
        tail_block_table,
        shared_metadata,
        tail_metadata,
        grouped_indices,
        unique_indices,
        tail_width,
    )
    _copy_tail_kernel[(batch, triton.cdiv(tail_width, 256))](
        prefix,
        tail,
        tail_lens,
        prefix.stride(0),
        tail.stride(0),
        common_tokens=common_tokens,
    )
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--batches", default="8,32,128,1024")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--tail-block", type=int, default=16)
    parser.add_argument("--private-tail", action="store_true")
    args = parser.parse_args()

    dg = deep_gemm._import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is required")
    dg.set_pdl(True)

    captured = torch.load(args.capture, map_location="cuda", weights_only=True)
    kv_pages = captured["kv_pages"].contiguous()
    page_bytes = kv_pages[0].numel()
    kv_values = torch.as_strided(
        kv_pages.view(torch.float8_e4m3fn),
        size=(kv_pages.shape[0], _PAGE_SIZE, _HEAD_DIM),
        stride=(page_bytes, _HEAD_DIM, 1),
    )
    kv_scales = torch.as_strided(
        kv_pages.view(torch.float32),
        size=(kv_pages.shape[0], _PAGE_SIZE),
        stride=(page_bytes // 4, 1),
        storage_offset=_PAGE_SIZE * _HEAD_DIM // 4,
    )
    seq_len = int(captured["seq_len"])
    common_tokens = seq_len // _PAGE_SIZE * _PAGE_SIZE
    num_pages = kv_pages.shape[0]

    for batch in map(int, args.batches.split(",")):
        q = captured["q_values"].unsqueeze(0).repeat(batch, 1, 1, 1)
        weights = captured["weights"].repeat(batch, 1)
        seq_lens = torch.full((batch, 1), seq_len, dtype=torch.int32, device="cuda")
        shared_lens = torch.full_like(seq_lens, common_tokens)
        tail_lens = seq_lens - common_tokens
        block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").repeat(
            batch, 1
        )
        if args.private_tail:
            block_table[:, common_tokens // _PAGE_SIZE] = (
                torch.arange(batch, dtype=torch.int32, device="cuda") * 104729
            ) % (num_pages - 1)
        unique_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
        grouped_indices = torch.zeros_like(unique_indices)
        tail_block_table = block_table[:, common_tokens // _PAGE_SIZE :].contiguous()
        full_meta = dg.get_paged_mqa_logits_metadata(
            seq_lens, _PAGE_SIZE, dg.get_num_sms(), indices=unique_indices
        )
        shared_meta = dg.get_paged_mqa_logits_metadata(
            shared_lens, _PAGE_SIZE, dg.get_num_sms(), indices=grouped_indices
        )
        tail_meta = dg.get_paged_mqa_logits_metadata(
            tail_lens, _PAGE_SIZE, dg.get_num_sms(), indices=unique_indices
        )

        full_launch = partial(
            _full_logits,
            dg,
            q,
            kv_pages,
            weights,
            seq_lens,
            block_table,
            full_meta,
            unique_indices,
        )
        grouped_launch = partial(
            _shared_prefix_logits,
            dg,
            q,
            kv_pages,
            kv_values,
            kv_scales,
            weights,
            seq_lens,
            shared_lens,
            block_table,
            shared_meta,
            grouped_indices,
            batch,
            common_tokens,
            args.tail_block,
        )
        segmented_launch = partial(
            _shared_prefix_segmented_logits,
            dg,
            q,
            kv_pages,
            weights,
            shared_lens,
            tail_lens,
            block_table,
            tail_block_table,
            shared_meta,
            tail_meta,
            grouped_indices,
            unique_indices,
            64,
        )
        stitched_launch = partial(
            _shared_prefix_stitched_logits,
            dg,
            q,
            kv_pages,
            weights,
            shared_lens,
            tail_lens,
            block_table,
            tail_block_table,
            shared_meta,
            tail_meta,
            grouped_indices,
            unique_indices,
            batch,
            common_tokens,
            64,
        )

        reference = full_launch()
        actual = grouped_launch()
        torch.cuda.synchronize()
        error = (actual[:, :seq_len] - reference[:, :seq_len]).abs()
        prefix, tail = segmented_launch()
        segmented = torch.cat(
            (prefix[:, :common_tokens], tail[:, : seq_len - common_tokens]), dim=1
        )
        segmented_error = (segmented - reference[:, :seq_len]).abs()
        stitched = stitched_launch()
        stitched_error = (stitched[:, :seq_len] - reference[:, :seq_len]).abs()
        full_us = _time(full_launch, args.repeats)
        grouped_us = _time(grouped_launch, args.repeats)
        segmented_us = _time(segmented_launch, args.repeats)
        stitched_us = _time(stitched_launch, args.repeats)
        print(
            f"batch={batch} full={full_us:.3f} us "
            f"shared+tail={grouped_us:.3f} us speedup={full_us / grouped_us:.3f}x "
            f"segmented={segmented_us:.3f} us "
            f"segmented_speedup={full_us / segmented_us:.3f}x "
            f"stitched={stitched_us:.3f} us "
            f"stitched_speedup={full_us / stitched_us:.3f}x "
            f"triton_error={error.max():.3g} "
            f"segmented_error={segmented_error.max():.3g} "
            f"stitched_error={stitched_error.max():.3g}"
        )


if __name__ == "__main__":
    main()
