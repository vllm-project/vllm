# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-assisted exact decode selection with persistent, reusable storage."""

import functools

import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init
from vllm.v1.worker.workspace import current_workspace_manager

CANDIDATE_CAPACITY = 8192


@functools.cache
def has_litetopk_decode() -> bool:
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
    ):
        return False
    if not hasattr(torch.ops._C, "litetopk_decode"):
        return False
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::litetopk_decode", "CUDA"
    ):
        return False
    _lazy_init()
    dg = _import_deep_gemm()
    return getattr(dg, "paged_mqa_logits_histogram_version", 0) >= 1


def get_litetopk_workspace(max_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One fixed allocation per stream and workspace lane, shared across layers.

    States always precede the full candidate region, even for smaller row views.
    Every successful selection resets its histogram and candidates and balances
    its state counters, so captured replays need no reset kernel.
    """
    manager = current_workspace_manager()
    key = ("litetopk_decode", torch.cuda.current_stream().cuda_stream)
    histogram = manager.get_persistent(
        (key, "histogram"), (max_rows, 1024), torch.int32, zero_init=True
    )
    workspace = manager.get_persistent(
        (key, "workspace"),
        (max_rows * (16 + CANDIDATE_CAPACITY * 8),),
        torch.uint8,
        zero_init=True,
    )
    return histogram, workspace


def supports_litetopk_decode(q, q_scale, weights, page_size, topk, next_n) -> bool:
    fp32 = (
        q.dtype == torch.float8_e4m3fn
        and q_scale is None
        and q.shape[1] <= 4
        and q.shape[2:] == (32, 128)
        and weights.dtype == torch.float32
        and page_size == 64
        and topk == 2048
        and 1 <= next_n <= 4
    )
    bf16 = (
        q.dtype == torch.int8
        and q_scale is not None
        and q_scale.dtype == torch.int32
        and q.shape[1:] == (1, 32, 64)
        and q_scale.shape == q.shape[:3]
        and weights.dtype in (torch.float32, torch.bfloat16)
        and page_size == 128
        and topk == 512
        and 1 <= next_n <= 6
    )
    if not (fp32 or bf16) or not has_litetopk_decode():
        return False
    return fp32 or has_litetopk_bf16()


@functools.cache
def has_litetopk_bf16() -> bool:
    dg = _import_deep_gemm()
    return all(
        callable(getattr(dg, name, None))
        for name in (
            "get_paged_mqa_logits_bf16_metadata",
            "fp4_paged_mqa_logits_bf16",
        )
    )


def get_litetopk_bf16_metadata(lengths, indices, next_n):
    dg = _import_deep_gemm()
    return dg.get_paged_mqa_logits_bf16_metadata(
        lengths,
        128,
        dg.get_num_sms(),
        indices=indices,
        tokens_per_request=next_n,
    )


def litetopk_bf16_scores(
    q,
    kv_cache,
    weights,
    lengths,
    table,
    indices,
    max_model_len,
    next_n,
    histogram,
    schedule=None,
):
    dg = _import_deep_gemm()
    if schedule is None:
        schedule = get_litetopk_bf16_metadata(lengths, indices, next_n)
    # The n=6 swizzled kernel prefers packed weights for long-context decode.
    if next_n == 6 or not getattr(dg, "paged_mqa_logits_bf16_fp32_weights", False):
        weights = weights.to(torch.bfloat16)
    return dg.fp4_paged_mqa_logits_bf16(
        q,
        kv_cache,
        weights,
        lengths,
        table,
        schedule,
        max_model_len,
        indices=indices,
        histogram=histogram,
        tokens_per_request=next_n,
    )


def litetopk_select(scores, lengths, histogram, output, workspace) -> None:
    """Write logical indices, ordered by score then lower logical index on ties.

    Output order is unspecified. Short rows are padded with -1. Histogram and
    workspace must belong exclusively to this stream until selection finishes.
    """
    torch.ops._C.litetopk_decode(
        scores, lengths, histogram, output, workspace, CANDIDATE_CAPACITY
    )
