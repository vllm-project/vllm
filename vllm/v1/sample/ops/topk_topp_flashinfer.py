# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined top-k/top-p masking on top of FlashInfer candidate selection.

FlashInfer selects the ``top_k_max`` highest logits per row. A fused Triton
kernel then applies the per-row top-k cutoff, filters by top-p and scatters the
survivors back into a dense masked-logit tensor.

Eligibility follows FlashInfer's top-k-first gate: every row must have an active
per-row top-k within ``[1, MAX_TOP_K]`` and the vocabulary must be large enough.
Callers pass the host-side bounds of top-k, so no device readback is needed.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import has_flashinfer

# FlashInfer's top-k-first gate, applied to the batch bounds instead of a
# scalar k: flashinfer/sampling.py:1579-1594, `_top_k_first_fast_path_applicable`
# (v0.7.2.post1, commit beabf5004585). The thresholds are empirical there.
MAX_TOP_K = 256
MIN_VOCAB_SIZE = 65536


def can_apply_top_k_top_p_flashinfer(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    top_k_bounds: tuple[int, int] | None,
) -> bool:
    """Whether the FlashInfer top-k-first path supports this batch.

    Args:
        logits: [batch_size, vocab_size] logits to mask.
        k: [batch_size] int32 per-row top-k values, or None to disable top-k.
        top_k_bounds: Host-side (min, max) of ``k`` over all rows, or None when
            the caller does not track top-k on the host.

    Returns:
        True if every row has an active top-k in range and FlashInfer is usable.

    """
    if k is None or top_k_bounds is None:
        return False
    if not current_platform.is_cuda() or not has_flashinfer():
        return False
    top_k_min, top_k_max = top_k_bounds
    vocab_size = logits.shape[-1]
    return (
        k.shape == (logits.shape[0],)
        and 0 < top_k_min <= top_k_max <= MAX_TOP_K
        and top_k_max < vocab_size
        and vocab_size >= MIN_VOCAB_SIZE
    )


def apply_top_k_top_p_flashinfer(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor | None,
    top_k_bounds: tuple[int, int],
) -> torch.Tensor:
    """Return dense masked logits for a batch that passed the eligibility gate.

    Args:
        logits: [batch_size, vocab_size] logits to mask.
        k: [batch_size] int32 per-row top-k values, all within ``top_k_bounds``.
        p: [batch_size] float32 per-row top-p values, or None to disable top-p.
        top_k_bounds: Host-side (min, max) of ``k`` over all rows.

    Returns:
        A new [batch_size, vocab_size] tensor with masked logits set to -inf.

    """
    import flashinfer

    top_k_min, top_k_max = top_k_bounds
    # The scatter ignores candidate order unless top-p or per-row k needs it.
    values, indices = flashinfer.top_k(
        logits.contiguous(),
        top_k_max,
        sorted=p is not None or top_k_min != top_k_max,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=logits.shape[-1] % 4 != 0,
    )
    masked = torch.full_like(logits, float("-inf"))
    if p is None and top_k_min == top_k_max:
        return masked.scatter_(1, indices, values)
    _candidates_to_logits_kernel[(logits.shape[0],)](
        values,
        indices,
        k,
        p,
        masked,
        values.stride(0),
        values.stride(1),
        indices.stride(0),
        indices.stride(1),
        masked.stride(0),
        masked.stride(1),
        k.stride(0),
        p.stride(0) if p is not None else 0,
        top_k_max,
        p is not None,
        top_k_min == top_k_max,
        triton.next_power_of_2(top_k_max),
        num_warps=4,
    )
    return masked


@triton.jit
def _candidates_to_logits_kernel(
    values_ptr,
    indices_ptr,
    k_ptr,
    p_ptr,
    output_ptr,
    value_row_stride,
    value_col_stride,
    index_row_stride,
    index_col_stride,
    output_row_stride,
    output_col_stride,
    k_stride,
    p_stride,
    TOP_K_MAX: tl.constexpr,
    HAS_P: tl.constexpr,
    UNIFORM_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Filter [batch, top_k_max] FP32 candidates and scatter retained logits."""
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    rank = TOP_K_MAX - 1 - offset
    valid = offset < TOP_K_MAX
    value = tl.load(
        values_ptr + row * value_row_stride + rank * value_col_stride,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    index = tl.load(
        indices_ptr + row * index_row_stride + rank * index_col_stride,
        mask=valid,
        other=0,
    )
    if not UNIFORM_K:
        row_k = tl.load(k_ptr + row * k_stride)
        value = tl.where(rank < row_k, value, -float("inf"))
    keep = valid
    if HAS_P:
        weight = tl.exp(value - tl.max(value, 0))
        probability = weight / tl.sum(weight, 0)
        cumulative = tl.cumsum(probability, 0)
        p = tl.load(p_ptr + row * p_stride)
        keep = valid & (~(cumulative <= 1.0 - p) | (offset == TOP_K_MAX - 1))
    tl.store(
        output_ptr + row * output_row_stride + index * output_col_stride,
        value,
        mask=keep,
    )
