# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined top-k/top-p masking on top of FlashInfer candidate selection.

FlashInfer selects the ``top_k_max`` highest logits per row. A fused Triton
kernel then orders the candidates, applies the per-row top-k cutoff, filters by
top-p and scatters the survivors back into the logits, masked in place.

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
    """Mask a batch that passed the eligibility gate, in place.

    Args:
        logits: [batch_size, vocab_size] logits to mask.
        k: [batch_size] int32 per-row top-k values, all within ``top_k_bounds``.
        p: [batch_size] float32 per-row top-p values, or None to disable top-p.
        top_k_bounds: Host-side (min, max) of ``k`` over all rows.

    Returns:
        ``logits``, with masked logits set to -inf.

    """
    import flashinfer

    top_k_min, top_k_max = top_k_bounds
    # The tie-break alone fixes the selected set, so FlashInfer can skip sorting
    # and deterministic ordering. The kernel below orders the candidates itself.
    values, indices = flashinfer.top_k(
        logits.contiguous(),
        top_k_max,
        sorted=False,
        deterministic=False,
        tie_break=1,
        dsa_graph_safe=logits.shape[-1] % 4 != 0,
    )
    logits.fill_(float("-inf"))
    if p is None and top_k_min == top_k_max:
        return logits.scatter_(1, indices, values)
    _candidates_to_logits_kernel[(logits.shape[0],)](
        values,
        indices,
        k,
        p,
        logits,
        values.stride(0),
        values.stride(1),
        indices.stride(0),
        indices.stride(1),
        logits.stride(0),
        logits.stride(1),
        k.stride(0),
        p.stride(0) if p is not None else 0,
        top_k_max,
        p is not None,
        triton.next_power_of_2(top_k_max),
        num_warps=4,
    )
    return logits


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
    BLOCK: tl.constexpr,
):
    """Order [batch, top_k_max] FP32 candidates by (logit desc, index asc), cut
    them at the row's k and p, and scatter the retained logits."""
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    valid = offset < TOP_K_MAX
    value = tl.load(
        values_ptr + row * value_row_stride + offset * value_col_stride,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    index = tl.load(
        indices_ptr + row * index_row_stride + offset * index_col_stride,
        mask=valid,
        other=0,
    ).to(tl.int64)
    # Ascending sort key: the order-preserving uint32 of the logit, inverted into
    # a signed high word, with the index in the low word.
    bits = tl.where(value == 0, 0.0, value).to(tl.int32, bitcast=True)
    bits = bits.to(tl.int64) & 0xFFFFFFFF
    ordered = tl.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)
    key = ((0x7FFFFFFF - ordered) << 32) | index
    key = tl.sort(tl.where(valid, key, 0x7FFFFFFFFFFFFFFF), 0)
    index = key & 0xFFFFFFFF
    ordered = 0x7FFFFFFF - (key >> 32)
    bits = tl.where(ordered >= 0x80000000, ordered - 0x80000000, 0xFFFFFFFF - ordered)
    value = bits.to(tl.uint32).to(tl.float32, bitcast=True)
    keep = offset < tl.load(k_ptr + row * k_stride)
    if HAS_P:
        top = tl.max(tl.where(keep, value, -float("inf")), 0)
        weight = tl.where(keep, tl.exp(value - top), 0.0)
        # Keep the smallest prefix whose mass reaches p.
        before = tl.cumsum(weight, 0) - weight
        p = tl.load(p_ptr + row * p_stride)
        keep = keep & (before < p * tl.sum(weight, 0))
    tl.store(
        output_ptr + row * output_row_stride + index * output_col_stride,
        value,
        mask=keep,
    )
