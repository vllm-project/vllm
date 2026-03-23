# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused softmax + entropy Triton kernel for should_stop logic.

Computes H = -sum(p * log(p)) where p = softmax(logits), in a single
memory pass to reduce bandwidth and kernel launch overhead.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _softmax_entropy_kernel(
    logits_ptr,
    entropy_ptr,
    stride_batch: tl.constexpr,
    stride_vocab: tl.constexpr,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    """Compute entropy = -sum(softmax(logits) * log(softmax(logits))) per row.

    Uses numerically stable formula:
      max_val = max(logits)
      Z = sum(exp(logits - max_val))
      S = sum(exp(logits - max_val) * (logits - max_val))
      entropy = log(Z) - S / Z
    """
    row_idx = tl.program_id(0)

    # First pass: find max (use scalar accumulators, not [1]-shaped blocks)
    max_val = float("-inf")
    for block_start in range(0, n_cols, BLOCK_N):
        col_offs = block_start + tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        ptr = logits_ptr + row_idx * stride_batch + col_offs * stride_vocab
        x = tl.load(ptr, mask=mask, other=float("-inf"))
        cur_max = tl.max(x)
        max_val = tl.maximum(max_val, cur_max)

    # Second pass: compute Z and S
    Z = 0.0
    S = 0.0
    for block_start in range(0, n_cols, BLOCK_N):
        col_offs = block_start + tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        ptr = logits_ptr + row_idx * stride_batch + col_offs * stride_vocab
        x = tl.load(ptr, mask=mask, other=0.0)
        x_shifted = x - max_val
        exp_x = tl.exp(x_shifted)
        exp_x = tl.where(mask, exp_x, 0.0)
        Z += tl.sum(exp_x)
        S += tl.sum(exp_x * x_shifted)

    entropy = tl.log(Z) - S / Z
    out_ptr = entropy_ptr + row_idx
    tl.store(out_ptr, entropy)


def softmax_entropy(logits: "torch.Tensor") -> "torch.Tensor":
    """Fused softmax + entropy: H = -sum(p*log(p)) where p=softmax(logits).

    Args:
        logits: [batch_size, vocab_size] float32

    Returns:
        entropy: [batch_size, 1] float32
    """
    from vllm.triton_utils.importing import HAS_TRITON

    if not HAS_TRITON:
        return _softmax_entropy_pytorch(logits)

    logits = logits.contiguous()
    batch_size, vocab_size = logits.shape
    entropy = logits.new_empty((batch_size, 1))

    BLOCK_N = 4096
    grid = (batch_size,)
    _softmax_entropy_kernel[grid](
        logits,
        entropy,
        stride_batch=logits.stride(0),
        stride_vocab=logits.stride(1),
        n_cols=vocab_size,
        BLOCK_N=BLOCK_N,
    )
    return entropy


def _softmax_entropy_pytorch(logits: "torch.Tensor") -> "torch.Tensor":
    """PyTorch fallback when Triton is unavailable."""
    import torch

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.special.entr(probs).sum(dim=-1, keepdim=True)
