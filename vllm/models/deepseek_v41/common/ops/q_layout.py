# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pad the local-head Q of the mega-attention layout to the kernel head count.

``wq_b`` is permuted at load (see ``fused_layout``) so the Q GEMM writes each
token as 32 chunks of ``[h_local, 16]``. The mega-attention kernel wants 32
chunks of ``[padded_heads, 16]`` with the padding heads zeroed, and it applies
RoPE itself -- so unlike the split-KV path, nothing here rotates Q.
"""

import torch

from vllm.triton_utils import tl, triton

_HEAD_DIM = 512
_Q_CHUNK = 16
_NUM_CHUNKS = _HEAD_DIM // _Q_CHUNK


@triton.jit
def _q_pad_kernel(
    q_in,
    q_out,
    N_LOCAL: tl.constexpr,
    N_PADDED: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    offs = tl.arange(0, N_PADDED * 16)
    valid = offs < N_LOCAL * 16
    x = tl.load(
        q_in + t * (N_LOCAL * 512) + c * (N_LOCAL * 16) + offs, mask=valid, other=0.0
    )
    tl.store(q_out + t * (N_PADDED * 512) + c * (N_PADDED * 16) + offs, x)


def pad_fused_q_heads(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    """Pad fused-layout Q from ``local_heads`` to ``padded_heads`` with zeros.

    Args:
        q: ``[num_tokens, local_heads, 512]`` bf16 in the fused layout, each
            token contiguous.
        padded_heads: the kernel's head count (64 or 128), >= ``local_heads``.

    Returns:
        ``[num_tokens, padded_heads, 512]``; ``q`` itself when no padding is
        needed, since the fused layout is unchanged by a no-op pad.
    """
    num_tokens, local_heads, head_dim = q.shape
    assert head_dim == _HEAD_DIM and q.stride(2) == 1 and q.stride(1) == head_dim
    assert padded_heads >= local_heads and padded_heads % _Q_CHUNK == 0
    if padded_heads == local_heads:
        return q
    out = torch.empty(
        (num_tokens, padded_heads, head_dim), dtype=q.dtype, device=q.device
    )
    if num_tokens == 0:
        return out
    _q_pad_kernel[(num_tokens, _NUM_CHUNKS)](
        q,
        out,
        N_LOCAL=local_heads,
        N_PADDED=padded_heads,
        num_warps=4,
    )
    return out
