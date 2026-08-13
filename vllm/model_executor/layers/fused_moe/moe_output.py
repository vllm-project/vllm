# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output contract between a MoE layer and a consumer that fuses its tail."""

from dataclasses import dataclass

import torch


@dataclass
class UnfinalizedMoEOutput:
    """Unfinalized output of a monolithic MoE kernel.

    Kernels that can stop after GEMM2 (the TRTLLM-Gen ``do_finalize=False`` path)
    hand back their permuted, unweighted output plus the routing weights and the
    permute map, so that the top-k reduction can be fused with whatever follows
    -- the shared-expert add and the tensor-parallel all-reduce -- instead of
    running as its own kernel.

    The buffers are consumed as-is by the fused kernels, which index
    ``gemm2_permuted`` by row: it must be densely packed at ``hidden_dim``, and
    its row count (an autotuner-dependent padded value) is never referenced.
    """

    # [num_permuted_rows, hidden_dim] permuted, unweighted GEMM2 output.
    gemm2_permuted: torch.Tensor
    # [num_tokens, top_k] routing weights, in the activation dtype -- consumers
    # index them as such, and a buffer typed wider than its contents reads as
    # garbage. These already carry routed_scaling_factor for routing methods that
    # fold it in.
    expert_weights: torch.Tensor
    # [num_tokens, top_k] int32 permute map; -1 = expert not local to this rank.
    expanded_idx_to_permuted_idx: torch.Tensor


@dataclass
class MoEOutput:
    """A MoE layer's output with its final reduction still open.

    Returned by layers whose MoE runs un-reduced (``reduce_results=False``) so
    that the consumer -- typically the next layer's RMSNorm -- can fuse the
    tensor-parallel all-reduce into itself instead of paying for a standalone
    one. Keeping the shared-expert output and the routed scale separate leaves
    that reduction the consumer's to schedule; when the routed output is still
    unfinalized, the top-k reduction is open too and can fold into the same
    kernel.

    Producers only leave the routed output unfinalized when a fused consumer can
    actually take that form -- the token ceiling and topology support are theirs
    to check -- so an ``UnfinalizedMoEOutput`` here means the fused path applies,
    and a consumer need not re-derive that.
    """

    # Un-reduced routed output, either finalized or not.
    routed: torch.Tensor | UnfinalizedMoEOutput
    # Un-reduced shared-expert output, folded in by whoever finalizes.
    shared_output: torch.Tensor | None = None
    # Applied to the routed sum before the shared add. Already folded into the
    # routing weights for methods that do so, in which case this is 1.0.
    routed_scaling_factor: float = 1.0
