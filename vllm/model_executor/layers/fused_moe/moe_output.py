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


def convert_flashinfer_moe_output(
    flashinfer_output: torch.Tensor | list[torch.Tensor],
    *,
    do_finalize: bool,
    num_tokens: int,
    top_k: int,
    finalized_output: torch.Tensor | None = None,
) -> torch.Tensor | UnfinalizedMoEOutput:
    """Normalize the two FlashInfer TRTLLM MoE return layouts.

    Args:
        flashinfer_output: Tensor returned by the FlashInfer BF16 wrapper's
            legacy finalized path, or its mode-dependent tensor list.
        do_finalize: Whether FlashInfer ran its top-k finalize step.
        num_tokens: Number of input tokens.
        top_k: Number of routed experts per token.
        finalized_output: Optional destination passed to FlashInfer's ``output``
            argument.

    Returns:
        A finalized tensor or the structured deferred-finalize output.

    Raises:
        ValueError: If FlashInfer returns an unexpected layout.
    """

    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if do_finalize:
        if isinstance(flashinfer_output, torch.Tensor):
            returned_output = flashinfer_output
        elif len(flashinfer_output) == 1 and isinstance(
            flashinfer_output[0], torch.Tensor
        ):
            returned_output = flashinfer_output[0]
        else:
            raise ValueError(
                "Finalized FlashInfer MoE output must contain exactly one tensor."
            )
        if finalized_output is not None and (
            returned_output.shape != finalized_output.shape
            or returned_output.dtype != finalized_output.dtype
            or returned_output.device != finalized_output.device
            or returned_output.data_ptr() != finalized_output.data_ptr()
        ):
            raise ValueError(
                "Finalized FlashInfer MoE output must alias the provided destination."
            )
        return returned_output

    if finalized_output is not None:
        raise ValueError("Deferred FlashInfer MoE output cannot have a destination.")
    if isinstance(flashinfer_output, torch.Tensor) or len(flashinfer_output) != 3:
        raise ValueError(
            "Deferred FlashInfer MoE output must contain GEMM2 output, routing "
            "weights, and a permutation map."
        )
    gemm2_permuted, expert_weights, expanded_idx = flashinfer_output
    if gemm2_permuted.ndim != 2:
        raise ValueError("Deferred FlashInfer GEMM2 output must be rank-2.")
    if expanded_idx.dtype != torch.int32:
        raise ValueError("Deferred FlashInfer permutation map must use int32.")
    expected_routes = num_tokens * top_k
    if expert_weights.numel() != expected_routes:
        raise ValueError(
            "Deferred FlashInfer routing weights must contain "
            f"num_tokens * top_k ({expected_routes}) elements."
        )
    if expanded_idx.numel() != expected_routes:
        raise ValueError(
            "Deferred FlashInfer permutation map must contain "
            f"num_tokens * top_k ({expected_routes}) elements."
        )
    if not (gemm2_permuted.device == expert_weights.device == expanded_idx.device):
        raise ValueError("Deferred FlashInfer outputs must share a device.")
    if not all(
        output.is_contiguous()
        for output in (gemm2_permuted, expert_weights, expanded_idx)
    ):
        raise ValueError("Deferred FlashInfer outputs must be contiguous.")
    return UnfinalizedMoEOutput(
        gemm2_permuted=gemm2_permuted,
        expert_weights=expert_weights.view(num_tokens, top_k),
        expanded_idx_to_permuted_idx=expanded_idx.view(num_tokens, top_k),
    )


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
