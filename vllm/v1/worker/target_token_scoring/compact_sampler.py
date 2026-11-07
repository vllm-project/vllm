"""Compact sampler: argmax over ``[B, K]`` + remap to target token ids.

The compact logits' column ``j`` is ``target_token_ids[j]``, **not** vocab id
``j``. Feeding ``[B, K]`` to the generic sampler would return ``0..K-1``, which
is a valid shape and dtype but the wrong tokens. So the compact path does its
own argmax, remaps to the real candidate ids, and builds a ``LogprobsTensors``
shaped exactly like the native ``gather_specific_token_logprobs`` output
(sampled in column 0, requested ids in columns 1..K) so downstream
``LogprobsProcessor`` / ``RequestOutput`` need no special-casing.

Normalization is fixed to ``target_set`` (log-softmax over the K candidates);
``full_vocab`` is rejected at admission.
"""

from __future__ import annotations

import torch

from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from .state import TargetTokenScoringState


def _target_set_logprobs(logits: torch.Tensor) -> torch.Tensor:
    """Stable log-softmax over the K candidate dimension (FP32)."""
    f = logits.float()
    m = f.amax(dim=-1, keepdim=True)
    # Rows that are all -inf would produce NaN; clamp the max to 0 so the
    # subtraction is finite. Such rows are surfaced via num_nans in
    # bookkeeping rather than silently producing a wrong argmax.
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    shifted = f - m
    return shifted - torch.logsumexp(shifted, dim=-1, keepdim=True)


def compact_sample(
    compact_logits: torch.Tensor,
    state: TargetTokenScoringState,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
    """Turn compact ``[B, K]`` logits into a native-shaped ``SamplerOutput``.

    Args:
        compact_logits: ``[B, K]`` projected logits (column j = candidate j).
        state: The wave's target-token-scoring contract (candidate ids).
        sampling_metadata: Wave sampling metadata (unused beyond presence;
            eligibility guarantees greedy, single-token, no processors).

    Returns:
        A ``SamplerOutput`` whose ``logprobs_tensors`` matches the shape the
        native ``gather_specific_token_logprobs`` would produce for the same
        ``logprob_token_ids``.
    """
    del sampling_metadata  # eligibility guarantees greedy/no-proc; unused

    num_reqs, k = compact_logits.shape
    target_ids = state.target_ids_tensor  # [K], on compact_logits.device
    device = compact_logits.device

    # Fail-closed on NaN: detect per-row NaN so it is surfaced via num_nans
    # in bookkeeping rather than producing a silently-wrong stable argmax.
    row_has_nan = torch.isnan(compact_logits).any(dim=-1)  # [B]

    # argmax over K candidates, remapped to real token ids (never raw j).
    selected_cols = compact_logits.argmax(dim=-1)  # [B]
    sampled = target_ids.gather(0, selected_cols)  # [B]
    # NaN rows cannot be trusted; keep them in the wave so bookkeeping counts
    # them, but do not emit a fabricated id. Use the argmax-of-zero fallback
    # (column 0) so the shape is valid; num_nans flags the request.
    sampled = torch.where(
        row_has_nan, target_ids.gather(0, torch.zeros_like(selected_cols)), sampled
    )
    sampled_token_ids = sampled.unsqueeze(-1)  # [B, 1]

    # Build the [B, K+1] logprobs table: col 0 = sampled, cols 1..K = candidates
    # (matching native gather_specific_token_logprobs layout).
    logprobs = _target_set_logprobs(compact_logits)  # [B, K] (FP32)
    sampled_logprob = logprobs.gather(-1, selected_cols.unsqueeze(-1))  # [B,1]
    full_logprobs = torch.cat([sampled_logprob, logprobs], dim=-1)  # [B, K+1]

    token_ids_table = torch.empty(
        num_reqs, k + 1, dtype=torch.int32, device=device
    )
    token_ids_table[:, 0] = sampled
    token_ids_table[:, 1:] = target_ids.unsqueeze(0).to(torch.int32)

    # Rank of the sampled token within the candidate set (0 for the argmax).
    ranks = torch.sum(
        (logprobs > sampled_logprob).to(torch.int64), dim=-1
    )  # [B]

    # Mask NaN rows so they do not emit misleading finite logprobs.
    if row_has_nan.any():
        full_logprobs = full_logprobs.masked_fill(
            row_has_nan.unsqueeze(-1), float("nan")
        )

    logprobs_tensors = LogprobsTensors(
        logprob_token_ids=token_ids_table,
        logprobs=full_logprobs,
        selected_token_ranks=ranks,
    )
    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=logprobs_tensors,
    )
