# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configurable TopK selection for sparse-attention indexers."""

from __future__ import annotations

import torch

_FLASHINFER_TIE_BREAK = {"small": 1, "large": 2}


def flashinfer_tie_break_value(mode: str) -> int:
    """Return FlashInfer's integer code for a canonical index tie-break."""
    try:
        return _FLASHINFER_TIE_BREAK[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DSA TopK tie-break {mode!r}; expected 'small' or 'large'."
        ) from exc


def flashinfer_deterministic_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    tie_break: int,
    *,
    row_starts: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select a deterministic TopK with a canonical boundary tie-break.

    Returned indices use the same coordinate system as ``logits``. For ragged
    prefill rows, ``row_starts`` therefore serves as both the selection-window
    start and the output offset.
    """
    if tie_break not in _FLASHINFER_TIE_BREAK.values():
        raise ValueError("FlashInfer deterministic TopK requires a tie-break")

    from flashinfer import top_k_ragged_transform

    lengths = lengths.reshape(-1).to(dtype=torch.int32).contiguous()
    if row_starts is None:
        row_starts = torch.zeros_like(lengths)
        selection_starts = None
    else:
        row_starts = row_starts.reshape(-1).to(dtype=torch.int32).contiguous()
        selection_starts = row_starts

    indices = top_k_ragged_transform(
        logits.contiguous(),
        row_starts,
        lengths,
        k,
        deterministic=True,
        tie_break=tie_break,
        dsa_graph_safe=True,
        row_starts=selection_starts,
    )
    if out is not None:
        out.copy_(indices)
        return out
    return indices
