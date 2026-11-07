"""Ephemeral state carried between ``execute_model`` and ``sample_tokens``."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TargetTokenScoringState:
    """Carries the compact-wave contract from ``execute_model`` to
    ``sample_tokens``.

    The compact ``[B, K]`` logits themselves ride the existing
    ``ExecuteModelState.logits`` field (so downstream NaN-counting and
    bookkeeping see a real tensor). This state carries only the metadata needed
    to turn those compact logits into a correct ``SamplerOutput``: the ordered
    candidate ids, where column ``j`` corresponds to ``target_token_ids[j]``
    (never to vocab id ``j``).

    Attributes:
        target_token_ids: Ordered candidate ids shared by the whole wave.
        target_ids_tensor: Device tensor of ``target_token_ids`` for gather.
    """

    target_token_ids: list[int]
    target_ids_tensor: torch.Tensor

    @classmethod
    def from_ids(
        cls,
        target_token_ids: list[int],
        device: torch.device,
    ) -> TargetTokenScoringState:
        """Build state from an ordered candidate-id list and a device."""
        return cls(
            target_token_ids=list(target_token_ids),
            target_ids_tensor=torch.as_tensor(
                target_token_ids, dtype=torch.long, device=device
            ),
        )
