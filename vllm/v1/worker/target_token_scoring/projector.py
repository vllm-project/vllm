"""Compact LM Head: index_select K candidate weight rows -> ``[B, K]``.

This is the device-side win: instead of ``[B, H] @ [V, H]^T -> [B, V]`` and a
tensor-parallel all-gather, we gather the K candidate weight rows once
(``index_select`` -> ``[K, H]``) and do a compact ``F.linear`` -> ``[B, K]``.
The selected weights are cached keyed on a robust signature so re-selection is
not a per-wave hot path.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .admission import AdmissionDecision
from .state import TargetTokenScoringState

if TYPE_CHECKING:
    from torch import nn


def _signature(
    target_ids: tuple[int, ...],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple:
    """Cache key. ``_version`` catches in-place writes that leave
    ``data_ptr`` unchanged but change the content."""
    return (
        target_ids,
        weight.data_ptr(),
        weight._version,
        tuple(weight.shape),
        weight.dtype,
        str(weight.device),
        None if bias is None else (
            bias.data_ptr(), bias._version, tuple(bias.shape), bias.dtype,
            str(bias.device),
        ),
    )


class CompactLMHeadCache:
    """Per-runner cache of index-selected candidate weight rows.

    Keyed by ``id(lm_head)`` with a signature that includes the target-id
    tuple, ``data_ptr`` and ``_version`` (so an in-place weight mutation
    invalidates the entry even though ``data_ptr`` is unchanged). Entries are
    evicted when the owning model is garbage-collected, via a ``weakref``
    finalizer installed on first miss.

    Owned by the runner rather than held as module globals so the cache's
    lifetime is bounded by the worker's lifetime and there is no hidden
    cross-model mutable state.
    """

    def __init__(self) -> None:
        self._cache: dict[int, tuple] = {}
        self._tracked: set[int] = set()

    def get(
        self,
        lm_head: nn.Module,
        target_ids: tuple[int, ...],
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return cached ``(selected_weight, selected_bias)``, selecting and
        caching on miss."""
        key = id(lm_head)
        sig = _signature(target_ids, weight, bias)
        entry = self._cache.get(key)
        if entry is not None and entry[0] == sig:
            return entry[1], entry[2]

        idx = torch.as_tensor(target_ids, dtype=torch.long, device=weight.device)
        sel_w = weight.index_select(0, idx).contiguous()
        sel_b = None if bias is None else bias.index_select(0, idx).contiguous()
        self._cache[key] = (sig, sel_w, sel_b)

        if key not in self._tracked:
            self._tracked.add(key)
            # Drop the cache entry once the owning model is gone. ``pop`` is
            # idempotent, so a second finalize (if any) is a harmless no-op.
            weakref.finalize(
                lm_head, lambda k=key: self._cache.pop(k, None)
            )
        return sel_w, sel_b


def _apply_layer_postprocess(
    logits_processor: object, logits: torch.Tensor
) -> torch.Tensor:
    """Apply the row-wise transforms the LM Head's own ``LogitsProcessor``
    would apply (soft_cap, scale) to the compact ``[B, K]`` logits.

    These transforms are per-row and order-independent, so they are correct on
    the compact tensor. The full ``LogitsProcessor`` is **not** invoked because
    its other branches assume a full-vocab width (``allowed_token_ids`` /
    ``bad_words`` masks index vocab ids); admission rejects every sampler-mask
    semantic precisely so only ``soft_cap``/``scale`` can be active here. This
    mirrors the head's postprocess rather than reusing the call site so the
    compact tensor never reaches a vocab-shaped code path.
    """
    soft_cap = getattr(logits_processor, "soft_cap", None)
    if soft_cap is not None:
        logits = torch.tanh(logits / soft_cap) * soft_cap
    scale = getattr(logits_processor, "scale", 1.0)
    if scale != 1.0:
        logits = logits * scale
    return logits


def project_target_token_logits(
    model: nn.Module,
    hidden_states: torch.Tensor,
    decision: AdmissionDecision,
    cache: CompactLMHeadCache,
) -> tuple[torch.Tensor, TargetTokenScoringState] | None:
    """Project ``hidden_states`` onto the K candidate weight rows.

    Args:
        model: The vLLM model exposing ``lm_head`` and ``logits_processor``.
        hidden_states: ``[B, H]`` hidden states at the scoring position.
        decision: A positive (``ok=True``) admission decision.
        cache: Per-runner cache of index-selected weight rows.

    Returns:
        ``(compact_logits[B, K], state)`` or ``None`` if the LM Head turns
        out to be non-compact at projection time (caller falls back native).
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return None
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    # Most models pass no embedding_bias; support it if the head exposes one.
    bias = getattr(lm_head, "bias", None)

    target_ids = tuple(decision.target_token_ids)  # type: ignore[arg-type]
    sel_w, sel_b = cache.get(lm_head, target_ids, weight, bias)

    logits = F.linear(hidden_states, sel_w, sel_b)  # [B, K]

    logits_processor = getattr(model, "logits_processor", None)
    if logits_processor is not None:
        logits = _apply_layer_postprocess(logits_processor, logits)

    state = TargetTokenScoringState.from_ids(list(target_ids), weight.device)
    return logits, state
