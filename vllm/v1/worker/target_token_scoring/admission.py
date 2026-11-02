"""Wave-level admission gate for the target-token-scoring fast path.

The compact path produces a single ``[B, K]`` tensor whose column ``j``
corresponds to ``target_token_ids[j]`` (not vocab id ``j``). Because that
shape is shared by the whole wave, eligibility must be decided **once per
wave, before** ``compute_logits`` runs, and any single failure must downgrade
the **entire** wave to native. Per-request mixing would let the generic
sampler mistake a compact column index for a vocab id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

    from vllm.config import ModelConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


@dataclass
class AdmissionDecision:
    """Result of evaluating one wave.

    Attributes:
        ok: Whether the whole wave may take the compact fast path.
        target_token_ids: The shared, ordered candidate ids (only when ok).
        reason: Human-readable fallback reason (only when not ok).
    """

    ok: bool
    target_token_ids: list[int] | None = None
    reason: str | None = None


def _req_sampling_params(
    req_id: str,
    requests: dict[str, CachedRequestState],
) -> SamplingParams | None:
    state = requests.get(req_id)
    return None if state is None else state.sampling_params


def _request_eligible(sp: SamplingParams) -> str | None:
    """Return a fallback reason if ``sp`` is ineligible, else ``None``."""
    ids = sp.logprob_token_ids
    if not ids:
        return "request has no logprob_token_ids"
    if sp.max_tokens != 1:
        return f"max_tokens={sp.max_tokens} != 1"
    # temperature == 0 is vLLM's greedy contract.
    if sp.temperature != 0:
        return f"temperature={sp.temperature} != 0 (non-greedy)"
    if sp.n != 1:
        return f"n={sp.n} != 1"
    if sp.prompt_logprobs is not None:
        return "prompt_logprobs requested"
    if sp.logit_bias:
        return "logit_bias set"
    if sp.allowed_token_ids:
        return "allowed_token_ids set (sampler-mask semantic, not sparse head)"
    if sp.bad_words:
        return "bad_words set"
    if getattr(sp, "guided_decoding", None) is not None:
        return "guided/structured output requested"
    if getattr(sp, "logits_processors", None):
        return "request-level logits_processors attached"
    norm = getattr(sp, "target_token_scoring_normalization", "full_vocab")
    if norm != "target_set":
        return (
            f"normalization={norm} != target_set "
            "(request did not opt into target-set semantics)"
        )
    return None


def _lm_head_eligible(lm_head: nn.Module) -> str | None:
    """Return a fallback reason if ``lm_head`` cannot be indexed compactly."""
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.ndim != 2:
        return "lm_head has no dense 2D weight"
    if getattr(lm_head, "tp_size", 1) != 1:
        return "lm_head is TP vocab-sharded"
    quant_method = getattr(lm_head, "quant_method", None)
    if quant_method is not None:
        # UnquantizedEmbeddingMethod is the only safe quant method here: its
        # rows are plain contiguous weights that index_select can gather.
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        if not isinstance(quant_method, UnquantizedEmbeddingMethod):
            return "lm_head is quantized/packed"
    return None


def evaluate_wave_admission(
    model_config: ModelConfig,
    input_batch: InputBatch,
    requests: dict[str, CachedRequestState],
    lm_head: nn.Module,
    *,
    spec_decode_metadata: object | None,
) -> AdmissionDecision:
    """Decide whether the current wave may take the compact fast path.

    Args:
        model_config: Engine model config (carries the engine flag).
        input_batch: The runner input batch (``req_ids`` / ``num_reqs``).
        requests: Per-req cached state (carries ``sampling_params``).
        lm_head: The model's LM head module.
        spec_decode_metadata: Wave-level spec-decode metadata; non-None
            disqualifies the wave.

    Returns:
        An :class:`AdmissionDecision`. Native fallback is the default; the
        compact path is only taken when every gate passes for every request.
    """
    if not getattr(model_config, "target_token_scoring", False):
        return AdmissionDecision(ok=False, reason="engine flag disabled")

    if spec_decode_metadata is not None:
        return AdmissionDecision(ok=False, reason="speculative decoding active")
    if getattr(model_config, "logits_processors", None):
        return AdmissionDecision(ok=False, reason="model-level logits_processors")

    head_reason = _lm_head_eligible(lm_head)
    if head_reason is not None:
        return AdmissionDecision(ok=False, reason=f"lm_head: {head_reason}")

    num_reqs = input_batch.num_reqs
    if num_reqs == 0:
        return AdmissionDecision(ok=False, reason="empty wave")

    req_ids = input_batch.req_ids
    shared_ids: list[int] | None = None
    for i in range(num_reqs):
        req_id = req_ids[i]
        sp = _req_sampling_params(req_id, requests)
        if sp is None:
            return AdmissionDecision(ok=False, reason=f"req {req_id}: no params")
        reason = _request_eligible(sp)
        if reason is not None:
            return AdmissionDecision(ok=False, reason=f"req {req_id}: {reason}")
        ids = list(sp.logprob_token_ids)
        if shared_ids is None:
            shared_ids = ids
        elif ids != shared_ids:
            return AdmissionDecision(
                ok=False, reason="candidate ids differ across requests"
            )

    assert shared_ids is not None
    # All candidates must lie within the local (unsharded) weight rows.
    num_rows = lm_head.weight.shape[0]
    if any(tid < 0 or tid >= num_rows for tid in shared_ids):
        return AdmissionDecision(ok=False, reason="candidate id out of range")

    return AdmissionDecision(
        ok=True,
        target_token_ids=shared_ids,
    )
