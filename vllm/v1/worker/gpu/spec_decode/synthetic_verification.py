# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed golden-schedule target compaction for synthetic verification."""

import numpy as np

import vllm.envs as envs
from vllm.config import VllmConfig


def resolve_synthetic_verify_max_drafts(vllm_config: VllmConfig) -> int | None:
    """Return the physically required draft rows for Kimi synthetic greedy."""
    if not envs.VLLM_KIMI_K3_SYNTHETIC_VERIFY_COMPACTION:
        return None

    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_dspark():
        raise ValueError(
            "Kimi synthetic verifier compaction requires DSpark speculation"
        )
    model_type = getattr(vllm_config.model_config.hf_text_config, "model_type", "")
    if model_type != "kimi_linear":
        raise ValueError(
            "VLLM_KIMI_K3_SYNTHETIC_VERIFY_COMPACTION only supports Kimi-K3"
        )
    if spec_config.rejection_sample_method != "synthetic":
        raise ValueError(
            "Kimi synthetic verifier compaction requires synthetic rejection"
        )
    if spec_config.draft_sample_method != "greedy":
        raise ValueError(
            "Kimi synthetic verifier compaction requires greedy draft sampling"
        )
    if spec_config.enable_adaptive_verification:
        raise ValueError(
            "Kimi synthetic verifier compaction is incompatible with adaptive "
            "verification"
        )

    rates = spec_config.synthetic_acceptance_rates
    if rates is None:
        raise ValueError(
            "Synthetic acceptance rates must be resolved before compaction"
        )
    max_drafts = sum(rate > 0.0 for rate in rates)
    if max_drafts == 0 or max_drafts >= spec_config.num_speculative_tokens:
        return None
    return max_drafts


def compact_synthetic_verification_counts(
    num_scheduled_tokens: np.ndarray,
    logical_num_draft_tokens: np.ndarray,
    max_drafts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compact physical target rows without changing logical draft accounting.

    Real scheduler outputs carry ``logical drafts + bonus`` rows. Synthetic
    warmup already carries ``max_drafts + bonus`` physical rows while retaining
    a logical K-sized placeholder list, so that already-compacted layout is
    accepted unchanged.
    """
    if num_scheduled_tokens.shape != logical_num_draft_tokens.shape:
        raise ValueError("Scheduled-token and draft-count arrays must have equal shape")
    if max_drafts < 0:
        raise ValueError(f"max_drafts must be non-negative, got {max_drafts}")
    physical_drafts = np.minimum(logical_num_draft_tokens, max_drafts)
    needs_compaction = logical_num_draft_tokens > max_drafts
    remappable = num_scheduled_tokens >= logical_num_draft_tokens
    already_compacted = needs_compaction & (num_scheduled_tokens == physical_drafts + 1)
    if np.any(~remappable & ~already_compacted):
        raise ValueError(
            "Draft count cannot exceed scheduled-token count unless the "
            "synthetic verifier rows are already compacted"
        )

    physical_tokens = num_scheduled_tokens.copy()
    physical_tokens[remappable] = (
        num_scheduled_tokens[remappable]
        - logical_num_draft_tokens[remappable]
        + physical_drafts[remappable]
    )
    return physical_tokens, physical_drafts


def can_compact_synthetic_verification(
    logical_num_draft_tokens: np.ndarray,
    max_drafts: int,
) -> bool:
    """Return whether any request needs compact physical verification rows.

    The scheduler sends ``-1`` token placeholders when asynchronous scheduling
    is disabled; the real proposals remain in the worker's persistent GPU
    request state. Consequently only the scheduled draft *lengths* are valid
    compaction metadata here. Inspecting placeholder values would switch a
    globally compact runner back to an incompatible full-width verifier.
    """
    if max_drafts < 0:
        raise ValueError(f"max_drafts must be non-negative, got {max_drafts}")
    if np.any(logical_num_draft_tokens < 0):
        raise ValueError("Draft counts must be non-negative")
    return logical_num_draft_tokens.size > 0 and bool(
        np.any(logical_num_draft_tokens > max_drafts)
    )
