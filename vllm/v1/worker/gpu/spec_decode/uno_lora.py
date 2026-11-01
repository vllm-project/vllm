# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA mappings for shared-model Uno drafting.

Uno's draft runs the target model with one base-weighted seed row and ``k - 1``
rows through Uno's adapter per request. ``UnoSpeculator`` installs these
mappings through the worker LoRA manager's public ``set_active_adapters``.
"""

from __future__ import annotations


def draft_lora_mapping(
    num_input_tokens: int,
    batch_size: int,
    k: int,
    uno_lora_id: int,
) -> tuple[int, ...]:
    """Return the draft token mapping, including graph padding.

    Each request has one base weighted seed row and ``k - 1`` rows using the
    Uno adapter.  Rows after ``batch_size * k`` are graph padding and stay on
    the base model.  The prompt mapping is intentionally separate: callers
    should use ``base_prompt_mapping`` because draft logits are produced by the
    base vocabulary head.
    """
    if num_input_tokens < 0 or batch_size < 1 or k < 1:
        raise ValueError("Uno draft mapping dimensions must be positive")
    actual = batch_size * k
    if num_input_tokens < actual:
        raise ValueError("draft mapping cannot truncate real query rows")
    per_request = (0,) + (int(uno_lora_id),) * (k - 1)
    return per_request * batch_size + (0,) * (num_input_tokens - actual)


def base_prompt_mapping(num_requests: int) -> tuple[int, ...]:
    """Return an all base prompt/sampler mapping."""
    if num_requests < 0:
        raise ValueError("base prompt mapping length must be nonnegative")
    return (0,) * num_requests


__all__ = [
    "base_prompt_mapping",
    "draft_lora_mapping",
]
