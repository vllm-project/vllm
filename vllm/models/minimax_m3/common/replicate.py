# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated attention for MiniMax-M3 (VLLM_MINIMAX_M3_REPLICATE_ATTN).

Layout decisions shared by the decoder layers and the attention metadata builders,
kept free of kernel imports so they can be checked on their own.
"""

from typing import Any

import vllm.envs as envs


def attention_tp_size(tensor_parallel_size: int) -> int:
    """TP degree attention layers are built with: 1 when every rank holds all heads."""
    return 1 if envs.VLLM_MINIMAX_M3_REPLICATE_ATTN else tensor_parallel_size


def unsupported_layout(parallel_config: Any) -> str | None:
    """Why replicated attention cannot run with this layout, or None.

    Expert and context parallelism turn the TP-synced Triton autotune off, so
    replicated attention could drift across ranks there."""
    reasons = []
    if parallel_config.enable_expert_parallel:
        reasons.append("expert parallelism")
    if (
        parallel_config.decode_context_parallel_size > 1
        or parallel_config.prefill_context_parallel_size > 1
    ):
        reasons.append("decode/prefill context parallelism")
    return ", ".join(reasons) or None
