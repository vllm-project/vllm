# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""
Knorm KV Cache Compression for vllm-hust.

Knorm (Devoto et al., 2024) evicts KV cache blocks with high key L2 norms,
which tend to receive lower attention scores during decoding.

Usage:
    export VLLM_KNORM_COMPRESSION_RATIO=0.5  # keep 50% of KV cache
    export VLLM_KNORM_WARMUP_TOKENS=32       # protect first 32 tokens
    vllm-hust serve /data/models/Qwen2.5-7B-Instruct --enforce-eager
"""

from vllm.knorm.attention_backend import (
    clear_pending_norms,
    get_pending_norms,
    install_ascend_wrapper,
    store_layer_norms,
)
from vllm.knorm.config import KnormConfig
from vllm.knorm.hooks import attach_knorm_scores, collect_knorm_scores
from vllm.knorm.manager import (
    KnormFullAttentionManager,
    drain_block_scores,
    submit_block_scores,
)

__all__ = [
    "KnormConfig",
    "KnormFullAttentionManager",
    "attach_knorm_scores",
    "clear_pending_norms",
    "collect_knorm_scores",
    "drain_block_scores",
    "get_pending_norms",
    "install_ascend_wrapper",
    "store_layer_norms",
    "submit_block_scores",
]
