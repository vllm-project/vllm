# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache protocol selection for EAGLE engines."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def is_lookahead_block_hashing_enabled(vllm_config: "VllmConfig") -> bool:
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or not speculative_config.use_eagle_block_drop()
        or speculative_config.disable_lookahead_block_hashes
    ):
        return False
    # The hash covers only the next token; multi-module MTP reads further ahead.
    if vllm_config.num_prefill_lookahead_tokens > 1:
        return False
    # HiSparse is not validated with lookahead hashes.
    if vllm_config.attention_config.hisparse_config is not None:
        return False
    # KV connectors keep the legacy hashes.
    if vllm_config.kv_transfer_config is not None:
        return False
    return vllm_config.cache_config.enable_prefix_caching
