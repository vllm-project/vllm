# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Account for the native sparse-MLA RoPE allocation during KV planning."""

from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _pool_bytes_per_block,
)


class AsmKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    def _get_pool_bytes_per_block(self, vllm_config, kv_cache_groups) -> int:
        base = _pool_bytes_per_block(vllm_config, kv_cache_groups)
        # The separately allocated BF16 RoPE pool adds 128 B per 512 B KV row.
        return (base * 5 + 3) // 4
