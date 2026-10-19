# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's model runner state on ROCm: the shared state plus the
specialized decode's cache checks."""

from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables


class DeepseekV41ROCMModelState(DeepseekV41ModelState):
    """Checks the specialized decode's caches, the backbone's and the DSpark
    draft's, for every cache the runner binds: the throwaway one memory
    profiling allocates, then the real one."""

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        from vllm.models.deepseek_v41.amd.dspark import bind_draft_decode_caches

        super().initialize_kv_cache(kv_cache_config, block_tables)
        if self.model.specialized_decode is not None:
            self.model.specialized_decode.bind_caches()
        bind_draft_decode_caches()
