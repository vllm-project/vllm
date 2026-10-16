# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2's model runner state: the default state, plus the lifecycle of the
specialized decode when ``decode_backend=latency_specialized`` selects it."""

from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class DeepseekV32ModelState(DefaultModelState):
    """Builds the specialized decode once the weights are loaded, and prepares it for
    every bound KV cache (the CUDA-graph memory-profiling cache, then the real one)."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if model.specialized_decode_config is not None:
            model.build_specialized_decode()

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        if self.model.specialized_decode is not None:
            self.model.specialized_decode.prepare_kv()
