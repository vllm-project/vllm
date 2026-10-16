# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3's model runner state: the hybrid (KDA + MLA) state, plus the lifecycle of
the specialized decode when ``--specialized-decode`` selects it."""

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class KimiK3ModelState(MambaHybridModelState):
    """Builds the specialized decode once the weights are loaded and the aux layers
    set, and prepares it for every bound KV cache (the profiling cache, then the
    real one)."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        if model.decode_spec is not None:
            model.build_specialized_decode()

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.bind_caches()
