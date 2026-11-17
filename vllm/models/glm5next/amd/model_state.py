# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash model-runner state on ROCm: builds the specialized decode once the
weights are loaded and binds it to every KV cache vLLM allocates."""

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class Glm5NextModelState(MambaHybridModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        if model.specialized_decode_config is not None:
            if model.specialized_decode is not None:
                raise ValueError("--monokernel: reloading weights is not supported")
            model.build_specialized_decode()

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.prepare_kv()
