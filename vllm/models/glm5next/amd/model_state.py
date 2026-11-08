# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash model-runner state: builds the specialized decode once the weights
are loaded and binds it to every KV cache vLLM allocates."""

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class Glm5NextModelState(MambaHybridModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        language_model = (
            model.language_model if hasattr(model, "language_model") else model
        )
        self.backbone = language_model.model
        if self.backbone.specialized_decode_config is not None:
            self.backbone.build_specialized_decode()

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.backbone.specialized_decode
        if decode is not None:
            decode.bind_caches()
