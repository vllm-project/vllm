# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 model-runner state and specialized-decode cache lifecycle."""

from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class MiniMaxM3ModelState(DefaultModelState):
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

    def close(self) -> None:
        decode = self.backbone.specialized_decode
        if decode is not None:
            decode.close()
