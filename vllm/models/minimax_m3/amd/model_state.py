# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 model-runner state and specialized-decode cache lifecycle."""

import weakref
from typing import Any

from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class MiniMaxM3ModelState(DefaultModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        language_model = (
            model.language_model if hasattr(model, "language_model") else model
        )
        self.backbone = language_model.model
        self._decode_finalizer = None
        if self.backbone.specialized_decode_config is not None:
            self.backbone.build_specialized_decode()
            decode = self.backbone.specialized_decode
            assert decode is not None
            self._decode_finalizer = weakref.finalize(self, decode.close)
            finalizer: Any = self._decode_finalizer
            finalizer.atexit = False

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.backbone.specialized_decode
        if decode is not None:
            decode.bind_caches()

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
        super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
        decode = self.backbone.specialized_decode
        if decode is not None:
            decode.executor.output_step()

    def close(self) -> None:
        if self._decode_finalizer is not None:
            self._decode_finalizer()
