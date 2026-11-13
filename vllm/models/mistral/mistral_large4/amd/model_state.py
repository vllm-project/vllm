# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4 model-runner state: builds the specialized decode after load."""

import weakref
from typing import Any

from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class MistralLarge4ModelState(DefaultModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        language_model = getattr(model, "language_model", model)
        self.backbone = language_model.model
        self._decode_finalizer = None
        if self.backbone.specialized_decode_config is not None:
            # after load, before memory profiling: the repacked weights count
            self.backbone.build_specialized_decode(vllm_config)
            decode = self.backbone.specialized_decode
            assert decode is not None
            self._decode_finalizer = weakref.finalize(self, decode.close)
            finalizer: Any = self._decode_finalizer
            finalizer.atexit = False

    def close(self) -> None:
        if self._decode_finalizer is not None:
            self._decode_finalizer()
