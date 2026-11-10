# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model-runner state of ``Qwen3_5ForCausalLMBase``: vLLM's hybrid state,
plus the lifecycle of the specialized decode when ``--monokernel`` selects
it."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.models.qwen3_5.amd.specialized_decode.hooks import SpecializedDecodeMixin
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

if TYPE_CHECKING:
    from vllm.models.qwen3_5.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )


class Qwen3_5ModelState(MambaHybridModelState):
    """With ``--monokernel``, builds the specialized decode once the weights
    are loaded, and checks it against every bound KV cache (the CUDA-graph
    memory-profiling cache, then the real one)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.specialized_decode: SpecializedDecode | None = None
        if (
            isinstance(model, SpecializedDecodeMixin)
            and model.specialized_decode_config is not None
        ):
            model.build_specialized_decode()
            self.specialized_decode = model.specialized_decode

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        if self.specialized_decode is not None:
            self.specialized_decode.bind_caches()


__all__ = ["Qwen3_5ModelState"]
