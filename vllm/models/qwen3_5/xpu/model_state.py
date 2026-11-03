# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5's model runner state on XPU with the mono decode (``--monokernel``):
the hybrid (GatedDeltaNet + attention) state, plus the lifecycle of the
executor (``specialized_decode.runtime.decode``)."""

import torch

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class Qwen3_5XpuModelState(MambaHybridModelState):
    """Builds the mono decode once the weights are loaded, binds it to every KV
    cache vLLM allocates (the memory-profiling cache, then the real one), and
    runs the executor's per-step check with the logits."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        self.decode = model.build_specialized_decode()

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        self.decode.bind_caches()

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = super().compute_sample_logits(hidden_states)
        self.decode.after_step()
        return logits
