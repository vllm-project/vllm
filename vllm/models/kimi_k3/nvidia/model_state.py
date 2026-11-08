# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3's model runner state: the hybrid (KDA + MLA) state, plus the lifecycle of
the specialized decode when ``decode_backend=latency_specialized`` selects it, and
the deferred RecoverSSM where the KDA layers verify with it."""

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
        if model.specialized_decode_config is not None:
            model.build_specialized_decode()
            model.specialized_decode.bind_model_state(self)
            return
        backbone = model.model
        layers = backbone.layers[backbone.start_layer : backbone.end_layer]
        if any(
            getattr(layer.self_attn, "deferred_verify", None) is not None
            for layer in layers
        ):
            # Without the specialized decode, the stock KDA layers install the
            # same deferred RecoverSSM: records by request slot and the commit
            # after sampling recording the pending replays.
            from .specialized_decode.recoverssm.state import (
                install_layers,
                install_model_state,
            )

            install_layers(layers)
            install_model_state(self)

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.prepare_kv()
