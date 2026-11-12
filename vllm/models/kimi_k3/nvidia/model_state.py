# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3's model runner state: the hybrid (KDA + MLA) state, plus the lifecycle of
the specialized decode when `decode_backend=latency_specialized` selects it, and
the pending replays of the deferred RecoverSSM where the KDA layers verify with it."""

import torch

from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class KimiK3ModelState(MambaHybridModelState):
    """Builds the specialized decode once the weights are loaded and the aux layers
    set, and prepares it for every bound KV cache (the profiling cache, then the
    real one). Without it, KDA layers with a deferred RecoverSSM verify get
    `recoverssm_pending`: per request slot, the accepted tokens of its last verify
    not yet folded into its checkpoint (the next verify replays them)."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        self.recoverssm_pending: torch.Tensor | None = None
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
            self.recoverssm_pending = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )

    def add_request(self, req_index, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        if self.recoverssm_pending is not None:
            # A new or resumed request starts from a committed (prefilled) state.
            self.recoverssm_pending[req_index].fill_(0)

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        if self.recoverssm_pending is not None:
            slots = None if for_capture else input_batch.idx_mapping
            for group_list in attn_groups:
                for group in group_list:
                    builder = group.get_metadata_builder(0)
                    if (
                        isinstance(builder, KimiK3KDAMetadataBuilder)
                        and builder.use_recoverssm
                    ):
                        builder.bind_deferred_recoverssm(self.recoverssm_pending, slots)
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.prepare_kv()
