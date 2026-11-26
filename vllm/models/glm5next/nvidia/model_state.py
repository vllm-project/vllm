# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash's model runner state: the hybrid (KDA + MLA) state, plus the
lifecycle of the specialized decode when `decode_backend=latency_specialized`
selects it."""

import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class Glm5NextModelState(MambaHybridModelState):
    """Builds the specialized decode once the weights are loaded, and prepares it
    for every bound KV cache (the profiling cache, then the real one)."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        if model.specialized_decode_config is not None:
            model.build_specialized_decode()

    def prepare_attn(self, input_batch, *args, **kwargs):
        """With KDA ReplaySSM, the runner's trackers commit the verify rows: GDN's
        spec rows, which need not lead the batch. Every other row is a prefill."""
        attn_metadata = super().prepare_attn(input_batch, *args, **kwargs)
        metadata = next(
            (m for m in attn_metadata.values() if isinstance(m, GDNAttentionMetadata)),
            None,
        )
        # Autotune dummy runs and draft capture can omit Mamba metadata.
        if self._use_replayssm_trackers and metadata is not None:
            self._replayssm_query_start_loc = input_batch.query_start_loc
            spec_rows = metadata.spec_sequence_masks_cpu
            if spec_rows is None:
                spec_rows = torch.zeros(input_batch.num_reqs, dtype=torch.bool)
            async_tensor_h2d(
                ~spec_rows, out=self._is_prefilling_gpu[: spec_rows.numel()]
            )
        return attn_metadata

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.prepare_kv()
