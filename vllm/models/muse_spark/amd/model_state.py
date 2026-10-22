# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's model runner state on ROCm: vLLM's default state, plus the
lifecycle of the specialized decode when ``--monokernel`` (or
``decode_backend=latency_specialized``) selects it. With it off the hooks below
do nothing."""

from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class MuseSparkModelState(DefaultModelState):
    """Builds the specialized decode once the weights are loaded, prepares it for
    every bound KV cache (the CUDA-graph memory-profiling cache, then the real
    one) and closes its startup once the decoder graphs are captured."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        if getattr(model, "specialized_decode_config", None) is not None:
            # stage-2 checks, peer memory and every compile, before profiling
            model.build_specialized_decode(vllm_config)

    def _decode(self):
        return getattr(self.language_model, "specialized_decode", None)

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self._decode()
        if decode is not None:
            # stage 3 and the startup self-test, at every bind; MRV2's
            # persistent block tables pin the kernels' metadata addresses
            decode.prepare_kv(kv_cache_config, block_tables)

    def capture_inner_cudagraphs(self, *args, **kwargs):
        super().capture_inner_cudagraphs(*args, **kwargs)
        decode = self._decode()
        if decode is not None:
            # the capture warm-ups' health vote, then the serving wait budget
            decode.after_capture()
