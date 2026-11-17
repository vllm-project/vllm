# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's model runner state: the default state, plus the lifecycle of the
specialized decode when `decode_backend=latency_specialized` selects it, and
its vocab-sharded speculative verification."""

from typing import Any

import torch

from vllm.models.muse_spark.nvidia.specialized_decode.runtime.draft_fp4_mlp import (
    reset_calibrations,
)
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class MuseSparkModelState(DefaultModelState):
    """Builds the specialized decode once the weights are loaded and the aux layers
    set, and prepares it for every bound KV cache (the profiling cache, then the
    real one).

    With the specialized decode, where it applies, verification is
    vocab-sharded (`VocabShardedVerifier`): `prepare_inputs` notes each step's
    batch, `compute_sample_logits` returns the candidate packets of a batch
    that verifies there instead of its logits, and the rejection sampler of
    `custom_sampler` verifies them; every other batch takes the stock path.

    The first request after a dummy sampler run (profiling and warm-up end
    with one) resets the draft MLP amax calibration, which the dummy passes
    fed.
    """

    _dummy_runs_observed = False

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        self.verifier = None
        if model.specialized_decode_config is not None:
            model.build_specialized_decode()

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.prepare_kv()

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        if self.language_model.specialized_decode_config is None:
            return None
        from vllm.models.muse_spark.nvidia.specialized_decode.sample.vocab_sharded import (  # noqa: E501
            VocabShardedRejectionSampler,
            VocabShardedVerifier,
        )

        self.verifier = VocabShardedVerifier.build(
            self.vllm_config, self.language_model, sampler, self.device
        )
        if self.verifier is None:
            return None
        return sampler, VocabShardedRejectionSampler(self.verifier)

    def add_request(self, req_index, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        if self._dummy_runs_observed:
            self._dummy_runs_observed = False
            reset_calibrations()
        if self.verifier is not None:
            self.verifier.add_request(req_index, new_req_data.sampling_params)

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        if self.verifier is not None:
            self.verifier.apply_staged_writes()

    def prepare_inputs(self, input_batch, req_states):
        if self.verifier is not None:
            self.verifier.plan(input_batch)
        return super().prepare_inputs(input_batch, req_states)

    def dummy_sampler_run(self, sampler: Any, hidden_states: torch.Tensor) -> bool:
        self._dummy_runs_observed = True
        if self.verifier is not None:
            self.verifier.warm_up(hidden_states)
        return False

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.verifier is not None:
            packets = self.verifier.logits(hidden_states)
            if packets is not None:
                return packets
        return super().compute_sample_logits(hidden_states)
