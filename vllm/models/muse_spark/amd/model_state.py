# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark on ROCm: Model Runner V2 state."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


class MuseSparkModelState(DefaultModelState):
    """Passes the sampled rows to steps that may run token-sharded, so the
    final all-gather sends only those rows.

    Every row is passed when a request in the batch reads all of them (prompt
    logprobs). The persistent buffer keeps the input address fixed between a
    CUDA graph's capture and its replays.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        spec = vllm_config.speculative_config
        # A drafter fed the final hidden states, or a microbatched step, needs
        # every row.
        self.min_sharded_tokens = (
            model.model.token_shard_min_tokens
            if (spec is None or spec.uses_aux_hidden_states)
            and not vllm_config.parallel_config.use_ubatching
            else 0
        )
        self.logits_indices = torch.zeros(
            self.max_num_tokens if self.min_sharded_tokens else 0,
            dtype=torch.int64,
            device=device,
        )
        self.all_rows_req_ids: set[str] = set()

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        params = new_req_data.sampling_params
        if params is not None and (
            params.prompt_logprobs is not None
            or params.prompt_logprob_token_ids is not None
        ):
            self.all_rows_req_ids.add(new_req_data.req_id)

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.all_rows_req_ids.discard(req_id)

    def _may_shard(self, num_tokens: int) -> bool:
        return 0 < self.min_sharded_tokens <= num_tokens

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        num_tokens = input_batch.num_tokens_after_padding
        if not self._may_shard(num_tokens):
            return model_inputs
        if self.all_rows_req_ids.isdisjoint(input_batch.req_ids):
            rows = self.logits_indices[: input_batch.logits_indices.shape[0]]
            rows.copy_(input_batch.logits_indices)
        else:
            rows = torch.arange(num_tokens, out=self.logits_indices[:num_tokens])
        model_inputs["logits_indices"] = rows
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self._may_shard(num_tokens):
            model_inputs["logits_indices"] = torch.arange(
                num_tokens, out=self.logits_indices[:num_tokens]
            )
        return model_inputs
