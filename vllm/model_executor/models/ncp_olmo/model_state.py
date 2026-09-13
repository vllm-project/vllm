# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""V2 runner state adapter for request-scoped NCP-ArchPreview HLM tensors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from .state import ConceptRequestStateStore


class NCPOlmoModelState(DefaultModelState):
    """Connect NCP-ArchPreview HLM state to the V2 runner request lifecycle."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        request_states = getattr(model, "request_states", None)
        if not isinstance(request_states, ConceptRequestStateStore):
            raise TypeError(
                "NCP-ArchPreview model is missing its HLM request state store"
            )
        self.request_states = request_states

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        self.request_states.add_request(
            new_req_data.req_id,
            computed_tokens=int(new_req_data.num_computed_tokens),
        )
        try:
            super().add_request(req_index, new_req_data)
        except BaseException:
            self.request_states.remove_request(new_req_data.req_id)
            raise

    def remove_request(self, req_id: str) -> None:
        self.request_states.remove_request(req_id)
        super().remove_request(req_id)

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        req_ids = tuple(str(req_id) for req_id in input_batch.req_ids)
        admitted = req_states.req_id_to_index
        is_dummy_batch = bool(req_ids) and all(
            req_id not in admitted for req_id in req_ids
        )
        if is_dummy_batch:
            model_inputs["request_segments"] = ()
            return model_inputs

        missing_runner_requests = [
            req_id for req_id in req_ids if req_id not in admitted
        ]
        if missing_runner_requests:
            raise RuntimeError(
                "NCP-ArchPreview received a mixed real/dummy V2 batch: "
                f"{missing_runner_requests!r} are absent from RequestState"
            )
        model_inputs["request_segments"] = self.request_states.resolve_segments(
            input_batch
        )
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        model_inputs["request_segments"] = ()
        return model_inputs
