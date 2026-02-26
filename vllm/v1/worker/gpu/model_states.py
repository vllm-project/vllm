# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.mrope_utils import MRopeState
from vllm.v1.worker.gpu.states import RequestState


class ModelState:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device

        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens

        self.uses_mrope = self.model_config.uses_mrope
        if self.uses_mrope:
            self.mrope_state = MRopeState(
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_tokens,
                max_model_len=self.max_model_len,
                device=self.device,
            )

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.uses_mrope:
            # Pre-compute M-RoPE positions for prefill.
            assert new_req_data.prefill_token_ids is not None
            self.mrope_state.init_prefill_mrope_positions(
                req_index,
                self.model,  # type: ignore
                new_req_data.prefill_token_ids,
                mm_features=new_req_data.mm_features,
            )

    def apply_staged_writes(self) -> None:
        if self.uses_mrope:
            self.mrope_state.apply_staged_writes()

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        if not self.uses_mrope:
            # Common case (1D positions).
            return {}

        # Prepare M-RoPE positions.
        self.mrope_state.prepare_mrope_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        mrope_positions = self.mrope_state.mrope_positions[
            :, : input_batch.num_tokens_after_padding
        ]
        return {"positions": mrope_positions}

    def prepare_dummy_inputs(
        self, num_reqs: int, num_tokens: int
    ) -> dict[str, torch.Tensor | None]:
        if not self.uses_mrope:
            return {}
        mrope_positions = self.mrope_state.mrope_positions[:, :num_tokens]
        return {"positions": mrope_positions}
