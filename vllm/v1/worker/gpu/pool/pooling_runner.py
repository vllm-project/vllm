# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.tasks import PoolingTask
from vllm.v1.worker.gpu.input_batch import InputBatch


class PoolingRunner:
    def __init__(self, model: nn.Module):
        self.model = cast(VllmModelForPooling, model)

        # req_id -> list of hidden states
        self.hidden_states: dict[str, list[torch.Tensor]] = {}

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        if not is_pooling_model(self.model):
            return []
        return list(self.model.pooler.get_supported_tasks())

    def finish_request(self, req_id: str) -> None:
        self.hidden_states.pop(req_id, None)

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> list[torch.Tensor | None]:
        last_hidden_states = hidden_states[input_batch.logits_indices]
        last_hidden_states = F.normalize(last_hidden_states, p=2, dim=-1)
        return list(last_hidden_states.unbind(dim=0))

    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        F.normalize(hidden_states, p=2, dim=-1)
        return
