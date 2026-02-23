# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import torch

from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch


class PoolingRunner:
    def __init__(self):
        # req_id -> list of hidden states
        self.hidden_states: dict[str, list[torch.Tensor]] = {}

    def set_model(self, model) -> None:
        self.model = cast(VllmModelForPooling, model)

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
        num_reqs = input_batch.num_reqs
        query_start_loc_list = input_batch.query_start_loc_np.tolist()

        out: list[torch.Tensor | None] = []
        for i in range(num_reqs):
            last = query_start_loc_list[i + 1] - 1
            out.append(hidden_states[last])
        return out

    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> PoolerOutput:
        return
