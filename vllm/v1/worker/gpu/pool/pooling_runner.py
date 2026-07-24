# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

_SUPPORTED_TASKS: frozenset[PoolingTask] = frozenset({"embed", "classify"})


class PoolingRunner:
    def __init__(self, model: nn.Module, vllm_config: VllmConfig):
        self.model = cast(VllmModelForPooling, model)
        self.model_config = vllm_config.model_config
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.supported_tasks = frozenset(self.get_supported_tasks(model))
        if not self.supported_tasks:
            model_tasks = sorted(self.model.pooler.get_supported_tasks())
            raise ValueError(
                "Model Runner V2 supports only sequence-level pooling tasks "
                f"{sorted(_SUPPORTED_TASKS)}, but this model supports {model_tasks}. "
                "Set VLLM_USE_V2_MODEL_RUNNER=0 to use this model."
            )
        self.pooling_params: dict[int, PoolingParams] = {}
        self.pooling_states: dict[int, PoolingStates] = {}
        self.prompt_token_ids: dict[int, torch.Tensor] = {}

    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        if not is_pooling_model(model):
            return []
        return sorted(model.pooler.get_supported_tasks() & _SUPPORTED_TASKS)

    def add_request(
        self,
        req_index: int,
        pooling_params: PoolingParams,
        prompt_token_ids: list[int],
    ) -> None:
        task = pooling_params.task
        if task not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task: {task!r}. "
                f"Supported tasks: {sorted(self.supported_tasks)}"
            )
        self.model.pooler.get_pooling_updates(task).apply(pooling_params)
        self.pooling_params[req_index] = pooling_params
        self.pooling_states[req_index] = PoolingStates()
        self.prompt_token_ids[req_index] = torch.tensor(
            prompt_token_ids, dtype=torch.int64
        )

    def remove_request(self, req_index: int) -> None:
        self.pooling_params.pop(req_index, None)
        if state := self.pooling_states.pop(req_index, None):
            state.clean()
        self.prompt_token_ids.pop(req_index, None)

    def _get_pooling_metadata(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        device: torch.device,
    ) -> PoolingMetadata:
        req_indices = input_batch.idx_mapping_np.tolist()
        pooling_params = [self.pooling_params[i] for i in req_indices]
        pooling_states = [self.pooling_states[i] for i in req_indices]
        prompt_lens = torch.from_numpy(
            req_states.prompt_len.np[input_batch.idx_mapping_np].copy()
        )

        prompt_token_ids_cpu = None
        prompt_token_ids = None
        if any(params.requires_token_ids for params in pooling_params):
            max_prompt_len = int(prompt_lens.max())
            prompt_token_ids_cpu = torch.zeros(
                (input_batch.num_reqs, max_prompt_len),
                dtype=torch.int64,
                pin_memory=PIN_MEMORY,
            )
            for i, req_index in enumerate(req_indices):
                token_ids = self.prompt_token_ids[req_index]
                prompt_token_ids_cpu[i, : token_ids.numel()] = token_ids
            prompt_token_ids = prompt_token_ids_cpu.to(device, non_blocking=True)

        return PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_cpu=prompt_token_ids_cpu,
            pooling_params=pooling_params,
            pooling_states=pooling_states,
        )

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[PoolerOutput, torch.Tensor]:
        hidden_states = hidden_states[: input_batch.num_tokens]
        pooling_metadata = self._get_pooling_metadata(
            input_batch, req_states, hidden_states.device
        )
        num_reqs = input_batch.num_reqs
        seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        pooling_metadata.build_pooling_cursor(
            input_batch.num_scheduled_tokens,
            seq_lens_cpu,
            device=hidden_states.device,
            query_start_loc_gpu=input_batch.query_start_loc[: num_reqs + 1],
        )
        pooler_output = self.model.pooler(hidden_states, pooling_metadata)

        prompt_len = req_states.prompt_len.gpu[input_batch.idx_mapping]
        is_valid = input_batch.seq_lens[:num_reqs] == prompt_len
        return pooler_output, is_valid

    def _dummy_pooler_run_task(
        self, hidden_states: torch.Tensor, task: PoolingTask
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        num_reqs = min(num_tokens, self.max_num_reqs)
        num_scheduled_tokens = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_scheduled_tokens[-1] += num_tokens % num_reqs
        prompt_lens = torch.from_numpy(num_scheduled_tokens)

        pooling_params = PoolingParams(task=task)
        pooling_params.verify(self.model_config)
        self.model.pooler.get_pooling_updates(task).apply(pooling_params)
        prompt_token_ids = None
        if pooling_params.requires_token_ids:
            prompt_token_ids = torch.zeros(
                (num_reqs, int(prompt_lens.max())),
                dtype=torch.int64,
                device=hidden_states.device,
            )
        pooling_metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_cpu=None
            if prompt_token_ids is None
            else prompt_token_ids.cpu(),
            pooling_params=[pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for _ in range(num_reqs)],
        )
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens,
            seq_lens_cpu=prompt_lens,
            device=hidden_states.device,
        )
        return self.model.pooler(hidden_states, pooling_metadata)

    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> PoolerOutput:
        output_sizes: dict[PoolingTask, int] = {}
        for task in sorted(self.supported_tasks):
            output = self._dummy_pooler_run_task(hidden_states, task)
            outputs = output if isinstance(output, list) else output.unbind()
            output_sizes[task] = sum(o.nbytes for o in outputs if o is not None)
            del output

        max_task = max(output_sizes, key=output_sizes.get)  # type: ignore[arg-type]
        return self._dummy_pooler_run_task(hidden_states, max_task)
