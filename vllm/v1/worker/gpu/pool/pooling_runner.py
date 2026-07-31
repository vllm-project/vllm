# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

_SUPPORTED_TASKS: frozenset[PoolingTask] = frozenset(
    {"embed", "classify", "token_classify"}
)


class PoolingRunner:
    def __init__(self, model: nn.Module, vllm_config: VllmConfig):
        self.model = cast(VllmModelForPooling, model)
        self.model_config = vllm_config.model_config
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        model_tasks = tuple(sorted(self.model.pooler.get_supported_tasks()))
        enabled_tasks = self._get_enabled_tasks(self.model_config)
        selected_task = self.model_config.get_pooling_task(model_tasks)
        if selected_task not in enabled_tasks:
            hint = (
                "Set an explicitly supported task or VLLM_USE_V2_MODEL_RUNNER=0."
                if enabled_tasks.intersection(model_tasks)
                else "Set VLLM_USE_V2_MODEL_RUNNER=0 to use this model."
            )
            raise ValueError(
                "Model Runner V2 supports pooling tasks "
                f"{sorted(enabled_tasks)}, but this model selects "
                f"{selected_task!r} from {list(model_tasks)}. {hint}"
            )
        self.supported_tasks = frozenset(
            self.get_supported_tasks(model, self.model_config)
        )
        if not self.supported_tasks:
            raise ValueError(
                "Model Runner V2 supports pooling tasks "
                f"{sorted(enabled_tasks)}, but this model supports "
                f"{list(model_tasks)}. "
                "Set VLLM_USE_V2_MODEL_RUNNER=0 to use this model."
            )
        self.pooling_params: dict[int, PoolingParams] = {}
        self.pooling_states: dict[int, PoolingStates] = {}
        self.prompt_token_ids: dict[int, torch.Tensor] = {}

    @staticmethod
    def _get_enabled_tasks(model_config: ModelConfig) -> frozenset[PoolingTask]:
        # Token classification is validated only for unchunked encoder-only prefill.
        if model_config.attn_type == "encoder_only":
            return _SUPPORTED_TASKS
        return _SUPPORTED_TASKS - {"token_classify"}

    @staticmethod
    def get_supported_tasks(
        model: nn.Module, model_config: ModelConfig
    ) -> list[PoolingTask]:
        if not is_pooling_model(model):
            return []
        enabled_tasks = PoolingRunner._get_enabled_tasks(model_config)
        return sorted(model.pooler.get_supported_tasks() & enabled_tasks)

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
        if pooling_params.requires_token_ids:
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
            for i, (req_index, params) in enumerate(zip(req_indices, pooling_params)):
                if not params.requires_token_ids:
                    continue
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
    ) -> tuple[PoolerOutput, list[bool]]:
        hidden_states = hidden_states[: input_batch.num_tokens]
        pooling_metadata = self._get_pooling_metadata(
            input_batch, req_states, hidden_states.device
        )
        num_reqs = input_batch.num_reqs
        # Pooling has no speculative tokens, so this CPU upper bound is exact.
        seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        pooling_metadata.build_pooling_cursor(
            input_batch.num_scheduled_tokens,
            seq_lens_cpu,
            device=hidden_states.device,
            query_start_loc_gpu=input_batch.query_start_loc[: num_reqs + 1],
        )
        pooler_output = self.model.pooler(hidden_states, pooling_metadata)

        finished_mask = pooling_metadata.get_pooling_cursor().is_finished().tolist()
        return pooler_output, finished_mask

    def _dummy_pooler_run_task(
        self, hidden_states: torch.Tensor, task: PoolingTask
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        num_reqs = min(num_tokens, self.max_num_reqs)
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)
        if num_extra > 0:
            num_scheduled_tokens[-num_extra:] += 1
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
        try:
            return self.model.pooler(hidden_states, pooling_metadata)
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise
            raise RuntimeError(
                "CUDA out of memory occurred when warming up pooler "
                f"({task=}) with {num_reqs} dummy requests. Please try "
                "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                "initializing the engine."
            ) from e

    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        for task in sorted(self.supported_tasks):
            self._dummy_pooler_run_task(hidden_states, task)
