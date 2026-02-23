# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.model_executor.models import VllmModelForPooling, is_pooling_model
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.jsontree import json_map_leaves
from vllm.v1.outputs import ModelRunnerOutput, PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates


class PoolingRunner:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        vocab_size: int,
        req_states,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.device = device
        self.vocab_size = vocab_size
        self.req_states = req_states

        self.pooling_params: dict[str, PoolingParams] = {}
        self.pooling_states: dict[str, PoolingStates] = {}
        self._model: VllmModelForPooling | None = None

    def set_model(self, model) -> None:
        self._model = cast(VllmModelForPooling, model)

    def finish_request(self, req_id: str) -> None:
        self.pooling_params.pop(req_id, None)
        self.pooling_states.pop(req_id, None)

    def add_request(self, req_id: str, pooling_params: PoolingParams) -> None:
        assert self._model is not None
        task = pooling_params.task
        assert task is not None, "You did not set `task` in the API"
        to_update = self._model.pooler.get_pooling_updates(task)
        to_update.apply(pooling_params)
        self.pooling_params[req_id] = pooling_params
        self.pooling_states[req_id] = PoolingStates()

    def get_supported_pooling_tasks(self, model) -> list[PoolingTask]:
        if not is_pooling_model(model):
            return []
        model = cast(VllmModelForPooling, model)
        return list(model.pooler.get_supported_tasks())

    def get_pooling_metadata(self, input_batch) -> PoolingMetadata:
        pooling_params = [self.pooling_params[rid] for rid in input_batch.req_ids]
        pooling_states = [self.pooling_states[rid] for rid in input_batch.req_ids]

        prompt_lens_np = self.req_states.prompt_len.np[input_batch.idx_mapping_np]
        prompt_lens = torch.from_numpy(prompt_lens_np)

        prompt_token_ids = None
        if any(p.requires_token_ids for p in pooling_params):
            max_prompt_len = int(prompt_lens_np.max()) if len(prompt_lens_np) else 0
            if max_prompt_len > 0:
                req_indices = torch.from_numpy(input_batch.idx_mapping_np).to(
                    device=self.device, dtype=torch.int64
                )
                all_token_ids = self.req_states.all_token_ids.gpu
                prompt_token_ids = all_token_ids.index_select(0, req_indices)[
                    :, :max_prompt_len
                ].clone()
                prompt_lens_gpu = torch.from_numpy(prompt_lens_np).to(
                    device=self.device
                )
                pad_mask = torch.arange(max_prompt_len, device=self.device).unsqueeze(
                    0
                ) >= prompt_lens_gpu.unsqueeze(1)
                prompt_token_ids[pad_mask] = self.vocab_size

        return PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            pooling_params=pooling_params,
            pooling_states=pooling_states,
        )

    def update_progress(self, input_batch) -> None:
        if input_batch.num_reqs == 0:
            return
        idx_mapping_np = input_batch.idx_mapping_np
        num_scheduled = input_batch.num_scheduled_tokens
        req_indices = torch.from_numpy(idx_mapping_np).to(
            device=self.device, dtype=torch.int64
        )
        delta = torch.from_numpy(num_scheduled).to(
            device=self.device, dtype=self.req_states.num_computed_tokens.gpu.dtype
        )
        self.req_states.num_computed_tokens.gpu.index_add_(0, req_indices, delta)

        computed_prefill = self.req_states.num_computed_prefill_tokens
        computed_prefill[idx_mapping_np] += num_scheduled
        np.minimum(
            computed_prefill, self.req_states.prefill_len.np, out=computed_prefill
        )

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch,
        kv_connector_output,
    ) -> ModelRunnerOutput:
        assert self._model is not None
        num_reqs = input_batch.num_reqs
        hidden_states = hidden_states[: input_batch.num_tokens]
        seq_lens_cpu = input_batch.seq_lens[:num_reqs].cpu()

        pooling_metadata = self.get_pooling_metadata(input_batch)
        pooling_metadata.build_pooling_cursor(
            input_batch.num_scheduled_tokens,
            seq_lens_cpu=seq_lens_cpu,
            device=hidden_states.device,
        )

        raw_pooler_output: PoolerOutput = self._model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        raw_pooler_output = json_map_leaves(
            lambda x: None if x is None else x.to("cpu", non_blocking=True),
            raw_pooler_output,
        )
        model_runner_output.pooler_output = [
            out if include else None
            for out, include in zip(raw_pooler_output, finished_mask)
        ]
        torch.cuda.synchronize()
        return model_runner_output

    def dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        assert self._model is not None
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs

        req_num_tokens = num_tokens // num_reqs
        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(self.model_config)
        to_update = self._model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for _ in range(num_reqs)],
        )
        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )

        try:
            return self._model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            raise

    def dummy_pooler_run(self, hidden_states: torch.Tensor) -> PoolerOutput:
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            return torch.tensor([])

        supported_pooling_tasks = self.get_supported_pooling_tasks(self._model)
        if not supported_pooling_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support "
                "any pooling tasks. See "
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        output_size: dict[PoolingTask, float] = {}
        for task in supported_pooling_tasks:
            output = self.dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self.dummy_pooler_run_task(hidden_states, max_task)
