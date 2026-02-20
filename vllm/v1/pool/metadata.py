# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.utils.platform_utils import is_pin_memory_available

pin_memory = is_pin_memory_available()


@dataclass(slots=True)
class PoolingCursor:
    index: list[int]
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_scheduled_tokens_cpu: torch.Tensor

    def __getitem__(self, indices: slice):
        return PoolingCursor(
            index=self.index[indices],
            first_token_indices_gpu=self.first_token_indices_gpu[indices],
            last_token_indices_gpu=self.last_token_indices_gpu[indices],
            prompt_lens_cpu=self.prompt_lens_cpu[indices],
            seq_lens_cpu=self.seq_lens_cpu[indices],
            num_scheduled_tokens_cpu=self.num_scheduled_tokens_cpu[indices],
        )

    def is_partial_prefill(self):
        return not torch.all(self.prompt_lens_cpu == self.num_scheduled_tokens_cpu)

    def is_finished(self):
        return self.prompt_lens_cpu == self.seq_lens_cpu


class PoolingStates:
    def __init__(self):
        # for chunked prefill with ALL pooling
        self.hidden_states_cache: list[torch.Tensor] = []

    def clean(self):
        self.hidden_states_cache.clear()


@dataclass(slots=True)
class PoolingMetadata:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor  # CPU Tensor
    prompt_token_ids: torch.Tensor | None
    pooling_params: list[PoolingParams]
    pooling_states: list[PoolingStates]
    pooling_cursor: PoolingCursor | None = None
    tasks: list[PoolingTask] = field(init=False)

    def __post_init__(self) -> None:
        pooling_params = self.pooling_params

        tasks: list[PoolingTask] = [
            task
            for pooling_param in pooling_params
            if (task := pooling_param.task) is not None
        ]
        assert len(pooling_params) == len(tasks)

        self.tasks = tasks

    def __getitem__(self, indices: slice):
        return PoolingMetadata(
            prompt_lens=self.prompt_lens[indices],
            prompt_token_ids=None
            if self.prompt_token_ids is None
            else self.prompt_token_ids[indices],
            pooling_params=self.pooling_params[indices],
            pooling_states=self.pooling_states[indices],
            pooling_cursor=None
            if self.pooling_cursor is None
            else self.pooling_cursor[indices],
        )

    def get_prompt_token_ids(self) -> list[torch.Tensor]:
        prompt_token_ids = self.prompt_token_ids
        assert prompt_token_ids is not None, (
            "Please set `requires_token_ids=True` in `get_pooling_updates`"
        )

        return [prompt_token_ids[i, :num] for i, num in enumerate(self.prompt_lens)]

    def get_pooling_cursor(self) -> PoolingCursor:
        pooling_cursor = self.pooling_cursor
        assert pooling_cursor is not None, "Should call `build_pooling_cursor` first"

        return pooling_cursor

    def build_pooling_cursor(
        self,
        num_scheduled_tokens_np: np.ndarray,
        seq_lens_cpu: torch.Tensor,
        device: torch.device,
    ):
        n_seq = len(num_scheduled_tokens_np)
        prompt_lens = self.prompt_lens

        assert len(prompt_lens) == n_seq

        index = list(range(n_seq))
        num_scheduled_tokens_cpu = torch.from_numpy(num_scheduled_tokens_np)
        cumsum = torch.zeros(
            n_seq + 1, dtype=torch.int64, pin_memory=pin_memory, device="cpu"
        )
        torch.cumsum(num_scheduled_tokens_cpu, dim=0, out=cumsum[1:])
        cumsum = cumsum.to(device, non_blocking=True)
        self.pooling_cursor = PoolingCursor(
            index=index,
            first_token_indices_gpu=cumsum[:n_seq],
            last_token_indices_gpu=cumsum[1:] - 1,
            prompt_lens_cpu=prompt_lens,
            seq_lens_cpu=seq_lens_cpu,
            num_scheduled_tokens_cpu=num_scheduled_tokens_cpu,
        )
