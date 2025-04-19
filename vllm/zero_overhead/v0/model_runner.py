# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from vllm.sequence import SequenceGroupMetadata
from vllm.utils import async_tensor_h2d
from vllm.worker.model_runner import ModelInputForGPU, ModelInputForGPUBuilder
from vllm.zero_overhead.v0.sampler import get_last_sampler
from vllm.zero_overhead.v0.update_input import UpdateInputTokens


class ZeroOverheadModelInputForGpuBuilder(ModelInputForGPUBuilder):

    def __init__(self, runner, finished_requests_ids=None):
        super().__init__(runner, finished_requests_ids)
        self.req_ids = []

    def prepare(self,
                finished_requests_ids: Optional[list[str]] = None) -> None:
        self.req_ids.clear()
        return super().prepare(finished_requests_ids)

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        seq_ids = seq_group_metadata.seq_data.keys()
        n_seqs = len(seq_ids)
        seq_ids = list(seq_ids)
        for seq_idx in range(n_seqs):
            self.req_ids.append(seq_ids[seq_idx])
        return super().add_seq_group(seq_group_metadata)

    def build(self) -> ModelInputForGPU:
        model_input = super().build()
        last_sampler = get_last_sampler()
        if last_sampler is not None:
            input_ids = async_tensor_h2d(self.req_ids, torch.long,
                                         self.runner.device,
                                         self.runner.pin_memory)
            last_ids = async_tensor_h2d(last_sampler.seq_id.tolist(),
                                        torch.long, self.runner.device,
                                        self.runner.pin_memory)
            UpdateInputTokens(model_input.input_tokens, input_ids,
                              last_sampler.sampled_token_ids_tensor, last_ids)
        return model_input
