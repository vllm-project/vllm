"""
ModelRunner input batch
"""

from typing import Optional, List, Dict

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.request_batch_base import RequestBatchBase, BatchInputs
from vllm.v1.core.scheduler import  RunningRequestData
from vllm.v1.worker.model_runner_device_tensors import ModelRunnerDeviceSamplingTensors

import numpy as np
import torch

class RequestBatch(RequestBatchBase):

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        pin_memory: bool):
        super().__init__(max_num_reqs, max_model_len, max_num_blocks_per_req, pin_memory)

    def prepare_inputs(self,
                       num_scheduled_tokens: np.array,
                       block_size: int,
                       block_table_device_tensor: Optional[torch.Tensor] = None,
                       input_tokens_device_tensor: Optional[torch.Tensor] = None,
                       input_positions_device_tensor: Optional[torch.Tensor] = None,
                       slot_mapping_device_tensor: Optional[torch.Tensor] = None) -> Optional[BatchInputs]:
        """
        Translate batch into numpy arrays for model execute.
        When device_tensors are available, kickoff a non-blocking cpu-to-device transfer as soon as
        the cpu tensors are prepared. 
        """
        num_reqs = self.num_reqs()

        if num_reqs == 0:
            # Empty batch
            return None

        if block_table_device_tensor is not None:
            # Trigger block table copy
            block_table_device_tensor[:num_reqs].copy_(
                self.cpu_tensors.block_table.tensor[:num_reqs],
                non_blocking=True)

        assert len(num_scheduled_tokens) == num_reqs
        indices = np.arange(num_reqs)
        req_indices = np.repeat(indices, num_scheduled_tokens)
        # TODO (varun) : get this directly from scheduler outputs
        total_num_scheduled_tokens = np.sum(num_scheduled_tokens) 

        # Input positions
        token_positions: torch.Tensor = self.make_token_positions(num_scheduled_tokens, req_indices)
        assert len(token_positions) == total_num_scheduled_tokens
        if input_positions_device_tensor is not None:
            input_positions_device_tensor[:total_num_scheduled_tokens].copy_(
                token_positions,
                non_blocking=True)

        # Token indices
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = token_positions + req_indices * self.max_model_len

        # Input tokens 
        token_ids: torch.Tensor = self.make_token_ids(token_indices)
        assert len(token_ids) == total_num_scheduled_tokens
        if input_tokens_device_tensor is not None:
            input_tokens_device_tensor[:total_num_scheduled_tokens].copy_(
                token_ids,
                non_blocking=True)

        # Slot mapping
        slot_mapping: torch.Tensor = self.make_slot_mapping(token_indices, block_size)
        assert len(slot_mapping) == total_num_scheduled_tokens
        if slot_mapping_device_tensor is not None:
            slot_mapping_device_tensor[:total_num_scheduled_tokens].copy_(
                slot_mapping,
                non_blocking=True)
        
        return BatchInputs(input_tokens_cpu_tensor=token_ids,
                           input_positions_np=token_positions,
                           slot_mapping_cpu_tensor=slot_mapping)


    def make_sampling_metadata(self,
        device_tensors: ModelRunnerDeviceSamplingTensors,
        skip_copy: bool = False) -> SamplingMetadata:
        """
        Transfer cpu sampling to device, if a copy is required, and
        translate the batch into SamplingMetadata for model sampling.
        """

        num_reqs: int = self.num_reqs()

        if not skip_copy:
            device_tensors.temperature[:num_reqs].copy_(
                self.cpu_tensors.temperature.tensor[:num_reqs], non_blocking=True)
            device_tensors.top_p[:num_reqs].copy_(
                self.cpu_tensors.top_p.tensor[:num_reqs], non_blocking=True)
            device_tensors.top_k[:num_reqs].copy_(
                self.cpu_tensors.top_k.tensor[:num_reqs], non_blocking=True)

        return SamplingMetadata(
            temperature=device_tensors.temperature[:num_reqs],
            all_greedy=self.all_greedy(),
            all_random=self.all_random(),
            top_p=device_tensors.top_p[:num_reqs],
            top_k=device_tensors.top_k[:num_reqs],
            no_top_p=self.no_top_p(),
            no_top_k=self.no_top_k(),
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs(),
        )
