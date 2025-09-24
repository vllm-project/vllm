# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, cast

import numpy as np
import torch

from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState


class SamplingInputBatch:

    def __init__(self, max_num_reqs: int):
        self.temperature_cpu = np.empty(max_num_reqs, dtype=np.float32)
        self.top_p_cpu = np.empty(max_num_reqs, dtype=np.float32)
        self.top_k_cpu = np.empty(max_num_reqs, dtype=np.int32)


class InputBatch:
    """Persistent input batch, based on InputBatch for GPU/TPU backends."""

    def __init__(
            self,
            max_num_reqs: int,
            max_model_len: int,
            max_num_batched_tokens: int,
            block_sizes: list[int],  # The block_size of each kv cache group
    ):
        self.max_num_reqs = max_num_reqs

        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            dtype=torch.int32,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()

        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=False,
            device="cpu",
            block_sizes=block_sizes,
        )

        self.req_output_token_ids: list[Optional[list[int]]] = []

        # Sampling-related.
        self.sampling = SamplingInputBatch(max_num_reqs)

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids
        # Number of token ids in token_ids_cpu.
        self.num_tokens[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        # Sampling-related.
        sampling_params = request.sampling_params
        assert sampling_params is not None, "pooling requests not supported yet"
        self.sampling.temperature_cpu[req_index] = sampling_params.temperature
        self.sampling.top_p_cpu[req_index] = sampling_params.top_p
        self.sampling.top_k_cpu[req_index] = sampling_params.top_k

    def remove_request(self, req_id: str) -> Optional[int]:
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        return req_index

    def condense(self, empty_req_indices: list[int]) -> None:
        """Move non-empty requests down into lower, empty indices.
        
        Args:
            empty_req_indices: empty batch indices, sorted descending.
        """
        num_reqs = self.num_reqs
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Swap the states.
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens = self.num_tokens[last_req_index]
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens]
            self.num_tokens[empty_index] = num_tokens
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table.move_row(last_req_index, empty_index)

            # Sampling-related.
            sampling = self.sampling
            sampling.temperature_cpu[empty_index] = sampling.temperature_cpu[
                last_req_index]
            sampling.top_p_cpu[empty_index] = sampling.top_p_cpu[
                last_req_index]
            sampling.top_k_cpu[empty_index] = sampling.top_k_cpu[
                last_req_index]

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[self.num_reqs:]
        del self.req_output_token_ids[self.num_reqs:]
