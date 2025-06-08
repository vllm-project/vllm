# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.v1.worker.block_table import MultiGroupBlockTable


@dataclass
class BaseRequestState:

    req_id: str
    prompt_token_ids: list[int]
    mm_inputs: list[MultiModalKwargs]
    mm_positions: list[PlaceholderRange]
    block_ids: list[list[int]]
    num_computed_tokens: int
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[int] = None
    lora_request: Optional[LoRARequest] = None

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_token_id(self, idx: int) -> int:
        raise NotImplementedError


RequestState = TypeVar("RequestState", bound=BaseRequestState)


class BaseInputBatch(Generic[RequestState]):

    def __init__(
            self,
            max_num_reqs: int,
            max_model_len: int,
            max_num_batched_tokens: int,
            device: torch.device,
            pin_memory: bool,
            vocab_size: int,
            block_sizes: list[int],  # The block_size of each kv cache group
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs, ),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = \
            self.num_computed_tokens_cpu_tensor.numpy()

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
        )

        # lora related
        self.request_lora_mapping = np.zeros((self.max_num_reqs, ),
                                             dtype=np.int32)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

    def refresh(self):
        pass

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def add_request(
        self,
        request: RequestState,
        req_index: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    def _add_request(
        self,
        request: RequestState,
        req_index: Optional[int] = None,
    ) -> int:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
        else:
            self._req_ids[req_index] = req_id

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids

        # Number of token ids in token_ids_cpu.
        self.num_tokens[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        # Add request lora ID
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # No LoRA
            self.request_lora_mapping[req_index] = 0
        return req_index

    def remove_request(self, req_id: str) -> Optional[int]:
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None

        # LoRA
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            self.lora_id_to_request_ids[lora_id].discard(req_id)
            if len(self.lora_id_to_request_ids[lora_id]) == 0:
                self.lora_id_to_request_ids.pop(lora_id)
                self.lora_id_to_lora_request.pop(lora_id)
            self.request_lora_mapping[req_index] = 0

        return req_index

    def _make_prompt_token_ids_tensor(self) -> Optional[torch.Tensor]:
        if self.num_reqs == 0:
            return None

        max_prompt_len = self.num_prompt_tokens[:self.num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:self.
                                                 num_reqs, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(self.num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i]:] = self.vocab_size
        return prompt_token_ids_cpu_tensor.to(device=self.device,
                                              non_blocking=True)

    # with move=True, the semantic is to move from right to left
    # i2 = i1
    def swap_or_move_states(self,
                            i1: int,
                            i2: int,
                            move: bool = False) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]

        assert old_id_i2 is not None

        if not move:
            assert old_id_i1 is not None
            tmp1 = (
                self._req_ids[i1],
                self.req_id_to_index[old_id_i1],
                self.num_tokens[i1],
                self.num_prompt_tokens[i1],
                self.num_computed_tokens_cpu[i1],
                self.request_lora_mapping[i1],
            )
            # empty index <- last index
            self.req_id_to_index[old_id_i1] = self.req_id_to_index[old_id_i2]
        else:
            req_id = self._req_ids[i2]
            assert req_id is not None
            self.req_id_to_index[req_id] = i1

        self._req_ids[i1] = self._req_ids[i2]
        self.num_tokens[i1] = self.num_tokens[i2]
        self.num_prompt_tokens[i1] = self.num_prompt_tokens[i2]
        self.num_computed_tokens_cpu[i1] = self.num_computed_tokens_cpu[i2]
        self.request_lora_mapping[i1] = self.request_lora_mapping[i2]

        if not move:
            (
                self._req_ids[i2],
                self.req_id_to_index[old_id_i2],
                self.num_tokens[i2],
                self.num_prompt_tokens[i2],
                self.num_computed_tokens_cpu[i2],
                self.request_lora_mapping[i2],
            ) = tmp1
            # NOTE: the following is unsafe
            # self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...], =\
            #     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]
            # instead, we need to temporiarily copy the data for
            # one of the indices
            # TODO(lucas): optimize this by only copying valid indices
            tmp = self.token_ids_cpu[i1, ...].copy()
            self.token_ids_cpu[i1, ...] = self.token_ids_cpu[i2, ...]
            self.token_ids_cpu[i2, ...] = tmp
            self.block_table.swap_row(i1, i2)
        else:
            self._req_ids[i2] = None
            num_tokens = self.num_tokens[i1]
            self.token_ids_cpu[i1, :num_tokens] = self.token_ids_cpu[
                i2, :num_tokens]
            self.block_table.move_row(i2, i1)  # not a mistake

    def condense(self, empty_req_indices: list[int]) -> None:
        num_reqs = self.num_reqs
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
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

            self.swap_or_move_states(empty_index, last_req_index, True)

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[self.num_reqs:]

    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        Given the num_scheduled_tokens for each request in the batch, return
        datastructures used to activate the current LoRAs.
        Returns:
            1. prompt_lora_mapping: A tuple of size self.num_reqs where,
               prompt_lora_mapping[i] is the LoRA id to use for the ith prompt.
            2. token_lora_mapping: A tuple of size np.sum(num_scheduled_tokens)
               where, token_lora_mapping[i] is the LoRA id to use for ith token.
            3. lora_requests: Set of relevant LoRA requests.
        """

        req_lora_mapping = self.request_lora_mapping[:self.num_reqs]
        prompt_lora_mapping = tuple(req_lora_mapping)
        token_lora_mapping = tuple(
            req_lora_mapping.repeat(num_scheduled_tokens))
        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values())

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    @property
    @abstractmethod
    def num_reqs(self) -> int:
        raise NotImplementedError
