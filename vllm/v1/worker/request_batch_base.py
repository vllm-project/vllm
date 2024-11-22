from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Dict, Set
from vllm.sampling_params import SamplingType
from vllm.v1.core.scheduler import  RunningRequestData
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.model_runner_device_tensors import (
    ModelRunnerDeviceTensors,
    ModelRunnerDeviceSamplingTensors)

from vllm.v1.worker.cached_request_state import CachedRequestState
from abc import ABC, abstractmethod

import torch
import numpy as np

@dataclass
class BatchInputs:
    """
    Batch data reprensented as numpy arrays for model execute.
    """
    input_tokens_cpu_tensor : torch.Tensor
    input_positions_np : np.array
    slot_mapping_cpu_tensor : torch.Tensor

@dataclass(slots=True)
class CPUTensor:
    tensor: torch.Tensor 
    np_tensor: np.array

    @staticmethod
    def build(tensor: torch.Tensor) -> "CPUTensor":
        return CPUTensor(tensor, tensor.numpy())

# TODO (varun) : Make the torch tensors first class ?
@dataclass
class BatchCPUSamplingTensors:
    temperature: CPUTensor
    top_p: CPUTensor
    top_k: CPUTensor

    @staticmethod
    def build(max_num_reqs: int,
              pin_memory: bool) -> "BatchCPUSamplingTensors":
        def make_tensor(dtype: torch.dtype) -> CPUTensor:
            tensor = torch.empty((max_num_reqs, ),
                                 dtype = dtype,
                                 pin_memory=pin_memory,
                                 device="cpu")
            return CPUTensor.build(tensor)

        return BatchCPUSamplingTensors(
            temperature = make_tensor(torch.float32),
            top_p = make_tensor(torch.float32),
            top_k = make_tensor(torch.int32))
    
@dataclass(slots=True)
class BatchCPUTensors:
    """
    Batched CPU tensors maintained by RequestBatch.
    Note that all the tensors have the request index as their major dimension.
    """
    token_ids_np: np.array 
    num_computed_tokens_np: np.array
    block_table: CPUTensor 
    # Sampling tensors
    temperature: CPUTensor
    top_p: CPUTensor
    top_k: CPUTensor

    @staticmethod
    def make(max_num_reqs : int,
              max_model_len : int,
              max_num_blocks_per_req: int,
              pin_memory: bool) -> "BatchCPUTensors":

        token_ids_np = np.empty((max_num_reqs, max_model_len), dtype=np.int32)
        num_computed_tokens_np = np.empty(max_num_reqs, dtype=np.int32)

        block_table = CPUTensor.build(
            tensor = torch.zeros((max_num_reqs, max_num_blocks_per_req),
                                 dtype=torch.int32,
                                 pin_memory=pin_memory,
                                 device="cpu"))

        temperature = CPUTensor.build(
            tensor = torch.empty((max_num_reqs, ),
                                 dtype = torch.float32,
                                 pin_memory=pin_memory,
                                 device="cpu"))

        top_p = CPUTensor.build(
            tensor = torch.empty((max_num_reqs, ),
                                 dtype = torch.float32,
                                 pin_memory=pin_memory,
                                 device="cpu"))

        top_k = CPUTensor.build(
            tensor = torch.empty((max_num_reqs, ),
                                 dtype = torch.int32,
                                 pin_memory=pin_memory,
                                 device="cpu"))

        return BatchCPUTensors(token_ids_np = token_ids_np,
                               num_computed_tokens_np = num_computed_tokens_np,
                               block_table = block_table,
                               temperature = temperature,
                               top_p = top_p,
                               top_k = top_k)

    def add_request_data(self, req_index: int, request: CachedRequestState) -> None:
        """
        Given a request index and some request data, update the CPU tensors at the request index,
        with data in request data.
        """
        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.token_ids_np[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_np[req_index, start_idx:end_idx] = request.output_token_ids

        self.num_computed_tokens_np[req_index] = request.num_computed_tokens

        num_blocks = len(request.block_ids)
        self.block_table.np_tensor[req_index, :num_blocks] = request.block_ids

        sampling_params = request.sampling_params
        self.temperature.np_tensor[req_index] = sampling_params.temperature
        self.top_p.np_tensor[req_index] = sampling_params.top_p
        self.top_k.np_tensor[req_index] = sampling_params.top_k

    # to req index is the empty
    def transfer(self, from_req_index: int, to_req_index: int) -> None:
        """
        Copies data from `from_req_index` to `to_req_index`
        """
        # TODO(woosuk): Optimize the copy of token_ids_cpu and
        # block_table_cpu.
        self.token_ids_np[to_req_index] = self.token_ids_np[from_req_index]
        self.num_computed_tokens_np[to_req_index] = self.num_computed_tokens_np[from_req_index]
        self.block_table.np_tensor[to_req_index] = self.block_table.np_tensor[from_req_index]
        self.temperature.np_tensor[to_req_index] = self.temperature.np_tensor[from_req_index]
        self.top_p.np_tensor[to_req_index] = self.top_p.np_tensor[from_req_index]
        self.top_k.np_tensor[to_req_index] = self.top_k.np_tensor[from_req_index]

    def update_request_state(self, req_index: int, 
                            request_data: RunningRequestData,
                            num_existing_block_ids: int):
        """
        Given a request index, a running request data (a delta) update the states
        in the cpu tensors.
        num_existing_block_ids, is used in updating the block table. This is required
        as we don't track how many block ids for a request are valid in the block table.
        """
        # Update the num_computed_tokens.
        self.num_computed_tokens_np[req_index] = (request_data.num_computed_tokens)

        # Update the block table.
        num_new_blocks = len(request_data.new_block_ids)
        if num_new_blocks == 0:
            return
        start_index = num_existing_block_ids
        end_index = start_index + num_new_blocks
        self.block_table.np_tensor[req_index, start_index:end_index] = \
                            request_data.new_block_ids

class RequestBatchAbstract(ABC):

    def __init__(self,
                 max_num_reqs: int,
                 max_model_len: int,
                 max_num_blocks_per_req: int,
                 pin_memory: bool):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.pin_memory = pin_memory

    @abstractmethod
    def remove_requests(self, req_ids: List[str]) -> None:
        raise NotImplementedError


    @abstractmethod
    def add_request(self,
                    request: "CachedRequestState") -> None:
        """
        Add the new or resumed requests to the persistent batch.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def condense(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_states(self,
                      request_id: str,
                      request_data: RunningRequestData,
                      num_existing_block_ids: int) -> None:
        """
        Update state of the request in batch.
        """
        raise NotImplementedError

    @abstractmethod
    def append_token_id(self, request_id: str, token_id: np.int32, token_idx: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def rewind_generator(self, request_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_seq_lens_tensor(self,
                             num_scheduled_tokens: np.array) -> np.array:
        """
        Given the number of tokens scheduled per request, return the sequence lengths
        of the requests based on cpu_tensors.num_computed_tokens_np
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
    
    @abstractmethod
    def make_sampling_metadata(self,
        sampling_device_tensors: ModelRunnerDeviceSamplingTensors,
        skip_copy: bool = False) -> SamplingMetadata:
        """
        Transfer cpu sampling to device, if a copy is required, and
        translate the batch into SamplingMetadata for model sampling.
        """
        raise NotImplementedError

    @abstractmethod
    def request_ids(self) -> List[str]:
        # Return request ids in order that they appear in the batch.
        raise NotImplementedError

    @abstractmethod
    def num_reqs(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def all_greedy(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def all_random(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def no_top_p(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def no_top_k(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def max_num_logprobs(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def no_logprob(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def no_prompt_logprob(self) -> bool:
        raise NotImplementedError

class RequestBatchBase(RequestBatchAbstract):

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        pin_memory: bool,
    ):
        super().__init__(max_num_reqs, max_model_len, max_num_blocks_per_req, pin_memory)

        self.req_ids: List[Optional[str]] = [None] * max_num_reqs
        self.req_id_to_index: Dict[str, int] = {}
        # Track fragmentation due to request-removal so the empty slots
        # can be re-used for new requests.
        self.empty_req_indices: List[int] = []
        self.is_empty_req_indices_sorted: bool = True

        # Batch CPU Tensors
        self.cpu_tensors = \
            BatchCPUTensors.make(self.max_num_reqs, self.max_model_len, self.max_num_blocks_per_req, self.pin_memory)

        # Batch Request info
        self.greedy_reqs: Set[str] = set()
        self.random_reqs: Set[str] = set()
        self.top_p_reqs: Set[str] = set()
        self.top_k_reqs: Set[str] = set()
        # req_index -> generator
        self.generators: Dict[int, torch.Generator] = {}
        self.num_logprobs: Dict[str, int] = {}
        self.prompt_logprob_reqs: Set[str] = set()

    def _add_request(
        self,
        request: "CachedRequestState",
        req_index: int,
    ) -> None:
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        self.req_ids[req_index] = req_id
        self.req_id_to_index[req_id] = req_index

        self.cpu_tensors.add_request_data(req_index, request)

        sampling_params = request.sampling_params
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.greedy_reqs.add(req_id)
        else:
            self.random_reqs.add(req_id)
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        if sampling_params.top_k > 0:
            self.top_k_reqs.add(req_id)

        self.generators[req_index] = request.generator

        num_logprobs = sampling_params.logprobs
        if num_logprobs is not None and num_logprobs > 0:
            self.num_logprobs[req_id] = num_logprobs
        if sampling_params.prompt_logprobs:
            self.prompt_logprob_reqs.add(req_id)

    def _remove_request(self, req_id: str) -> None:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return
        self.req_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.prompt_logprob_reqs.discard(req_id)

        self.empty_req_indices.append(req_index)
        self.is_empty_req_indices_sorted = False
        return req_index

    def remove_requests(self, req_ids: List[str]) -> None:
        for req_id in req_ids:
            self._remove_request(req_id)

    def add_request(self,
                    request: "CachedRequestState") -> None:
        """
        Add the new or resumed requests to the persistent batch.
        """
        # sort empty_req_indices so the smaller ones can be filled first.
        if not self.is_empty_req_indices_sorted:
            self.empty_req_indices = sorted(self.empty_req_indices, reverse = True)
            self.is_empty_req_indices_sorted = True

        # The smaller empty indices are filled first.
        if self.empty_req_indices:
            # Fill the empty index.
            req_index = self.empty_req_indices.pop()
        else:
            # Append to the end.
            req_index = self.num_reqs()

        self._add_request(request, req_index)

    def clear(self) -> None:
        self.req_ids = [None] * self.max_num_reqs
        self.req_id_to_index.clear()
        self.greedy_reqs.clear()
        self.random_reqs.clear()
        self.top_p_reqs.clear()
        self.top_k_reqs.clear()
        self.generators.clear()
        self.num_logprobs.clear()
        self.prompt_logprob_reqs.clear()

    def is_condensed(self) -> bool:
        val = all([x is not None for x in self.req_ids[:self.num_reqs()]]) and \
                all([x is None for x in self.req_ids[self.num_reqs():]])
        if not val:
            print (f"num reqs : {self.num_reqs()}")
            valid_reqs = self.req_ids[:self.num_reqs()]
            invalid_reqs = self.req_ids[self.num_reqs():]

            print (f" - empy in valid reqs {valid_reqs.index(None)}")
            print (f" - valid reqs   {len(valid_reqs)}   : {valid_reqs}")
            print (f" - invalid reqs {len(invalid_reqs)} : {invalid_reqs} ")
            print (f" - valid correct : {all([x is not None for x in self.req_ids[:self.num_reqs()]])}")
            print (f" - invalid correct : {all([x is None for x in self.req_ids[self.num_reqs():]])}")
            print (f" empty indices : {self.empty_req_indices} ")

        return val

    def condense(self) -> None:
        if self.num_reqs() == 0:
            # The batched states are empty.
            return
        if not self.empty_req_indices:
            # The batch is packed already.
            return

        # sort empty_req_indices.
        if not self.is_empty_req_indices_sorted:
            self.empty_req_indices = sorted(self.empty_req_indices, reverse = True)
            self.is_empty_req_indices_sorted = True

        last_req_index = self.num_reqs() + len(self.empty_req_indices) - 1
        while self.empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in self.empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = self.empty_req_indices.pop()
            if empty_index >= last_req_index:
                self.empty_req_indices.clear()
                break

            assert self.req_ids[last_req_index] is not None,  \
                    (f"Invalid last_req_index {last_req_index}, "
                     f" num_reqs {self.num_reqs()}"
                     f" empty_indices {self.empty_req_indices}")
                     
            # Swap the states.
            req_id = self.req_ids[last_req_index]
            self.req_ids[empty_index] = req_id
            self.req_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            self.cpu_tensors.transfer(to_req_index=empty_index,
                                     from_req_index=last_req_index)

            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

    def update_states(self,
                      request_id: str,
                      request_data: RunningRequestData,
                      num_existing_block_ids: int) -> None:
        """
        Update state of the request in batch.
        """
        req_index = self.req_id_to_index[request_id]
        self.cpu_tensors.update_request_state(req_index, request_data, num_existing_block_ids)

    def append_token_id(self, request_id: str, token_id: np.int32, token_idx: int) -> None:
        req_idx: int = self.req_id_to_index[request_id]
        self.cpu_tensors.token_ids_np[req_idx, token_idx] = token_id

    def rewind_generator(self, request_id: str) -> None:
        req_idx: int = self.req_id_to_index[request_id]
        generator = self.generators.get(req_idx)
        if generator is not None:
            # This relies on cuda-specific torch-internal impl details
            generator.set_offset(generator.get_offset() - 4)

    def make_token_positions(self,
                             num_scheduled_tokens: np.array,
                             token_req_indices: Optional[np.array]) -> torch.Tensor:
        """
        Given the number of scheduled tokens for per request, translate the
        batch into token positions.

        E.g. If there are 3 requests in batch, where,
         self.cpu_tensors.num_computed_tokens => [4, 10, 3]
         num_scheduled_tokens => [3, 4, 2]
         then,
         return [4, 5, 6, 10, 11, 12, 13, 3, 4]
        """

        max_num_scheduled_tokens = num_scheduled_tokens.max()
        assert max_num_scheduled_tokens > 0

        if token_req_indices is None:
            indices = np.arange(self.num_reqs())
            token_req_indices = np.repeat(indices, num_scheduled_tokens)

        num_tokens : int = len(token_req_indices)
        positions = torch.empty(num_tokens, 
                                dtype=torch.int64,
                                device="cpu",
                                pin_memory=self.pin_memory)

        # Get batched arange
        # e.g. num_schedule_tokens [3, 4, 2]
        # arange => [0, 1, 2, 0, 1, 2, 3, 0, 1]
        arange_matrix = np.tile(np.arange(max_num_scheduled_tokens),
                                (self.num_reqs(), 1))
        mask = arange_matrix < num_scheduled_tokens[:, np.newaxis]
        arange = arange_matrix[mask]

        ## Input Positions
        np.add(self.cpu_tensors.num_computed_tokens_np[token_req_indices],
               arange,
               out = positions.numpy())

        return positions


    def make_token_ids(self,
                       token_indices: torch.Tensor) -> torch.Tensor:
        """
        Given the token indices of the requests, that is flattened to match
        cpu_tensors.token_ids_np, select the tokens and return as numpy array.
        """
        num_tokens : int = len(token_indices)
        token_ids = torch.empty(num_tokens, 
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)

        torch.index_select(
            torch.from_numpy(self.cpu_tensors.token_ids_np).flatten(),
            0,
            token_indices,
            out = token_ids)
        return token_ids

    def make_slot_mapping(self,
                          token_indices: torch.Tensor,
                          block_size: int) -> torch.Tensor:
        """
        Given the token indices of the requests, that is flattened to match
        cpu_tensors.token_ids_np, return the slot mapping for the tokens.
        """
        num_tokens : int = len(token_indices)
        slot_mapping = torch.empty(num_tokens, 
                                dtype=torch.int64,
                                device="cpu",
                                pin_memory=self.pin_memory)

        block_numbers = self.cpu_tensors.block_table.tensor.flatten()[
            token_indices // block_size]
        block_offsets = token_indices % block_size
        torch.add(block_numbers * block_size,
                  block_offsets,
                  out=slot_mapping)
        return slot_mapping

    def make_seq_lens_tensor(self,
                             num_scheduled_tokens: np.array) -> np.array:
        """
        Given the number of tokens scheduled per request, return the sequence lengths
        of the requests based on cpu_tensors.num_computed_tokens_np
        """
        return self.cpu_tensors.num_computed_tokens_np[:self.num_reqs()]  + num_scheduled_tokens

    @abstractmethod
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
        raise NotImplementedError
    
    @abstractmethod
    def make_sampling_metadata(self,
        sampling_device_tensors: ModelRunnerDeviceSamplingTensors,
        skip_copy: bool = False) -> SamplingMetadata:
        """
        Transfer cpu sampling to device, if a copy is required, and
        translate the batch into SamplingMetadata for model sampling.
        """
        raise NotImplementedError

    def request_ids(self) -> List[str]:
        return self.req_ids[:self.num_reqs()]

    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    def max_num_logprobs(self) -> int:
        return max(self.num_logprobs.values()) if self.num_logprobs else 0

    def no_logprob(self) -> bool:
        return len(self.num_logprobs) == 0

    def no_prompt_logprob(self) -> bool:
        return len(self.prompt_logprob_reqs) == 0