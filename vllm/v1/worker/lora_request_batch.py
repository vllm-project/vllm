import torch
import numpy as np

from typing import Dict, TypeAlias, Optional, List, Tuple

from vllm.v1.worker.request_batch_base import RequestBatchAbstract, BatchInputs
from vllm.v1.worker.request_batch import RequestBatch
from vllm.v1.worker.cached_request_state import CachedRequestState
from vllm.v1.core.scheduler import  RunningRequestData
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.model_runner_device_tensors import ModelRunnerDeviceSamplingTensors
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest

LoRAID: TypeAlias = int

class LoRARequestBatch(RequestBatchAbstract):
    """
    LoRARequestBatch maintains one RequestBatch object for all possible
    LoRA IDs.
    The Create, Update, Delete methods dispatch to the RequestBatch corresponding
    to the request LoRA ID.
    The Read methods, combine / collate information from difference RequestBatch
    objects.
    """

    # Assume LoRA IDs are greater than 0.
    NO_LORA_ID: LoRAID = 0

    def _make_request_batch(self) -> RequestBatch:
        return RequestBatch(self.max_num_reqs,
                            self.max_model_len,
                            self.max_num_blocks_per_req,
                            self.pin_memory)

    def _get_lora_id_from_request(self, request: CachedRequestState) -> LoRAID:

        if request.lora_request is None:
            return self.NO_LORA_ID

        lora_id: LoRAID = request.lora_request.lora_int_id

        assert lora_id != self.NO_LORA_ID, \
            (f"LoRA request ID cannot be equal to NO_LORA_ID"
            f"({self.NO_LORA_ID})")

        # Each lora_id gets it own batch
        return lora_id

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        pin_memory: bool,
    ):
        super().__init__(max_num_reqs, max_model_len, max_num_blocks_per_req, pin_memory)

        self.req_id_to_lora_id: Dict[str, LoRAID] = {}
        self.lora_id_to_batch: Dict[LoRAID, RequestBatch] = \
            {self.NO_LORA_ID: self._make_request_batch()} 
        self.lora_id_to_lora_request: Dict[LoRAID, LoRARequest] = {}

    def remove_requests(self, req_ids: List[str]) -> None:
        for req_id in req_ids:
            lora_id: LoRAID = self.req_id_to_lora_id[req_id] 
            self.lora_id_to_batch[lora_id].remove_requests([req_id])
            self.req_id_to_lora_id.pop(req_id)

    def add_request(self,
                    request: "CachedRequestState") -> None:
        """
        Add the new or resumed requests to the persistent batch.
        """
        lora_id: LoRAID = self._get_lora_id_from_request(request) 
        if lora_id not in self.lora_id_to_batch:
            self.lora_id_to_batch[lora_id] = self._make_request_batch()
            self.lora_id_to_lora_request[lora_id] = request.lora_request

        # Requests with the same LoRA ID must have the same LoRA request
        assert self.lora_id_to_lora_request[lora_id] == request.lora_request, \
            ("Encountered 2 different LoRA requests with the same LoRA ID"
             f"LoRA request A : {self.lora_id_to_lora_request[lora_id]}"
             f"LoRA request B : {request.lora_request}")

        self.lora_id_to_batch[lora_id].add_request(request)
        self.req_id_to_lora_id[request.req_id] = lora_id

    def clear(self) -> None:
        for batch in self.lora_id_to_batch.values():
            batch.clear()

    def is_condensed(self) -> bool:
        return all([batch.is_condensed() for batch in self.lora_id_to_batch.values()])

    def sanity_check(self, expected_num_reqs: int) -> None:
        assert self.is_condensed()
        assert self.num_reqs() == expected_num_reqs
        req_ids = set(self.request_ids())
        assert len(req_ids) == self.num_reqs()
        req_id_to_index_map = self.request_id_to_index()
        assert len(req_id_to_index_map) == expected_num_reqs 
        assert len(set(req_id_to_index_map.values())) == expected_num_reqs

    def condense(self) -> None:
        for batch in self.lora_id_to_batch.values():
            batch.condense()

    def update_states(self,
                      request_id: str,
                      request_data: RunningRequestData,
                      num_existing_block_ids: int) -> None:
        lora_id: LoRAID = self.req_id_to_lora_id[request_id]
        self.lora_id_to_batch[lora_id].update_states(request_id, request_data, num_existing_block_ids)

    def append_token_id(self, request_id: str, token_id: np.int32, token_idx: int):
        lora_id: LoRAID = self.req_id_to_lora_id[request_id]
        self.lora_id_to_batch[lora_id].append_token_id(request_id, token_id, token_idx)

    def rewind_generator(self, request_id: str):
        lora_id: LoRAID = self.req_id_to_lora_id[request_id]
        self.lora_id_to_batch[lora_id].rewind_generator(request_id)

    def request_ids(self) -> List[str]:
        return sum([batch.request_ids() for batch in self.lora_id_to_batch.values()], [])

    def num_reqs(self) -> int:
        return sum([batch.num_reqs() for batch in self.lora_id_to_batch.values()])

    def all_greedy(self) -> bool:
        return all([batch.all_greedy() for batch in self.lora_id_to_batch.values()])

    def all_random(self) -> bool:
        return all([batch.all_random() for batch in self.lora_id_to_batch.values()])

    def no_top_p(self) -> bool:
        return all([batch.no_top_p() for batch in self.lora_id_to_batch.values()])

    def no_top_k(self) -> bool:
        return all([batch.no_top_k() for batch in self.lora_id_to_batch.values()])

    def max_num_logprobs(self) -> int:
        return max([batch.max_num_logprobs() for batch in self.lora_id_to_batch.values()])

    def no_logprob(self) -> bool:
        return all([batch.no_logprob() for batch in self.lora_id_to_batch.values()])

    def no_prompt_logprob(self) -> bool:
        return all([batch.no_prompt_logprob() for batch in self.lora_id_to_batch.values()])


    def make_seq_lens_tensor(self,
                             num_scheduled_tokens: np.array) -> np.array:
                             
        assert len(num_scheduled_tokens) == self.num_reqs()

        seq_lens_list : List[np.array] = []
        req_offset: int = 0
        for batch in self.lora_id_to_batch.values():
            batch_num_reqs = batch.num_reqs()
            if batch_num_reqs == 0:
                continue
            seq_lens_list.append(batch.make_seq_lens_tensor(num_scheduled_tokens[req_offset : req_offset + batch_num_reqs]))
            req_offset += batch_num_reqs
        return np.concatenate(tuple(seq_lens_list))

    def prepare_inputs(self,
                       num_scheduled_tokens: np.array,
                       block_size: int,
                       block_table_device_tensor: torch.Tensor,
                       input_tokens_device_tensor: torch.Tensor,
                       input_positions_device_tensor: torch.Tensor,
                       slot_mapping_device_tensor: torch.Tensor) -> None:

        total_num_reqs: int = self.num_reqs()
        assert len(num_scheduled_tokens) == total_num_reqs, ""
        total_num_scheduled_tokens: int = np.sum(num_scheduled_tokens)

        start_req_offset: int = 0
        start_token_offset: int = 0
        for batch in self.lora_id_to_batch.values():
            """
            Collate BatchInputs from all batches
            """
            if batch.num_reqs() == 0:
                continue

            end_req_offset = start_req_offset + batch.num_reqs()
            end_token_offset = start_token_offset + np.sum(num_scheduled_tokens[start_req_offset : end_req_offset])

            batch.prepare_inputs(num_scheduled_tokens[start_req_offset : end_req_offset],
                                 block_size,
                                 block_table_device_tensor[start_req_offset : end_req_offset],
                                 input_tokens_device_tensor[start_token_offset : end_token_offset],
                                 input_positions_device_tensor[start_token_offset : end_token_offset],
                                 slot_mapping_device_tensor[start_token_offset : end_token_offset])

            start_req_offset = end_req_offset
            start_token_offset = end_token_offset
        assert start_req_offset == total_num_reqs
        assert start_token_offset == total_num_scheduled_tokens

    def make_sampling_metadata(self,
        device_tensors: ModelRunnerDeviceSamplingTensors,
        skip_copy: bool = False) -> SamplingMetadata:

        num_reqs: int = self.num_reqs()

        # Collate generators from batches
        request_generator_map: Dict[int, torch.Generator] = {}

        start_req_offset: int = 0

        for batch in self.lora_id_to_batch.values():
            if batch.num_reqs() == 0:
                continue

            end_req_offset = start_req_offset + batch.num_reqs()

            if not skip_copy:
                device_tensors.temperature[start_req_offset:end_req_offset].copy_(
                    batch.cpu_tensors.temperature.tensor[:batch.num_reqs()], non_blocking=True)
                device_tensors.top_p[start_req_offset:end_req_offset].copy_(
                    batch.cpu_tensors.top_p.tensor[:batch.num_reqs()], non_blocking=True)
                device_tensors.top_k[start_req_offset:end_req_offset].copy_(
                    batch.cpu_tensors.top_k.tensor[:batch.num_reqs()], non_blocking=True)

            batch_request_generator_map = {idx + start_req_offset : generator for idx, generator in batch.generators.items()}
            request_generator_map.update(batch_request_generator_map)

            start_req_offset = end_req_offset
        assert start_req_offset == num_reqs

        return SamplingMetadata(
            temperature=device_tensors.temperature[:num_reqs],
            all_greedy=self.all_greedy(),
            all_random=self.all_random(),
            top_p=device_tensors.top_p[:num_reqs],
            top_k=device_tensors.top_k[:num_reqs],
            no_top_p=self.no_top_p(),
            no_top_k=self.no_top_k(),
            generators=request_generator_map,
            max_num_logprobs=self.max_num_logprobs(),
        )

    def prepare_lora_inputs(self, num_scheduled_tokens: np.array) \
            -> Tuple[LoRAMapping, set[LoRARequest]]:
        """
        Construct and return LoRAMapping and the set of all LoRA Requests.
        """
        def batch_num_prompt_mapping(batch: RequestBatch,
                                     batch_req_offset: int):
            if batch.no_prompt_logprob():
                return batch.num_reqs()

            batch_req_id_to_index: Dict[str, int] = batch.request_id_to_index()

            # select request indices that require prompt logprobs. Offset those
            # indices with batch_req_offset so it can be used to index into
            # num_scheduled_tokens.
            prompt_logprobs_req_indices: List[int] = \
                [batch_req_id_to_index[req_id] + batch_req_offset \
                    for req_id in batch.request_ids() \
                        if req_id in batch.prompt_logprob_reqs]

            num_prompt_mapping: int = np.sum(num_scheduled_tokens[prompt_logprobs_req_indices])
            num_prompt_mapping += batch.num_reqs() - len(prompt_logprobs_req_indices)

            return num_prompt_mapping

        num_tokens: int = np.sum(num_scheduled_tokens)
        index_mapping: np.array = np.empty((num_tokens,), dtype=np.int32)
        # prompt_mapping could be as big as num_tokens depending on the
        # requests requesting prompt_logprobs
        prompt_mapping: np.array = np.empty((num_tokens,), dtype=np.int32)
        lora_requests: set[LoRARequest] = set()

        token_offset: int = 0
        req_offset: int = 0
        prompt_mapping_offset: int = 0
        for lora_id, batch in self.lora_id_to_batch.items():
            batch_num_reqs = batch.num_reqs()
            if batch_num_reqs == 0:
                continue

            if lora_id != self.NO_LORA_ID:
                lora_requests.add(self.lora_id_to_lora_request[lora_id])

            batch_num_tokens = np.sum(num_scheduled_tokens[req_offset:req_offset + batch_num_reqs])
            index_mapping[token_offset : token_offset + batch_num_tokens] = lora_id

            num_prompt_mapping = batch_num_prompt_mapping(batch, req_offset)
            prompt_mapping[prompt_mapping_offset : prompt_mapping_offset + num_prompt_mapping] = lora_id

            token_offset += batch_num_tokens
            req_offset += batch_num_reqs
            prompt_mapping_offset += num_prompt_mapping

        # TODO (varun) : Is there a way to remove cast to tuple ?
        # TODO (varun) : Not differentiating between prefill and decode for now.
        # needs some investigation.
        return LoRAMapping(index_mapping = tuple(index_mapping[:token_offset]),
                           prompt_mapping = tuple(prompt_mapping[:prompt_mapping_offset]),
                           is_prefill=True), lora_requests
