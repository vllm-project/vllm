# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch

from vllm.attention.backends.utils import PAD_SLOT_ID


class ConstantSizeCache(ABC):
    """
    Abstract base class for managing constant size caches 
    like Mamba and Minimax.
    """

    def __init__(self, max_batch_size: int):
        # Maps between the request id and a dict that maps between the seq_id
        # and its index inside the cache
        self.cache_indices_mapping: dict[str, dict[int, int]] = {}
        self.free_cache_indices = list(range(max_batch_size))

    @property
    @abstractmethod
    def cache(self) -> Any:
        """Return the underlying cache tensor(s)"""
        pass

    @abstractmethod
    def _copy_cache(self, from_index: int, to_index: int):
        """Copy cache data from one index to another"""
        pass

    def current_run_tensors(self, **kwargs) -> tuple:
        """
        Return the tensors for the current run's conv and ssm state.
        """
        if "seqlen_agnostic_capture_inputs" not in kwargs:
            # We get here only on Prefill/Eager mode runs
            request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
            finished_requests_ids = kwargs["finished_requests_ids"]

            self._release_finished_requests(finished_requests_ids)
            state_indices = self._prepare_current_run_cache(
                request_ids_to_seq_ids, finished_requests_ids)

            state_indices_tensor = torch.as_tensor(state_indices,
                                                   dtype=torch.int32,
                                                   device="cuda")
            cache_tensors = self.cache
        else:
            # CUDA graph capturing runs
            cache_tensors, state_indices_tensor = kwargs[
                "seqlen_agnostic_capture_inputs"]

        return (cache_tensors, state_indices_tensor)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """
        Copy the relevant state_indices into the CUDA graph input buffer 
        """
        assert all(
            key in kwargs
            for key in ["request_ids_to_seq_ids", "finished_requests_ids"])
        finished_requests_ids = kwargs["finished_requests_ids"]
        request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
        assert "seqlen_agnostic_capture_inputs" in input_buffers
        _, input_state_indices_buffer = input_buffers[
            "seqlen_agnostic_capture_inputs"]

        self._release_finished_requests(finished_requests_ids)
        state_indices = self._prepare_current_run_cache(
            request_ids_to_seq_ids, finished_requests_ids)
        cuda_graph_pad_len = input_state_indices_buffer.shape[0] - len(
            state_indices)
        state_indices.extend([PAD_SLOT_ID] * cuda_graph_pad_len)

        input_state_indices_buffer.copy_(
            torch.as_tensor(state_indices, dtype=torch.int32, device="cuda"))

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the Cache during the CUDA graph replay
        runs.
        """
        state_indices_tensor = torch.as_tensor([PAD_SLOT_ID] * batch_size,
                                               dtype=torch.int32,
                                               device="cuda")
        return (self.cache, state_indices_tensor)

    def _assign_seq_id_to_cache_index(self, cur_rid: str, seq_id: int,
                                      finished_requests_ids) -> int:
        """
        Assign (req_id,seq_id) pair to a `destination_index` index, if
        already occupied, move the occupying index to a free index.
        """
        if cur_rid in finished_requests_ids:
            # set as pad, do not allocate destination index
            return PAD_SLOT_ID
        elif cur_rid not in self.cache_indices_mapping:
            destination_index = self.free_cache_indices.pop()
            self.cache_indices_mapping[cur_rid] = {seq_id: destination_index}
            return destination_index
        elif seq_id not in (seq_ids2indices :=
                            self.cache_indices_mapping[cur_rid]):
            # parallel sampling , where n > 1, assume prefill have
            # already happened, so we copy the
            # existing cache into the siblings seq_ids caches
            index_exists = next(iter(seq_ids2indices.values()))
            # case of decoding n>1, copy prefill cache to decoding indices
            destination_index = self.free_cache_indices.pop()
            self._copy_cache(from_index=index_exists,
                             to_index=destination_index)
            self.cache_indices_mapping[cur_rid][seq_id] = destination_index
            return destination_index
        else:
            return self.cache_indices_mapping[cur_rid][seq_id]

    def _prepare_current_run_cache(
            self, request_ids_to_seq_ids: dict[str, list[int]],
            finished_requests_ids: list[str]) -> list[int]:
        return [
            self._assign_seq_id_to_cache_index(req_id, seq_id,
                                               finished_requests_ids)
            for req_id, seq_ids in request_ids_to_seq_ids.items()
            for seq_id in seq_ids
        ]

    def _release_finished_requests(self,
                                   finished_seq_groups_req_ids: list[str]):
        for req_id in finished_seq_groups_req_ids:
            if req_id in self.cache_indices_mapping:
                for seq_id in self.cache_indices_mapping[req_id]:
                    self.free_cache_indices.append(
                        self.cache_indices_mapping[req_id][seq_id])
                self.cache_indices_mapping.pop(req_id)
