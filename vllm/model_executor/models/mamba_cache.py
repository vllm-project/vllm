from typing import Dict, List, Optional

import torch

from vllm.attention.backends.abstract import AttentionMetadata


class MambaCacheManager:

    def __init__(self, dtype, num_mamba_layers, max_batch_size,
                 conv_state_shape, temporal_state_shape):

        conv_state = torch.empty(size=(num_mamba_layers, max_batch_size) +
                                 conv_state_shape,
                                 dtype=dtype,
                                 device="cuda")
        temporal_state = torch.empty(size=(num_mamba_layers, max_batch_size) +
                                     temporal_state_shape,
                                     dtype=dtype,
                                     device="cuda")

        self.mamba_cache = (conv_state, temporal_state)

        # Maps between the request id and a dict that maps between the seq_id
        # and its index inside the self.mamba_cache
        self.mamba_cache_indices_mapping: Dict[str, Dict[int, int]] = {}

    def current_run_tensors(self, input_ids: torch.Tensor,
                            attn_metadata: AttentionMetadata, **kwargs):
        """
        Return the tensors for the current run's conv and ssm state.
        """
        if "seqlen_agnostic_capture_inputs" not in kwargs:
            # We get here only on Prefill/Eager mode runs
            request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
            finished_requests_ids = kwargs["finished_requests_ids"]

            self._release_finished_requests(finished_requests_ids)
            mamba_cache_tensors = self._prepare_current_run_mamba_cache(
                request_ids_to_seq_ids, finished_requests_ids)

        else:
            # CUDA graph capturing runs
            mamba_cache_tensors = kwargs["seqlen_agnostic_capture_inputs"]

        return mamba_cache_tensors

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """
        Copy the relevant Mamba cache into the CUDA graph input buffer
        that was provided during the capture runs
        (JambaForCausalLM.mamba_gc_cache_buffer).
        """
        assert all(
            key in kwargs
            for key in ["request_ids_to_seq_ids", "finished_requests_ids"])
        finished_requests_ids = kwargs["finished_requests_ids"]
        request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]

        self._release_finished_requests(finished_requests_ids)
        self._prepare_current_run_mamba_cache(request_ids_to_seq_ids,
                                              finished_requests_ids)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the Mamba Cache during the CUDA graph
        replay runs.
        """
        return tuple(buffer[:, :batch_size] for buffer in self.mamba_cache)

    def _swap_mamba_cache(self, from_index: int, to_index: int):
        assert len(self.mamba_cache) > 0
        for cache_t in self.mamba_cache:
            cache_t[:, [to_index,from_index]] = \
             cache_t[:, [from_index,to_index]]

    def _copy_mamba_cache(self, from_index: int, to_index: int):
        assert len(self.mamba_cache) > 0
        for cache_t in self.mamba_cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)

    def _move_out_if_already_occupied(self, index: int,
                                      all_occupied_indices: List[int]):
        if index in all_occupied_indices:
            first_free_index = self._first_free_index_in_mamba_cache()
            # In case occupied, move the occupied to a new empty block
            self._move_cache_index_and_mappings(from_index=index,
                                                to_index=first_free_index)

    def _assign_seq_id_to_mamba_cache_in_specific_dest(self, cur_rid: str,
                                                       seq_id: int,
                                                       destination_index: int):
        """
        Assign (req_id,seq_id) pair to a `destination_index` index, if
        already occupied, move the occupying index to a free index.
        """
        all_occupied_indices = self._get_all_occupied_indices()
        if cur_rid not in self.mamba_cache_indices_mapping:
            self._move_out_if_already_occupied(
                index=destination_index,
                all_occupied_indices=all_occupied_indices)
            self.mamba_cache_indices_mapping[cur_rid] = {
                seq_id: destination_index
            }
        elif seq_id not in (seq_ids2indices :=
                            self.mamba_cache_indices_mapping[cur_rid]):
            # parallel sampling , where n > 1, assume prefill have
            # already happened now we only need to copy the already
            # existing cache into the siblings seq_ids caches
            self._move_out_if_already_occupied(
                index=destination_index,
                all_occupied_indices=all_occupied_indices)
            index_exists = list(seq_ids2indices.values())[0]
            # case of decoding n>1, copy prefill cache to decoding indices
            self._copy_mamba_cache(from_index=index_exists,
                                   to_index=destination_index)
            self.mamba_cache_indices_mapping[cur_rid][
                seq_id] = destination_index
        else:
            # already exists
            cache_index_already_exists = self.mamba_cache_indices_mapping[
                cur_rid][seq_id]
            if cache_index_already_exists != destination_index:
                # In case the seq id already exists but not in
                # the right destination, swap it with what's occupying it
                self._swap_pair_indices_and_mappings(
                    from_index=cache_index_already_exists,
                    to_index=destination_index)

    def _prepare_current_run_mamba_cache(
            self, request_ids_to_seq_ids: Dict[str, list[int]],
            finished_requests_ids: List[str]):
        running_indices = []
        request_ids_to_seq_ids_flatten = [
            (req_id, seq_id)
            for req_id, seq_ids in request_ids_to_seq_ids.items()
            for seq_id in seq_ids
        ]
        batch_size = len(request_ids_to_seq_ids_flatten)
        for dest_index, (request_id,
                         seq_id) in enumerate(request_ids_to_seq_ids_flatten):
            if request_id in finished_requests_ids:
                # Do not allocate cache index for requests that run
                # and finish right after
                continue
            self._assign_seq_id_to_mamba_cache_in_specific_dest(
                request_id, seq_id, dest_index)
            running_indices.append(dest_index)

        self._clean_up_first_bs_blocks(batch_size, running_indices)
        conv_state = self.mamba_cache[0][:, :batch_size]
        temporal_state = self.mamba_cache[1][:, :batch_size]

        return (conv_state, temporal_state)

    def _get_all_occupied_indices(self):
        return [
            cache_idx
            for seq_ids2indices in self.mamba_cache_indices_mapping.values()
            for cache_idx in seq_ids2indices.values()
        ]

    def _clean_up_first_bs_blocks(self, batch_size: int,
                                  indices_for_current_run: List[int]):
        # move out all of the occupied but currently not running blocks
        # outside of the first n blocks
        destination_indices = range(batch_size)
        max_possible_batch_size = self.mamba_cache[0].shape[1]
        for destination_index in destination_indices:
            if destination_index in self._get_all_occupied_indices() and  \
               destination_index not in indices_for_current_run:
                # move not running indices outside of the batch
                all_other_indices = list(
                    range(batch_size, max_possible_batch_size))
                first_avail_index = self._first_free_index_in_mamba_cache(
                    all_other_indices)
                self._swap_indices(from_index=destination_index,
                                   to_index=first_avail_index)

    def _move_cache_index_and_mappings(self, from_index: int, to_index: int):
        self._copy_mamba_cache(from_index=from_index, to_index=to_index)
        self._update_mapping_index(from_index=from_index, to_index=to_index)

    def _swap_pair_indices_and_mappings(self, from_index: int, to_index: int):
        self._swap_mamba_cache(from_index=from_index, to_index=to_index)
        self._swap_mapping_index(from_index=from_index, to_index=to_index)

    def _swap_mapping_index(self, from_index: int, to_index: int):
        for seq_ids2index in self.mamba_cache_indices_mapping.values():
            for seq_id, index in seq_ids2index.items():
                if from_index == index:
                    seq_ids2index.update({seq_id: to_index})
                elif to_index == index:
                    seq_ids2index.update({seq_id: from_index})

    def _update_mapping_index(self, from_index: int, to_index: int):
        for seq_ids2index in self.mamba_cache_indices_mapping.values():
            for seq_id, index in seq_ids2index.items():
                if from_index == index:
                    seq_ids2index.update({seq_id: to_index})
                    return

    def _release_finished_requests(self,
                                   finished_seq_groups_req_ids: List[str]):
        for req_id in finished_seq_groups_req_ids:
            if req_id in self.mamba_cache_indices_mapping:
                self.mamba_cache_indices_mapping.pop(req_id)

    def _first_free_index_in_mamba_cache(
            self, indices_range: Optional[List[int]] = None) -> int:
        assert self.mamba_cache is not None
        if indices_range is None:
            max_possible_batch_size = self.mamba_cache[0].shape[1]
            indices_range = list(range(max_possible_batch_size))
        all_occupied_indices = self._get_all_occupied_indices()
        for i in indices_range:
            if i not in all_occupied_indices:
                return i
        raise Exception("Couldn't find a free spot in the mamba cache! This"
                        "should never happen")
