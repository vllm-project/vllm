# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import get_ep_group
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import set_num_sms as deep_gemm_set_num_sms
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass
class UBatchSlice:
    request_slice: slice
    token_slice: slice

    def is_empty(self) -> bool:
        return (
            self.request_slice.start == self.request_slice.stop
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start


UBatchSlices: TypeAlias = list[UBatchSlice]


class SMControlContextManager:
    def __init__(
        self,
        comm_sms: int,
        set_comm_sms: Callable[[int], None],
        set_compute_sms: Callable[[int], None],
    ):
        """
        Context manager for controlling SM (Streaming Multiprocessor)
        allocation. Upon entering the context, it sets the number of SMs
        allocated for communication and computation to comm_sms and
        total_sms - comm_sms respectively. Upon exiting, it restores the
        allocation to use all available SMs (i.e. total_sms).

        Args:
            comm_sms (int): The number of SMs to allocate for communication.
                (The remainder will be used for computation.)
            set_comm_sms (Callable[[int], None]):
                A function that sets the number of SMs for communication.
            set_compute_sms (Callable[[int], None]):
                A function that sets the number of SMs for computation.
        """

        assert current_platform.is_cuda() or current_platform.is_rocm(), (
            "SM/CU control is supported on CUDA and ROCm platforms"
        )
        device = torch.accelerator.current_device_index()
        total_sms = num_compute_units(device)

        assert comm_sms < total_sms
        self.total_sms = total_sms
        self.compute_sms = total_sms - comm_sms
        self.comm_sms = comm_sms
        self.set_comm_sms = set_comm_sms
        self.set_compute_sms = set_compute_sms

    def __enter__(self):
        self.set_comm_sms(self.comm_sms)
        self.set_compute_sms(self.compute_sms)

    def __exit__(self, exc_type, exc_value, traceback):
        self.set_comm_sms(self.total_sms)
        self.set_compute_sms(self.total_sms)


def create_sm_control_context(vllm_config: VllmConfig) -> SMControlContextManager:
    """Reserve SMs for communication kernels while microbatches overlap."""
    comm_sms: int = envs.VLLM_DBO_COMM_SMS
    rocm_deepep_ht_dbo = (
        current_platform.is_rocm()
        and vllm_config.parallel_config.enable_dbo
        and vllm_config.parallel_config.all2all_backend == "deepep_high_throughput"
    )
    if rocm_deepep_ht_dbo:
        # On ROCm, reserving CUs for DeepEP HT communication under DBO
        # corrupts DP+EP generation accuracy. Keep the backend active, but
        # leave all CUs visible to the compute and communication kernels.
        comm_sms = 0

    set_comm_sms = lambda sms: None
    if vllm_config.parallel_config.enable_expert_parallel:
        # Currently only DeepEP highthroughput supports SM control so this
        # only affects that case.
        ep_group = get_ep_group()
        device_communicator = ep_group.device_communicator
        all2all_manager = None
        if device_communicator is not None:
            all2all_manager = device_communicator.all2all_manager

        if all2all_manager is not None:
            max_sms_used = all2all_manager.max_sms_used()
            if max_sms_used is not None:
                comm_sms = min(comm_sms, max_sms_used)

        if comm_sms > 0 and all2all_manager is not None:
            set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

    # TODO(lucas): support other kernels besides DeepGEMM
    set_compute_sms = lambda sms: None
    if has_deep_gemm() and comm_sms > 0:
        set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)

    return SMControlContextManager(
        comm_sms=comm_sms,
        set_comm_sms=set_comm_sms,
        set_compute_sms=set_compute_sms,
    )


def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool:
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


# This pads the last ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)

    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    if not should_ubatch:
        return None, None

    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    token_split_points = [split_point * i for i in range(1, num_ubatches)]

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices = []
    start_token = 0

    # Add the end point to the split points to make iteration easier
    all_points = token_split_points + [cu_num_tokens[-1]]

    for end_token in all_points:
        token_slice = slice(start_token, end_token)

        # Determine request slices using exclusive stop semantics
        # Ubatch includes requests whose tokens overlap [start_token, end_token)

        # Start at the request that contains the start_token
        # or the request starting exactly at start_token (if on boundary)
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # Stop at the request that starts at or after end_token
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        req_slice = slice(req_start, req_stop)
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        start_token = end_token

    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    return ubatch_slices, ubatch_slices_padded


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )
    # NOTE: last token can be outside of the last request if we have CG padding.

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    # Read raw fields to avoid triggering the deprecated D2H-syncing properties.
    seq_lens_cpu = (
        attn_metadata._seq_lens_cpu[request_slice]
        if attn_metadata._seq_lens_cpu is not None
        else None
    )
    seq_lens_cpu_upper_bound = (
        attn_metadata.seq_lens_cpu_upper_bound[request_slice]
        if attn_metadata.seq_lens_cpu_upper_bound is not None
        else None
    )
    num_computed_tokens_cpu = (
        attn_metadata._num_computed_tokens_cpu[request_slice]
        if attn_metadata._num_computed_tokens_cpu is not None
        else None
    )

    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu.clone()
            seq_lens_cpu[-1] -= tokens_skipped
        if seq_lens_cpu_upper_bound is not None:
            seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound.clone()
            seq_lens_cpu_upper_bound[-1] -= tokens_skipped

    assert seq_lens_cpu_upper_bound is not None
    # Preserve the max_seq_len override set during CUDA-graph capture so
    # the attention backend selects the correct kernel for SWA layers.
    max_seq_len = max(int(seq_lens_cpu_upper_bound.max()), attn_metadata.max_seq_len)

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results
