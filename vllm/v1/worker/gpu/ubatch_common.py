# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-agnostic microbatching (DBO) helpers.

This module must not import `vllm.forward_context` or other model runner
modules: `forward_context` and the V1 model runner import it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import ParallelConfig
from vllm.distributed import get_ep_group
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import set_num_sms as deep_gemm_set_num_sms
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.platform_utils import num_compute_units


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
        """Context manager for controlling SM (Streaming Multiprocessor)
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


def create_sm_control_context(
    parallel_config: ParallelConfig,
) -> SMControlContextManager:
    """Reserve SMs for communication kernels while microbatches overlap."""
    comm_sms: int = envs.VLLM_DBO_COMM_SMS
    rocm_deepep_ht_dbo = (
        current_platform.is_rocm()
        and parallel_config.enable_dbo
        and parallel_config.all2all_backend == "deepep_high_throughput"
    )
    if rocm_deepep_ht_dbo:
        # On ROCm, reserving CUs for DeepEP HT communication under DBO
        # corrupts DP+EP generation accuracy. Keep the backend active, but
        # leave all CUs visible to the compute and communication kernels.
        comm_sms = 0

    set_comm_sms = lambda sms: None
    if parallel_config.enable_expert_parallel:
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


def get_num_ubatches(parallel_config: ParallelConfig) -> int:
    """How many microbatches a step is split into; 1 when microbatching is off."""
    return parallel_config.num_ubatches if parallel_config.use_ubatching else 1


def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


def compute_ubatch_slices(
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
    num_reqs_padded: int,
    num_tokens_padded: int,
    num_ubatches: int,
    uniform_token_count: int | None = None,
) -> UBatchSlices:
    """Split a DP-padded batch into the slices its microbatches run on.

    Splitting at the midpoint of the DP-padded token count leaves the trailing
    microbatch anywhere from full to entirely inside the padding. Clamping its
    request slice onto the last request leaves it holding that request with
    zero query tokens -- the shape a straddling request already takes -- so it
    stays well-formed with no work to do, like a dummy run.

    A uniform batch of multi-token queries (spec decode verification) is split
    on a request boundary instead, so each microbatch stays a uniform decode.
    `uniform_token_count` must be agreed across DP ranks.
    """
    split_point = num_tokens_padded // num_ubatches
    if uniform_token_count is not None and uniform_token_count > 1:
        reqs_per_ubatch = num_tokens_padded // (uniform_token_count * num_ubatches)
        if reqs_per_ubatch > 0:
            split_point = reqs_per_ubatch * uniform_token_count
    token_bounds = [split_point * i for i in range(num_ubatches)]
    token_bounds.append(num_tokens_padded)

    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, np.int32)
    np.cumsum(num_scheduled_tokens, out=cu_num_tokens[1:])

    ubatch_slices = []
    for i in range(num_ubatches):
        start, stop = token_bounds[i], token_bounds[i + 1]
        # The microbatch holds the requests overlapping [start, stop); the
        # trailing one also takes every padded request.
        req_start = int(np.searchsorted(cu_num_tokens, start, side="right")) - 1
        req_stop = num_reqs_padded
        if i < num_ubatches - 1:
            req_stop = int(np.searchsorted(cu_num_tokens, stop, side="left"))
        ubatch_slices.append(
            UBatchSlice(
                slice(min(req_start, num_reqs - 1), min(req_stop, num_reqs_padded)),
                slice(start, stop),
            )
        )
    return ubatch_slices
