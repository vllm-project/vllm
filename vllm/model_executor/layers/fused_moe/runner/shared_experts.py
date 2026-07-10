# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)

logger = init_logger(__name__)


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (1,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (2,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (3,)


class SharedExperts(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        enable_dbo: bool,
        mk_can_overlap_shared_experts: Callable[[], bool],
    ):
        super().__init__()

        # The SharedExperts need to handle DBO since they can be called from
        # an MK's finalize method.  We keep a list of outputs indexed by current
        # DBO ubatch id to handle this case.  If DBO is not enabled, the
        # index is always 0 and the second output list element is ignored.
        self.enable_dbo = enable_dbo
        self._output: list[torch.Tensor | None] = [None, None]
        self._layer = layer
        self._moe_config = moe_config

        self._mk_can_overlap_shared_experts = mk_can_overlap_shared_experts

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")

        # Cross-stream sync via CUDA events (record()/wait()), NOT
        # record_stream()/wait_stream(). Events are legal inside cudagraph
        # capture; record_stream (caching-allocator async free) and wait_stream
        # are not, and cause hangs on ROCm under capture. One (fork, join) pair
        # per DBO ubatch id. `_fork_event[i]` marks the main-stream point the
        # aux stream must wait for before running; `_join_event[i]` marks aux
        # completion the main stream waits for before using the shared output.
        if self._stream is not None:
            self._fork_event = [torch.cuda.Event(), torch.cuda.Event()]
            self._join_event = [torch.cuda.Event(), torch.cuda.Event()]

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        parallel_config = self._moe_config.moe_parallel_config
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend != "allgather_reducescatter"
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP

        if self._mk_can_overlap_shared_experts():
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        should_run_shared_in_aux_stream = (
            current_platform.is_cuda_alike()
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def launch_overlapped(self, shared_experts_input: torch.Tensor) -> bool:
        """Enqueue shared experts on the aux stream WITHOUT joining back.

        Returns True if work was launched on the aux stream (the caller must
        then call :meth:`join_overlapped` after enqueuing the routed kernel),
        or False if the multi-stream order does not apply (caller should fall
        back to sequential execution).

        Submits shared-expert work to the aux stream *before* the routed
        kernel/collective so it overlaps the DP all-gather + fused MoE.
        Ordering is via CUDA events (capture-safe), so this works under
        cudagraph capture; wait_stream/record_stream would not.

        The fork event is recorded on the main stream first, so the aux stream
        waits for `shared_experts_input` to be ready before it runs. No
        record_stream() is needed: the join event ordering in join_overlapped
        keeps `shared_experts_input` alive until the aux stream is done, and the
        main stream is event-ordered after the aux output before consuming it.
        """
        if (
            self._determine_shared_experts_order(shared_experts_input)
            != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        ):
            return False
        assert self._stream is not None
        idx = self._output_idx
        assert self._output[idx] is None
        self._fork_event[idx].record(current_stream())
        with torch.cuda.stream(self._stream):
            self._fork_event[idx].wait(self._stream)
            self._output[idx] = self._layer(shared_experts_input)
            self._join_event[idx].record(self._stream)
        return True

    def join_overlapped(self) -> None:
        """Join the aux stream back after a :meth:`launch_overlapped` call.

        Makes the main stream wait (via event) for the aux-stream shared-expert
        work to finish before its output is consumed.
        """
        assert self._stream is not None
        self._join_event[self._output_idx].wait(current_stream())

    @property
    def _output_idx(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    @property
    def output(self) -> torch.Tensor:
        assert self._output[self._output_idx] is not None
        output = self._output[self._output_idx]
        self._output[self._output_idx] = None
        return output

    def forward(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            return None

        assert self._output[self._output_idx] is None

        self._output[self._output_idx] = self._layer(shared_experts_input)

        assert self._output[self._output_idx] is not None
