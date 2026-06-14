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

        # Replace the framework's `Stream.wait_stream`-based
        # shared/routed-expert overlap with the event-based
        # `torch.cuda.Event` record/wait pattern under CUDA graph capture.
        # The framework's existing `wait_stream`-based path silently
        # degenerates to sequential under CUDA-graph capture (observed
        # under nsys profiling: ALL MoE kernels land on the same
        # streamId), while the
        # event-based pattern actually parks shared-expert work on a
        # side-stream concurrent with the routed-expert FP4 BMM. When
        # the env var is unset (default), `_use_events` is False and
        # behaviour is bit-identical to the original framework path.
        self._use_events: bool = (
            envs.VLLM_MOE_SHARED_EXPERTS_TWO_STREAM
            and self._stream is not None
            and current_platform.is_cuda()
        )
        # Two pre/post event pairs -- indexed by ubatch id (DBO can run
        # two ubatches in parallel; only index 0 is used when DBO is
        # disabled).
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event] | None] = [
            None,
            None,
        ]
        if self._use_events:
            self._events[0] = (torch.cuda.Event(), torch.cuda.Event())
            if enable_dbo:
                self._events[1] = (torch.cuda.Event(), torch.cuda.Event())
            logger.debug_once(
                "Enabled event-based MoE shared/routed two-stream overlap"
            )

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
            current_platform.is_cuda()
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)

            if self._use_events:
                # Defer the cross-stream sync to
                # `_run_in_aux_stream`, where we use the event-based
                # `torch.cuda.Event` record/wait pair. `wait_stream`
                # collapses to sequential under CUDA-graph capture; we
                # avoid recording any `wait_stream` edge here so the
                # captured graph can express true parallelism between
                # the shared-expert and routed-expert paths.
                return

            # Mark sync start point for the aux stream since we will
            # run in parallel with router/gate.
            self._stream.wait_stream(current_stream())

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: assert that maybe_sync_shared_experts_stream has been called.

        if self._use_events:
            # Event-based two-stream pattern (mirrors
            # the MLA path's `unified_mla_kv_cache_update` /
            # `unified_mla_attention_with_output` flow). Under CUDA
            # graph capture, `Stream.wait_stream(other)` lowers to a
            # plain barrier on the captured graph, serializing both
            # streams (nsys profiling confirms ALL MoE kernels land on the
            # same streamId). `Event.record()` / `Event.wait()`
            # captures inter-stream dependencies as DAG edges, which
            # CUDA replay honours as real parallelism on the GPU.
            #
            # Protocol:
            #   1. Record `pre_event` on the default stream so the aux
            #      stream waits for the producer of
            #      `shared_experts_input` (e.g. the layernorm + clone)
            #      before reading it.
            #   2. On the aux stream, wait for `pre_event`, run the
            #      shared-expert layer (BF16 NVJet GEMMs on Kimi-K2.6),
            #      then record `post_event`.
            #   3. The default stream proceeds with the routed-expert
            #      FP4 BMMs concurrently. The join (`post_event.wait()`
            #      on the default stream) is performed by the caller
            #      via `join_event_overlap()` after `apply_monolithic`
            #      / `apply` returns, just before the
            #      `shared_output + fused_output` add.
            assert self._events[self._output_idx] is not None
            pre_event, post_event = self._events[self._output_idx]
            pre_event.record()
            with torch.cuda.stream(self._stream):
                pre_event.wait()
                output = self._layer(shared_experts_input)
                post_event.record()
            return output

        # Run shared experts in parallel on a separate stream.
        with torch.cuda.stream(self._stream):
            output = self._layer(shared_experts_input)
        current_stream().wait_stream(self._stream)

        return output

    def join_event_overlap(self) -> None:
        """Join the event-based shared-expert overlap.

        When the event-based pattern (`_use_events == True`) was used
        to launch the shared-expert layer onto the aux stream, the
        default stream must wait on the `post_event` before reading
        `self.output`. This is invoked by `MoERunner._apply_quant_method`
        after the routed-expert kernel(s) have been enqueued on the
        default stream so the join point captures all of the routed-side
        work as parallel with the shared-side work in the CUDA graph.

        When events are not in use, this is a no-op so callers can
        invoke it unconditionally.
        """
        if not self._use_events:
            return
        events = self._events[self._output_idx]
        if events is None:
            return
        _, post_event = events
        post_event.wait()

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

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output[self._output_idx] = self._run_in_aux_stream(
                shared_experts_input
            )
        else:
            self._output[self._output_idx] = self._layer(shared_experts_input)

        assert self._output[self._output_idx] is not None
