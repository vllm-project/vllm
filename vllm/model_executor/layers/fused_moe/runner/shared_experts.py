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

        if self._stream is not None:
            # One pair per DBO ubatch id.
            self._input_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
            self._output_ready_event = [torch.cuda.Event(), torch.cuda.Event()]

            # Persistent, module-owned copy targets for the shared-experts input
            # (one per DBO ubatch id). Sized to the largest cudagraph decode
            # batch so the copy in `maybe_forward_async` is allocation-free and
            # keeps a stable address under graph capture.
            self._input_buffer: list[torch.Tensor | None] = [None, None]
            self._max_input_rows = self._moe_config.max_capture_size

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config

    @property
    def _disable_shared_experts_overlap(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-safe backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain

        # Both these comm backends have been shown to be safe for shared expert overlap.
        _EPLB_OVERLAP_SAFE_BACKENDS = (
            "allgather_reducescatter",
            "flashinfer_nvlink_one_sided",
        )

        parallel_config = self._moe_config.moe_parallel_config
        if getattr(self._layer, "shard_sequence_parallel", False):
            # TODO: we may enable this to optimize further
            return True
        return (
            parallel_config.enable_eplb
            and parallel_config.all2all_backend not in _EPLB_OVERLAP_SAFE_BACKENDS
        ) or parallel_config.use_fi_nvl_two_sided_kernels

    @property
    def _should_enable_stream_overlap_heuristic(self) -> bool:
        # On ROCm, empirically it's shown that only DPA deployments benefit from
        # multi-stream shared experts
        if not current_platform.is_rocm():
            return True
        return self._moe_config.moe_parallel_config.dp_size > 1

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
            and self._should_enable_stream_overlap_heuristic
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def maybe_forward_async(self, shared_experts_input: torch.Tensor) -> bool:
        """Enqueue shared experts on the aux stream without waiting for them.

        Returns true if the shared experts were enqueued, false otherwise. Call
        `wait` to wait for the shared experts to finish if this returns true.
        """
        if (
            self._determine_shared_experts_order(shared_experts_input)
            != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        ):
            return False
        assert self._stream is not None
        idx = self._output_idx
        assert self._output[idx] is None

        # Some models (e.g. Qwen3.5) pass the same tensor as both hidden_states
        # and the shared-experts input, then mutate it in place inside the routed
        # experts. Copy the input into a private, module-owned buffer so the aux
        # stream reads a stable copy while the main stream is free to overwrite
        # the original. The copy runs on the main stream *before* the routed
        # experts are launched, so program order guarantees it captures the
        # pre-mutation values; the input_ready event (recorded after the copy)
        # then holds the aux stream until the copy is done.
        #
        # A persistent buffer (rather than a fresh clone) needs no
        # record_stream() to stay alive for the aux stream, so ordering is done
        # entirely with CUDA events. This is what keeps the path cudagraph-safe:
        # record_stream()/wait_stream() are illegal under HIP/CUDA graph capture,
        # whereas event record()/wait() are captured correctly.
        shared_experts_input = self._copy_input_to_buffer(shared_experts_input)

        self._input_ready_event[idx].record(current_stream())
        with torch.cuda.stream(self._stream):
            self._input_ready_event[idx].wait(self._stream)
            self._output[idx] = self._layer(shared_experts_input)
            self._output_ready_event[idx].record(self._stream)
        return True

    def _copy_input_to_buffer(self, x: torch.Tensor) -> torch.Tensor:
        """Copy the aliased shared-experts input into a persistent buffer.

        Returns a view of a module-owned buffer holding a private copy of ``x``,
        so the aux stream never reads the storage the routed experts mutate in
        place. The buffer is sized to the largest cudagraph decode batch and
        reused across steps, so the copy is allocation-free on the hot path and
        has a stable address under graph capture. Growth only happens eagerly
        (during warmup or eager execution), never mid-capture: cudagraph decode
        batches are captured largest-first and each size is warmed up eagerly
        before capture, so replayed sizes never exceed the pre-allocated buffer.

        Args:
            x: The shared-experts input, aliased with hidden_states.

        Returns:
            A ``[x.shape[0], ...]`` view of the persistent buffer, filled with a
            copy of ``x`` on the current (main) stream.
        """
        idx = self._output_idx
        buf = self._input_buffer[idx]
        num_rows = x.shape[0]
        if (
            buf is None
            or buf.shape[0] < num_rows
            or buf.shape[1:] != x.shape[1:]
            or buf.dtype != x.dtype
        ):
            rows = max(num_rows, self._max_input_rows)
            buf = torch.empty((rows, *x.shape[1:]), dtype=x.dtype, device=x.device)
            self._input_buffer[idx] = buf
        out = buf[:num_rows]
        out.copy_(x)
        return out

    def wait(self) -> None:
        """Block the main stream until `maybe_forward_async` output is ready."""
        assert self._stream is not None
        self._output_ready_event[self._output_idx].wait(current_stream())

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
