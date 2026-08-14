# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
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

# TEMP stream-topology instrumentation (read-only). Enable with
# VLLM_DEBUG_MOE_STREAMS=1; grep logs for "[MOE-STREAM-DEBUG]". The detailed
# lines fire ONLY on the MULTI_STREAM_OVERLAPPED (decode) path, so the budget
# is not consumed by warmup/prefill NO_OVERLAP calls. Remove once the stream
# question is resolved.
_STREAM_DEBUG = os.getenv("VLLM_DEBUG_MOE_STREAMS", "0") == "1"
_STREAM_DEBUG_MAX = 40
_stream_debug_count = 0

# Order-distribution tracking, used to prove the MULTI_STREAM path is reached
# and to separate it from the warmup/prefill NO_OVERLAP calls.
_order_counts: dict[str, int] = {}
_first_multi_logged = False


def _fmt_stream(stream: object) -> str:
    if stream is None:
        return "None"
    return f"id={id(stream):#x} cuda_stream={getattr(stream, 'cuda_stream', None)}"


def _stream_debug(msg: str) -> None:
    global _stream_debug_count
    if _stream_debug_count >= _STREAM_DEBUG_MAX:
        return
    _stream_debug_count += 1
    logger.warning("[MOE-STREAM-DEBUG] %s", msg)


def _track_order(order_name: str) -> None:
    """Count SharedExpertsOrder occurrences; log the first MULTI_STREAM once."""
    global _first_multi_logged
    _order_counts[order_name] = _order_counts.get(order_name, 0) + 1
    if order_name == "MULTI_STREAM_OVERLAPPED" and not _first_multi_logged:
        _first_multi_logged = True
        logger.warning(
            "[MOE-STREAM-DEBUG] FIRST MULTI_STREAM_OVERLAPPED seen "
            "(NO_OVERLAP count so far=%d, full order counts=%s)",
            _order_counts.get("NO_OVERLAP", 0),
            _order_counts,
        )


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
            # One pair per DBO ubatch id. `_input_ready_event` forks the aux
            # stream (aux waits until the main stream has produced the input);
            # `_output_ready_event` joins it back before the output is consumed.
            # Both the aux stream and the routed path only READ the shared input,
            # so no snapshot is taken here: the routed path copies the input
            # before its in-place kernel overwrites it (see
            # MoERunner._maybe_copy_routed_input). Ordering is via CUDA events
            # (record()/wait()), which are capture-safe; record_stream()/
            # wait_stream() are illegal under HIP/CUDA graph capture and are
            # deliberately avoided.
            self._input_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
            self._output_ready_event = [torch.cuda.Event(), torch.cuda.Event()]

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
        order = self._determine_shared_experts_order(shared_experts_input)
        if _STREAM_DEBUG:
            _track_order(order.name)
            if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
                _stream_debug(
                    f"maybe_forward_async: order={order.name} "
                    f"main(current_stream)={_fmt_stream(current_stream())} "
                    f"aux(self._stream)={_fmt_stream(self._stream)}"
                )
        if order != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            return False
        assert self._stream is not None
        idx = self._output_idx
        assert self._output[idx] is None

        # The aux stream reads ``shared_experts_input`` directly -- no snapshot.
        # Some models (e.g. Qwen3.5) alias this tensor with hidden_states and the
        # routed kernel overwrites it in place, but the routed path copies its
        # input first (MoERunner._maybe_copy_routed_input), so nothing ever
        # writes this tensor while the aux stream reads it. Only the fork
        # (input_ready) and join (output_ready) events are needed; keeping the
        # copy off both streams' waits is what preserves the overlap.
        self._input_ready_event[idx].record(current_stream())
        with torch.cuda.stream(self._stream):
            self._input_ready_event[idx].wait(self._stream)
            if _STREAM_DEBUG:
                _stream_debug(
                    "shared MLP (self._layer) running on "
                    f"{_fmt_stream(torch.cuda.current_stream())}"
                )
            self._output[idx] = self._layer(shared_experts_input)
            self._output_ready_event[idx].record(self._stream)
        return True

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
