# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
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


# Largest MoE input (tokens) whose graph captures the routed chain before the
# shared-expert branch (see `_defer_capture_launch`). 64 = 8 requests x 8 rows,
# i.e. decode with 7 speculative tokens: A/B on MiniMax-M3.1 MXFP4 (TP4,
# MI355X) gained at <= 8 requests and lost at 16 (+2.9% step time; +0.4% at 12),
# where the routed GEMMs are long enough that the shared branch captured second
# starts late enough to delay the join. The token count is static per captured
# graph, so this is a capture-time decision.
_LATE_FORK_MAX_TOKENS = 64


def _is_capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


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

        # HIP graphs keep the first captured branch of a fork on the parent's
        # stream and move later branches to another queue, which costs the
        # branch ~3 us of extra start latency and an extra completion hop at the
        # join. The routed experts are on the critical path and the shared
        # experts are not, so on ROCm the routed chain is captured first for small
        # decode batches (_LATE_FORK_MAX_TOKENS).
        self._defer_capture_launch = current_platform.is_rocm()
        self._deferred_input: list[torch.Tensor | None] = [None, None]

        # With fully-sharded LoRA the shared experts' LoRA layers issue TP
        # collectives themselves, so they must not run on the aux stream
        # (see _disable_shared_experts_overlap).
        vllm_config = get_current_vllm_config_or_none()
        lora_config = vllm_config.lora_config if vllm_config is not None else None
        self._lora_fully_sharded = (
            lora_config is not None and lora_config.fully_sharded_loras
        )

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
        else:
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once("Enabled separate cuda stream for MoE shared_experts")
                # One pair per DBO ubatch id to sync aux and main stream.
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
        #   - we are using fully-sharded LoRA: the shared experts' LoRA layers
        #     all-gather / all-reduce across TP, and so does the routed
        #     experts' fused MoE LoRA kernel on the main stream. Two
        #     collectives in flight on the same communicator from different
        #     streams become independent branches under CUDA graph capture,
        #     and replay order is not guaranteed to match across ranks, which
        #     deadlocks NCCL (https://github.com/vllm-project/vllm/issues/60845).
        if self._lora_fully_sharded:
            return True

        # Both these comm backends have been shown to be safe for shared expert overlap.
        _EPLB_OVERLAP_SAFE_BACKENDS = (
            "allgather_reducescatter",
            "deepep_v2",
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

    def maybe_forward_async(
        self, shared_experts_input: torch.Tensor, defer_launch: bool = False
    ) -> bool:
        """Enqueue shared experts on the aux stream without waiting for them.

        With `defer_launch` the aux-stream work is, during CUDA graph capture
        on ROCm only, not enqueued until `launch_deferred` (or `wait`) is
        called; the caller must call one of them after enqueueing the routed
        experts.

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
        self._input_ready_event[idx].record(current_stream())
        if (
            defer_launch
            and self._defer_capture_launch
            and shared_experts_input.shape[0] <= _LATE_FORK_MAX_TOKENS
            and _is_capturing()
        ):
            # The input-ready event is recorded here, so the shared experts
            # still depend only on this point of the main stream. Their launch
            # is deferred to `launch_deferred`, after the routed chain has been
            # captured (see `_defer_capture_launch`).
            self._deferred_input[idx] = shared_experts_input
        else:
            self._launch(idx, shared_experts_input)
        return True

    def _launch(self, idx: int, shared_experts_input: torch.Tensor) -> None:
        with torch.cuda.stream(self._stream):
            self._input_ready_event[idx].wait(self._stream)
            self._output[idx] = self._layer(shared_experts_input)
            self._output_ready_event[idx].record(self._stream)

    def launch_deferred(self) -> None:
        """Launch shared experts whose launch `maybe_forward_async` deferred."""
        idx = self._output_idx
        shared_experts_input = self._deferred_input[idx]
        if shared_experts_input is not None:
            self._deferred_input[idx] = None
            self._launch(idx, shared_experts_input)

    def wait(self) -> None:
        """Block the main stream until `maybe_forward_async` output is ready."""
        assert self._stream is not None
        self.launch_deferred()
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
