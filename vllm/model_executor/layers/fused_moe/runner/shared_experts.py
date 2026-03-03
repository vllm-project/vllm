# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
)

logger = init_logger(__name__)


class SharedExpertsOrder(IntEnum):
    # No shared experts.
    NONE = (0,)

    # Get rid of this one?  combine with BEFORE?
    # Note: this might be important for torch.compile reasons. Can
    # get rid of it after _moe_forward is undone.
    EXTERNAL = (1,)

    # No overlap - defensively called before MK.
    NO_OVERLAP = (2,)

    # Overlapped with dispatch/combine in DP/EP - called by the MK.
    MK_INTERNAL_OVERLAPPED = (3,)

    # Overlapped with the gate, router, experts in aux stream.
    MULTI_STREAM_OVERLAPPED = (4,)


class SharedExperts:
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        quant_method: QuantizeMethodBase,
    ):
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )

        # quant_method must be a FusedMoEMethodBase but we can't use the type
        # due to circular imports.
        assert isinstance(quant_method, FusedMoEMethodBase)

        self._output: torch.Tensor | None = None
        self._layer = layer
        self._moe_config = moe_config
        self._quant_method = quant_method
        self._use_dp_chunking = moe_config.moe_parallel_config.use_dp_chunking

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            self._stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self._stream = aux_stream()
            if self._stream is not None:
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

    @property
    def _has_external_experts(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        backend = self._moe_config.moe_parallel_config.all2all_backend
        return not (
            (
                self._moe_config.moe_parallel_config.enable_eplb
                and backend != "allgather_reducescatter"
            )
            or self._moe_config.moe_parallel_config.use_fi_nvl_two_sided_kernels
        )

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        if self._has_external_experts and not self._use_dp_chunking:
            return SharedExpertsOrder.EXTERNAL

        if self._quant_method.mk_owns_shared_expert:
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED

        should_run_shared_in_aux_stream = (
            current_platform.is_cuda()
            and not self._use_dp_chunking
            and self._stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        if should_run_shared_in_aux_stream:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        else:
            return SharedExpertsOrder.NO_OVERLAP

    def maybe_setup_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if experts_order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            assert self._stream is not None
            assert self._moe_config.disable_inplace

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            shared_experts_input.record_stream(self._stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            self._stream.wait_stream(current_stream())

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: assert that maybe_setup_shared_experts_stream has been called.

        # Run shared experts in parallel on a separate stream
        # NOTE: We start the separate stream here and mark the
        # sync end point immediately after it is done. This is
        # important to avoid excessive stream allocations by the cuda
        # graph replay later.
        with torch.cuda.stream(self._stream):
            # Note that hidden_states clone() is necessary here to avoid
            # conflict with the main stream
            output = self._layer(shared_experts_input)
        current_stream().wait_stream(self._stream)

        return output

    def _maybe_reduce_shared_output(self, output: torch.Tensor) -> torch.Tensor:
        if (
            self._quant_method.moe_kernel is not None
            and self._quant_method.moe_kernel.output_is_reduced()
            and get_tensor_model_parallel_world_size() > 1
        ):
            output = tensor_model_parallel_all_reduce(output)
        return output

    @property
    def output(self) -> torch.Tensor:
        assert self._output is not None
        output = self._output
        self._output = None
        if output is not None:
            output = self._maybe_reduce_shared_output(output)
        return output

    def apply(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
    ):
        experts_order = self._determine_shared_experts_order(shared_experts_input)

        if order != experts_order:
            return None

        assert self._output is None

        if order == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            self._output = self._run_in_aux_stream(shared_experts_input)
        else:
            self._output = self._shared_experts(shared_experts_input)
