# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
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

    # Called by modular kernel.
    INTERNAL = (2,)

    # Called right before quant_method is executed.
    BEFORE_QUANT_METHOD = (3,)

    # Called right after quant_method is executed (possibly with streaming).
    AFTER_QUANT_METHOD = (4,)


class SharedExperts:
    def __init__(
        self,
        shared_experts: torch.nn.Module,
        moe_config: FusedMoEConfig,
        quant_method: QuantizeMethodBase,
        reduce_results: bool,
    ):
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )

        # quant_method must be a FusedMoEMethodBase but we can't use the type
        # due to circular imports.
        assert isinstance(quant_method, FusedMoEMethodBase)

        self._output: torch.Tensor | None = None
        self._shared_experts = shared_experts
        self._moe_config = moe_config
        self._quant_method = quant_method
        self._reduce_results = reduce_results
        self._use_dp_chunking = moe_config.moe_parallel_config.use_dp_chunking

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            self._shared_experts_stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self._shared_experts_stream = aux_stream()
            if self._shared_experts_stream is not None:
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

    @property
    def _has_external_experts(self) -> bool:
        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        backend = self._moe_config.moe_parallel_config.all2all_backend
        return self._shared_experts is not None and not (
            (
                self._moe_config.moe_parallel_config.enable_eplb
                and backend != "allgather_reducescatter"
            )
            or self._moe_config.moe_parallel_config.use_fi_all2allv_kernels
        )

    @property
    def _has_mk_owned_shared_experts(self) -> bool:
        return (
            not self._quant_method.mk_owns_shared_expert
            and self._shared_experts is not None
        )

    @property
    def _must_reduce_shared_expert_outputs(self) -> bool:
        return (
            self._reduce_results
            and self._quant_method.moe_mk is not None
            and self._quant_method.moe_mk.output_is_reduced()
        )

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[SharedExpertsOrder, bool]:
        if self._shared_experts is None:
            return SharedExpertsOrder.NONE, False

        if self._has_external_experts and not self._use_dp_chunking:
            return SharedExpertsOrder.EXTERNAL, False

        if (
            not self._has_mk_owned_shared_experts
            or not self._moe_config.moe_parallel_config.use_all2all_kernels
        ):
            return SharedExpertsOrder.INTERNAL, False

        allow_shared_experts_stream = (
            current_platform.is_cuda()
            and self._has_mk_owned_shared_experts
            and not self._use_dp_chunking
            and self._shared_experts_stream is not None
            and hidden_states.shape[0]
            <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        )

        # Check if we need to run shared experts before matrix multiply because
        # matrix multiply may modify the hidden_states.
        run_shared_experts_before = (
            self._has_mk_owned_shared_experts and not allow_shared_experts_stream
        )

        if run_shared_experts_before:
            return SharedExpertsOrder.BEFORE_QUANT_METHOD, False
        else:
            return SharedExpertsOrder.AFTER_QUANT_METHOD, allow_shared_experts_stream

    def _call_with_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        assert self._shared_experts_stream is not None
        assert self._moe_config.disable_inplace

        # Record that the clone will be used by shared_experts_stream
        # to avoid gc issue from deallocation of hidden_states_clone
        # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
        # NOTE: We don't need shared_output.record_stream(current_stream())
        # because we synch the streams before using shared_output.
        shared_experts_input.record_stream(self._shared_experts_stream)

        # Mark sync start point for the separate shared experts
        # stream here since we want to run in parallel with the
        # router/gate (next op below)
        self._shared_experts_stream.wait_stream(current_stream())

        # Run shared experts in parallel on a separate stream
        # NOTE: We start the separate stream here and mark the
        # sync end point immediately after it is done. This is
        # important to avoid excessive stream allocations by the cuda
        # graph replay later.
        with torch.cuda.stream(self._shared_experts_stream):
            # Note that hidden_states clone() is necessary here to avoid
            # conflict with the main stream
            output = self._shared_experts(shared_experts_input)
        current_stream().wait_stream(self._shared_experts_stream)

        return output

    def _maybe_reduce_shared_out(self, shared_out: torch.Tensor) -> torch.Tensor:
        # Reduce shared expert outputs if necessary, since the MLP
        # should have been created with reduce_results=False.
        if (
            self._must_reduce_shared_expert_outputs
            and get_tensor_model_parallel_world_size() > 1
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out

    @property
    def output(self) -> torch.Tensor | None:
        assert (self._shared_experts is None) == (self._output is None)
        output = self._output
        self._output = None
        return output

    def apply(
        self,
        shared_experts_input: torch.Tensor,
        order: SharedExpertsOrder,
    ):
        experts_order, use_shared_experts_stream = self._determine_shared_experts_order(
            shared_experts_input,
        )

        if order != experts_order:
            return None

        assert self._shared_experts is not None
        assert self._output is None

        if order == SharedExpertsOrder.AFTER_QUANT_METHOD and use_shared_experts_stream:
            self._output = self._call_with_shared_experts_stream(shared_experts_input)
        else:
            self._output = self._shared_experts(shared_experts_input)

        if order == SharedExpertsOrder.EXTERNAL:
            # TODO: figure out how to combine this with maybe_reduce_output?
            # or get rid of it completely.
            assert self._output is not None
            self._output = self._maybe_reduce_shared_out(self._output)

        # TODO(bnell): potentially do AFTER reduce here insteed of in runner.
