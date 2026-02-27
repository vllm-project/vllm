# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from vllm.distributed import (
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op,
)

logger = init_logger(__name__)


def get_layer_from_name(layer_name: str) -> torch.nn.Module:
    forward_context: ForwardContext = get_forward_context()
    if layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    return forward_context.no_compile_layers[layer_name]


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's 'forward_dispatch' method.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    layer = get_layer_from_name(layer_name)
    return layer.runner.forward_dispatch(
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(layer_name)
    return layer.runner.forward_dispatch(
        layer,
        hidden_states,
        router_logits,
        shared_experts_input,
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Output shapes:
    # - fused_out: same as hidden_states (routed experts use transformed size)
    # - shared_out: same as shared_experts_input if provided, else same as
    #               hidden_states
    # (For latent MoE: shared experts use original hidden_size, not latent size)
    fused_out = torch.empty_like(hidden_states)

    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)

    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],  # ?
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    mutates_args=["hidden_states"],  # ?
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class MoERunnerBase(MoERunner):
    """
    Abstract base class providing common functionality for MoE runner implementations.

    This class serves as the foundation for concrete MoE runner implementations by
    providing shared state management and common utilities. It handles:
    - Common initialization and configuration management
    - Shared expert output reduction logic for tensor parallel scenarios
    - Base methods for tensor model parallel reductions
    - Common properties and utility functions used across different runner types

    Concrete subclasses must implement the abstract methods to define their specific
    execution strategies, such as standard execution, chunked processing, or other
    specialized approaches. The base class provides the infrastructure while
    allowing flexibility in the actual MoE computation implementation.

    Key abstract methods that subclasses must implement:
    - reduce_results: Determines whether results should be reduced across ranks
    - forward_impl: The core MoE computation logic specific to each runner type
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_input_transform: torch.nn.Module | None,
        gate: torch.nn.Module | None,
        shared_experts: SharedExperts | None,
        quant_method: FusedMoEMethodBase,
        reduce_results: bool,
        enable_dbo: bool,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.gate = gate
        self.shared_experts = shared_experts
        self.quant_method = quant_method
        self._reduce_results = reduce_results
        self.enable_dbo = enable_dbo
        self.enable_eplb = moe_config.moe_parallel_config.enable_eplb

        # Needed for string -> FusedMoE layer lookup in custom ops.
        self.layer_name = layer.layer_name

        self.forward_entry = self._select_forward(layer)

    def _select_forward(self, layer: torch.nn.Module) -> Callable:
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped forward_impl.
            return _moe_forward if self.shared_experts is None else _moe_forward_shared

        return (
            torch.ops.vllm.moe_forward
            if self.shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    @property
    @abstractmethod
    def reduce_results(self) -> bool:
        raise NotImplementedError

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        return (
            self.quant_method.moe_mk is not None
            and self.quant_method.moe_mk.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def apply_routed_input_transform(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by FusedMoE.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        TODO: For latent MoE bandwidth optimization, fc2_latent_proj could be
        moved inside SharedFusedMoE to all-reduce on the smaller latent
        dimension.

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self.shared_experts is not None else None,
        )

    def _maybe_reduce_output(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        trunc_sizes: list[int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        def trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            return x[..., :trunc_size]

        def reduce_and_trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            return trunc(self.maybe_all_reduce_tensor_model_parallel(x), trunc_size)

        if (
            not self.moe_config.is_sequence_parallel
            and self.reduce_results
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
        ):
            func = reduce_and_trunc
        else:
            func = trunc

        if isinstance(states, tuple):
            return tuple(
                [func(s, trunc_size) for s, trunc_size in zip(states, trunc_sizes)]
            )
        else:
            assert len(trunc_sizes) == 1
            return func(states, trunc_sizes[0])

    def _encode_layer_name(self) -> str:
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim = hidden_states.shape[-1]
        if self.moe_config.hidden_dim != transformed_hidden_dim:
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        if self.shared_experts is not None:
            orig_hidden_dims = [shared_experts_hidden_dim, transformed_hidden_dim]
        else:
            orig_hidden_dims = [transformed_hidden_dim]

        return hidden_states, orig_hidden_dims

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        if self.shared_experts is not None:
            assert shared_experts_input is not None
            self.shared_experts.apply(shared_experts_input, order)

    def _apply_quant_method(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        extra_tensor: torch.Tensor | None,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # Run this before quant_method to avoid inplace issues.
        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.BEFORE_QUANT_METHOD,
        )

        # TODO(bnell): deal with fp4 flashinfer tuple hidden states hack (#30014).
        # Figure out nicer way to do this.
        x_arg = hidden_states if extra_tensor is None else (hidden_states, extra_tensor)

        if self.quant_method.is_monolithic:
            fused_out = self.quant_method.apply_monolithic(
                layer=layer,
                x=x_arg,
                router_logits=router_logits,
            )
        else:
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            fused_out = self.quant_method.apply(
                layer=layer,
                x=x_arg,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.AFTER_QUANT_METHOD,
        )

        return (
            self.shared_experts.output if self.shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_gate(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)
        return router_logits

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            if isinstance(result, tuple):
                result = (result[0], result[1] + zero_expert_output)
            else:
                result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.
        or
        - A tuple of (shared experts output, new hidden_states).

        Calling sequence
        - forward
          - self.forward_entry (_moe_forward or _moe_forward_shared custom op)
            - forward_dispatch
              - forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reasons:
        1. the chunking loop in _forward_impl_chunked cannot be compiled by
           torch.compile
        2. pytorch cannot handle union types in custom op signatures so _moe_forward
           and _moe_forward_shared must be split.

        If _forward_impl_chunked can be implemented via torch.scan we can potentially
        get rid of _moe_forward and _moe_forward_shared and collapse the whole sequence
        into the 'forward' method.
        """

        # Apply transform for routed experts (e.g., latent projection for latent MoE)
        hidden_states, shared_experts_input = self.apply_routed_input_transform(
            hidden_states
        )

        hidden_states, og_hidden_dims = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        fused_output = self.forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            self._encode_layer_name(),
        )

        result = self._maybe_reduce_output(fused_output, og_hidden_dims)

        return self._maybe_add_zero_expert_output(result)

    def forward_dispatch(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # TODO(bnell): this can be removed after MK migration is complete.
        layer.ensure_moe_quant_config_init()

        router_logits = self._maybe_gate(hidden_states, router_logits)

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.EXTERNAL,
        )

        with self._sequence_parallel_context():
            return self.forward_impl(
                layer,
                hidden_states,
                router_logits,
                shared_experts_input,
            )

    @abstractmethod
    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
