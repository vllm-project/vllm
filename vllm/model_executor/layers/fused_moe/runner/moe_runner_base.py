# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

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
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
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
    HAS_OPAQUE_TYPE,
    ModuleName,
    direct_register_custom_op,
)

logger = init_logger(__name__)


def get_layer_from_name(layer_name: str) -> torch.nn.Module:  # FusedMoE
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
    layer = forward_context.no_compile_layers[layer_name]
    # assert isinstance(layer, FusedMoE)
    return layer


# On torch >= 2.11, layer_name is a hoisted ModuleName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | ModuleName
else:
    _layer_name_type = ModuleName if HAS_OPAQUE_TYPE else str


def _resolve_layer_name(layer_name: str | ModuleName) -> str:
    return layer_name.value if isinstance(layer_name, ModuleName) else layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's '_forward_dispatch' method.
# These functions should never be called directly since they do not
# include all the functionality of the MoE layer.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer._runner._forward_dispatch(
        hidden_states,
        router_logits,
        shared_experts_input,
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer._runner._forward_dispatch(
        hidden_states,
        router_logits,
        shared_experts_input,
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
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
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _unpack(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


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
    - _forward_impl: The core MoE computation logic specific to each runner type
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_input_transform: torch.nn.Module | None,
        gate: torch.nn.Module | None,
        shared_experts: torch.nn.Module | None,
        routed_experts: RoutedExperts,
        enable_dbo: bool,
        routed_output_transform: torch.nn.Module | None = None,
        apply_scale_to_output: bool = False,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.gate = gate
        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                mk_owns_shared_expert=routed_experts.quant_method.mk_owns_shared_expert,
            )
        self.routed_experts = routed_experts
        self.enable_dbo = enable_dbo
        self.enable_eplb = moe_config.moe_parallel_config.enable_eplb
        self.apply_scale_to_output = (
            apply_scale_to_output and routed_scaling_factor != 1.0
        )
        self.routed_scaling_factor = routed_scaling_factor

        # Needed for string -> FusedMoE layer lookup in custom ops.
        self.layer_name = layer_name

        self._forward_entry = self._select_forward()

    @property
    def is_internal_router(self) -> bool:
        return self.gate is not None

    @property
    def quant_method(self) -> FusedMoEMethodBase:
        return self.routed_experts.quant_method

    @property
    def shared_experts(self) -> SharedExperts | None:
        return self._shared_experts

    # TODO(bnell): Temporary hack. Get rid of this.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        self.routed_experts.quant_method = quant_method
        if self._shared_experts is not None:
            self._shared_experts._mk_owns_shared_expert = (
                quant_method.mk_owns_shared_expert
            )

    def _select_forward(self) -> Callable:
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped _forward_impl.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    def apply_routed_input_transform(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by FusedMoE.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

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
            hidden_states if self._shared_experts is not None else None,
        )

    def apply_routed_output_transform(
        self,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transform to routed expert output (e.g., latent to full dim).

        Used by latent MoE models (e.g., NemotronH) where routed experts
        operate in a compressed latent space and need projection back to
        the full hidden dimension before combining with shared expert output.
        """
        if self.routed_output_transform is not None:
            r = self.routed_output_transform(fused_output)
            fused_output = r[0] if isinstance(r, tuple) else r
        return fused_output

    def _maybe_apply_output_scale(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply routed_scaling_factor to the output with FP16 overflow
        protection.

        When apply_scale_to_output is True, scales the fused expert output
        by routed_scaling_factor. For FP16, avoids overflow by dividing
        shared_output by the scale instead (the decoder layer compensates
        with matching divisions).
        """
        if self.apply_scale_to_output:
            if fused_output.dtype != torch.float16:
                fused_output *= self.routed_scaling_factor
            elif shared_output is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
        return shared_output, fused_output

    def _must_reduce_shared_expert_output(self) -> bool:
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
            self._shared_experts is not None
            and self.quant_method.moe_kernel is not None
            and self.quant_method.moe_kernel.output_is_reduced()
        )

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """All-reduce shared expert output when the combine kernel already
        reduced fused output.

        This is the "early" all-reduce path. When the combine kernel produces
        already-reduced fused output, shared output must be reduced separately
        to match. See _must_reduce_shared_expert_output for details.
        """
        if self._must_reduce_shared_expert_output():
            assert shared_output is not None
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    def _maybe_reduce_output(
        self,
        states: torch.Tensor,
        trunc_size: int,
    ) -> torch.Tensor:
        """Truncate padded dimensions and all-reduce the combined output.

        This is the "late" all-reduce path. When neither fused nor shared
        output was individually reduced, the combined sum is all-reduced
        here. Skipped when sequence-parallel is active (SP handles its
        own reduction) or when the early path already reduced both outputs.
        """
        result = states[..., :trunc_size]

        if (
            not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not self._must_reduce_shared_expert_output()
        ):
            result = tensor_model_parallel_all_reduce(result)

        return result

    def _encode_layer_name(self) -> str | ModuleName:
        """Return the layer name string for custom op layer lookup.

        When torch.compile is active, returns "from_forward_context" so the
        custom op resolves the layer via ForwardContext at runtime (avoiding
        graph breaks). Falls back to the literal layer name for unit tests
        or when ForwardContext is unavailable.
        """
        if HAS_OPAQUE_TYPE:
            return ModuleName(self.layer_name)
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
    ) -> tuple[torch.Tensor, int]:
        """Pad hidden_states to moe_config.hidden_dim and compute the
        original dimension for later truncation.

        For latent MoE, the routed hidden_states may be smaller than
        hidden_dim. Padding ensures uniform tensor sizes through the
        fused MoE kernel. The returned trunc_size is used by
        _maybe_reduce_output to strip the padding from the result.
        """
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim = hidden_states.shape[-1]
        if (
            not self.quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        if self.routed_output_transform is not None and shared_experts_hidden_dim > 0:
            orig_hidden_dims = shared_experts_hidden_dim
        else:
            orig_hidden_dims = transformed_hidden_dim

        return hidden_states, orig_hidden_dims

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        """Trigger shared expert computation at the specified ordering point.

        Shared experts can run at different points relative to routed experts
        (EXTERNAL, BEFORE_QUANT_METHOD, AFTER_QUANT_METHOD) depending on the
        model's overlap strategy. Only fires if shared experts are configured
        and the order matches the shared experts' configured execution point.
        """
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts(shared_experts_input, order)

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).
        """
        # Run this before quant_method to avoid inplace issues.
        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.NO_OVERLAP,
        )

        if self.quant_method.is_monolithic:
            # Monolithic kernels: pass router_logits to routed_experts
            fused_out = self.routed_experts.forward(
                x=hidden_states,
                router_logits=router_logits,
            )
        else:
            # Modular kernels: select experts first, then call routed_experts
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            fused_out = self.routed_experts.forward(
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
        )

    def _sequence_parallel_context(self):
        """Return a context manager for sequence-parallel token
        redistribution.

        When sequence parallelism is active, returns a context that handles
        local size tracking for proper token scatter/gather. Otherwise
        returns a no-op context.
        """
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_overlap_gate_with_shared_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the gate module to compute router logits if provided.

        Used in overlapped mode where shared experts run in parallel with
        routed experts on a separate CUDA stream. The gate is separated
        from the router to allow this parallel execution.
        """
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.shared_experts is not None:
            self.shared_experts.maybe_setup_shared_experts_stream(shared_experts_input)

        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        return router_logits

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Add the zero expert's contribution to the final result.

        When a ZeroExpertRouter is used, it computes a bias-like output
        from the "zero expert" that is added to the combined routed+shared
        expert output.
        """
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.

        Calling sequence
        - forward
          - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
            - _forward_dispatch
              - _forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reasons:
        1. the chunking loop in ChunkingMoERunner._forward_impl cannot be compiled by
           torch.compile
        2. pytorch cannot handle union types in custom op signatures so
           _moe_forward and _moe_forward_shared must be split.

        If ChunkingMoERunner._forward_impl can be implemented via torch.scan we can
        potentially get rid of _moe_forward and _moe_forward_shared and collapse the
        whole sequence into the 'forward' method.
        """

        # Apply transform for routed experts (e.g., latent projection
        # for latent MoE)
        hidden_states, shared_experts_input = self.apply_routed_input_transform(
            hidden_states,
        )

        hidden_states, og_hidden_dim = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        router_logits = self._maybe_overlap_gate_with_shared_experts(
            hidden_states,
            router_logits,
            shared_experts_input,
        )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.EXTERNAL,
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            self._encode_layer_name(),
        )

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _must_reduce_shared_expert_output():
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_output.

        # Extract outputs from result
        shared_output, fused_output = _unpack(result)

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        # If combine kernel already reduced fused, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(shared_output)

        shared_output, fused_output = self._maybe_apply_output_scale(
            shared_output, fused_output
        )

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        result = self._maybe_reduce_output(result, og_hidden_dim)

        return self._maybe_add_zero_expert_output(result)

    def _forward_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Entry point called by the custom op to run the MoE computation.

        Handles pre-dispatch setup (gate application, external shared expert
        triggering, quant config init) then delegates to _forward_impl within
        the sequence-parallel context.
        """
        # TODO(bnell): this can be removed after MK migration is complete.
        self.routed_experts._ensure_moe_quant_config_init()

        with self._sequence_parallel_context():
            return self._forward_impl(
                hidden_states,
                router_logits,
                shared_experts_input,
            )

    @abstractmethod
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Core MoE computation to be implemented by subclasses.

        Performs expert routing, fused MoE kernel execution, and shared
        expert computation. Returns a single tensor (fused output only)
        or a tuple of (shared_output, fused_output) when shared experts
        are present.
        """
        raise NotImplementedError
