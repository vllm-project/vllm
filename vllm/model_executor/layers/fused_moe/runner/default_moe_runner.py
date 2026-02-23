# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
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
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

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


class DefaultMoERunner(MoERunner):
    """
    Default implementation of the MoE runner for executing Mixture of Experts layers.

    This class provides a comprehensive implementation for running MoE computations
    with support for:
    - Expert routing and token dispatching
    - Shared experts computation with optional parallel execution using CUDA streams
    - Data parallel (DP) chunking for large batch processing
    - Tensor model parallel and expert parallel operations
    - Various quantization methods and custom operators
    - Both monolithic and decomposed expert execution paths

    The runner handles the complete MoE forward pass including routing tokens to
    experts, executing expert computations, and combining results. It supports
    advanced features like overlapped execution of shared experts and optimized
    kernels for different parallel execution modes.

    Eventually, this class will be split up and specialized for different
    configurations, e.g. the presence or absence of shared experts, a gate, etc.
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
        self.reduce_results = reduce_results
        self.enable_dbo = enable_dbo
        self.enable_eplb = moe_config.moe_parallel_config.enable_eplb

        # Chunked all2all staging tensor
        # TODO rename these?
        # These need to exist ahead of time due to CUDAgraph construction
        # needing a fixed buffer address.
        # TODO: these could be global, i.e. shared by all layers
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None
        self._maybe_init_dp_chunking()

        self.use_dp_chunking = self.moe_config.moe_parallel_config.use_dp_chunking

        # Needed for string -> FusedMoE layer lookup in custom ops.
        self.layer_name = layer.layer_name

        self.moe_forward, self.moe_forward_impl = self._select_forward(layer)

    def _select_forward(self, layer: torch.nn.Module) -> tuple[Callable, Callable]:
        forward_impl_fn = (
            self.forward_impl_chunked if self.use_dp_chunking else self.forward_impl
        )
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped forward_impl.
            return (
                _moe_forward if self.shared_experts is None else _moe_forward_shared,
                forward_impl_fn,
            )

        return (
            torch.ops.vllm.moe_forward
            if self.shared_experts is None
            else torch.ops.vllm.moe_forward_shared,
            forward_impl_fn,
        )

    def _maybe_init_dp_chunking(self):
        if not self.use_dp_chunking:
            return

        assert self.batched_hidden_states is None
        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (2, moe.max_num_tokens, self.moe_config.num_logical_experts)
        else:
            states_shape = (moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (moe.max_num_tokens, self.moe_config.num_logical_experts)

        self.batched_hidden_states = torch.zeros(
            states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

        self.batched_router_logits = torch.zeros(
            logits_shape,
            dtype=moe.router_logits_dtype,
            device=torch.cuda.current_device(),
        )

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
            and not self.use_dp_chunking
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

    def _allocate_dp_chunking_outputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        assert self.use_dp_chunking

        # Assert the inputs are of the proper type and shape.
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None

        assert self.batched_hidden_states.dtype == hidden_states.dtype, (
            f"{self.batched_hidden_states.dtype} == {hidden_states.dtype}"
        )
        assert self.batched_router_logits.dtype == router_logits.dtype, (
            f"{self.batched_router_logits.dtype} == {router_logits.dtype}"
        )

        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == router_logits.size(-1)

        final_fused_hidden_states = torch.empty_like(hidden_states)
        if self.shared_experts is not None:
            final_shared_hidden_states = torch.empty_like(hidden_states)
        else:
            final_shared_hidden_states = None

        return final_shared_hidden_states, final_fused_hidden_states

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

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
        )

    def _maybe_dispatch(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        extra_tensor: torch.Tensor | None = None

        if self.do_naive_dispatch_combine:
            post_quant_allgather = (
                self.moe_config.dp_size > 1
                and self.moe_config.use_ep
                and getattr(self.quant_method, "do_post_quant_allgather", False)
            )

            extra_tensors: list[torch.Tensor] | None = None

            if post_quant_allgather:
                hidden_states_to_dispatch, extra_tensors = (
                    self.quant_method.prepare_dp_allgather_tensor(
                        layer, hidden_states, router_logits
                    )
                )
            else:
                hidden_states_to_dispatch = hidden_states

            result = get_ep_group().dispatch_router_logits(
                hidden_states_to_dispatch,
                router_logits,
                self.moe_config.is_sequence_parallel,
                extra_tensors=extra_tensors,
            )

            if len(result) == 3:
                hidden_states, router_logits, extra_tensors = result
                assert isinstance(extra_tensors, list) and len(extra_tensors) == 1
                extra_tensor = extra_tensors[0]
            else:
                hidden_states, router_logits = result

        # NOTE: Similar with DP, PCP also needs dispatch and combine. For
        # simplicity, AgRsAll2All was added separately for PCP here. Maybe
        # we should modify All2AllManager abstraction to better support PCP.
        # TODO(bnell): see what we can do here
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return hidden_states, router_logits, extra_tensor

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if self.do_naive_dispatch_combine:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(
                hidden_states,
                dim=0,
            )
            # need RS for shared_output?

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

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
          - self.moe_forward (_moe_forward or _moe_forward_shared custom op)
            - forward_dispatch
              - moe_forward_impl (forward_impl or forward_impl_chunked)

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reasons:
        1. the chunking loop in forward_impl_chunked cannot be compiled by torch.compile
        2. pytorch cannot handle union types in custom op signatures so _moe_forward and
           _moe_forward_shared must be split.

        If forward_impl_chunked can be implemented via torch.scan we can potentially get
        rid of _moe_forward and _moe_forward_shared and collapse the whole sequence into
        the 'forward' method.
        """

        # Apply transform for routed experts (e.g., latent projection for latent MoE)
        hidden_states, shared_experts_input = self.apply_routed_input_transform(
            hidden_states
        )

        hidden_states, og_hidden_dims = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )

        fused_output = self.moe_forward(
            hidden_states,
            router_logits,
            shared_experts_input,
            self._encode_layer_name(),
        )

        return self._maybe_reduce_output(fused_output, og_hidden_dims)

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
            return self.moe_forward_impl(
                layer,
                hidden_states,
                router_logits,
                shared_experts_input,
            )

    def _slice_and_copy_input(
        self,
        out_slice: torch.Tensor,
        orig: torch.Tensor | None,
        start: int,
        end: int,
    ) -> torch.Tensor:
        assert orig is not None
        slice_size = end - start
        orig_slice = orig[start:end, :]
        if self.enable_dbo:
            assert out_slice.dim() == 3
            batch_buffer_idx = dbo_current_ubatch_id()
            out_slice = out_slice[batch_buffer_idx, :]

        assert out_slice.size(0) >= slice_size
        out_slice = out_slice[:slice_size, :]
        out_slice.copy_(orig_slice, non_blocking=True)
        return out_slice

    def forward_impl_chunked(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        final_shared_hidden_states, final_fused_hidden_states = (
            self._allocate_dp_chunking_outputs(hidden_states, router_logits)
        )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.moe_config.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.moe_config.sp_size
            )

        num_tokens = hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            chunk_sizes = ctx.dp_metadata.chunked_sizes(
                self.moe_config.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            )
            with chunk_sizes:
                hidden_states_chunk = self._slice_and_copy_input(
                    self.batched_hidden_states,
                    hidden_states,
                    chunk_start,
                    chunk_end,
                )

                router_logits_chunk = self._slice_and_copy_input(
                    self.batched_router_logits,
                    router_logits,
                    chunk_start,
                    chunk_end,
                )

                shared_experts_input_chunk = (
                    shared_experts_input[chunk_start:chunk_end, :]
                    if shared_experts_input is not None
                    else None
                )

                shared_output_chunk, hidden_states_chunk = self._apply_quant_method(
                    layer=layer,
                    hidden_states=hidden_states_chunk,
                    extra_tensor=None,
                    router_logits=router_logits_chunk,
                    shared_experts_input=shared_experts_input_chunk,
                )

                # Store outputs
                # TODO(bnell): document when chunk_start >= num_tokens
                if chunk_start < num_tokens:
                    final_fused_hidden_states[chunk_start:chunk_end, :].copy_(
                        hidden_states_chunk, non_blocking=True
                    )
                    if self.shared_experts is not None:
                        assert shared_output_chunk is not None
                        assert final_shared_hidden_states is not None
                        final_shared_hidden_states[chunk_start:chunk_end, :].copy_(
                            shared_output_chunk, non_blocking=True
                        )

        if self.shared_experts is None:
            return final_fused_hidden_states
        else:
            assert final_shared_hidden_states is not None
            return (final_shared_hidden_states, final_fused_hidden_states)

    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # TODO(bnell): parts of the dispatch/combine steps will go away once
        # #32567 lands and the remaining kernels are made MKs.  The PCP
        # code will probably remain
        hidden_states, router_logits, extra_tensor = self._maybe_dispatch(
            layer,
            hidden_states,
            router_logits,
        )

        shared_output, hidden_states = self._apply_quant_method(
            layer=layer,
            hidden_states=hidden_states,
            extra_tensor=extra_tensor,
            router_logits=router_logits,
            shared_experts_input=shared_experts_input,
        )

        return self._maybe_combine(
            shared_output,
            hidden_states,
        )
