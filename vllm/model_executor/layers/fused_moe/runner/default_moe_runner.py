# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


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
    configurations, e.g. the presense or absence of shared experts, a gate, etc.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        moe_quant_config: FusedMoEQuantConfig | None,
        router: FusedMoERouter,
        routed_input_transform: torch.nn.Module | None,
        gate: torch.nn.Module | None,
        shared_experts: torch.nn.Module | None,
        quant_method: FusedMoEMethodBase,
        reduce_results: bool,
        enable_dbo: bool,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.moe_quant_config = moe_quant_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.gate = gate
        self.shared_experts = shared_experts
        self.quant_method = quant_method
        self.reduce_results = reduce_results
        self.enable_dbo = enable_dbo

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            self.shared_experts_stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

        lname = layer.layer_name.replace(".", "_")

        def _moe_forward(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            return self.forward_impl(
                layer, hidden_states, router_logits, shared_experts_input
            )

        def _moe_forward_shared(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.forward_impl(
                layer, hidden_states, router_logits, shared_experts_input
            )

        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped forward_impl.
            if self.shared_experts is None:
                self.moe_forward = _moe_forward
            else:
                self.moe_forward = _moe_forward_shared
        else:
            if self.shared_experts is None:
                op_name = f"moe_forward{lname}"
                if not hasattr(torch.ops.vllm, op_name):
                    direct_register_custom_op(
                        op_name=op_name,
                        op_func=_moe_forward,
                        mutates_args=["hidden_states"],
                        fake_impl=DefaultMoERunner._moe_forward_fake,
                        tags=(torch.Tag.needs_fixed_stride_order,),
                    )
                self.moe_forward = getattr(torch.ops.vllm, op_name)
            else:
                op_name = f"moe_forward_shared{lname}"
                if not hasattr(torch.ops.vllm, op_name):
                    direct_register_custom_op(
                        op_name=op_name,
                        op_func=_moe_forward_shared,
                        mutates_args=["hidden_states"],
                        fake_impl=DefaultMoERunner._moe_forward_shared_fake,
                        tags=(torch.Tag.needs_fixed_stride_order,),
                    )
                self.moe_forward = getattr(torch.ops.vllm, op_name)

        # Chunked all2all staging tensor
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

    @staticmethod
    def _moe_forward_fake(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    @staticmethod
    def _moe_forward_shared_fake(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
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

    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_config.moe_parallel_config.use_pplx_kernels
            or self.moe_config.moe_parallel_config.use_deepep_ll_kernels
            or self.moe_config.moe_parallel_config.use_mori_kernels
            or self.moe_config.moe_parallel_config.use_fi_all2allv_kernels
        ) and envs.VLLM_ENABLE_MOE_DP_CHUNK

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        shared_input: torch.Tensor | None,
        has_separate_shared_experts: bool,
        use_chunked_impl: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        use_shared_experts_stream = (
            current_platform.is_cuda()
            and has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        hidden_states_clone: torch.Tensor | None = None
        if use_shared_experts_stream:
            assert self.shared_experts_stream is not None

            shared_experts_input = (
                shared_input if shared_input is not None else hidden_states
            )

            # Clone BEFORE switching streams to avoid race condition
            # where routed_expert kernel may mutate hidden_states.
            hidden_states_clone = shared_experts_input.clone()

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            hidden_states_clone.record_stream(self.shared_experts_stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, hidden_states_clone

    def ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

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
        assert self.quant_method is not None
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

    def _reduce_output(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        trunc_sizes: list[int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        def trunc(idx: int, x: torch.Tensor) -> torch.Tensor:
            return x[..., : trunc_sizes[idx]]

        def reduce_and_trunc(idx: int, x: torch.Tensor) -> torch.Tensor:
            return trunc(idx, self.maybe_all_reduce_tensor_model_parallel(x))

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
            assert len(trunc_sizes) == len(states)
            return tuple([func(i, s) for i, s in enumerate(states)])
        else:
            assert len(trunc_sizes) == 1
            return func(0, states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # For latent MoE: save ORIGINAL hidden_states before transform
        # (shared_experts need original dimension, routed experts use transformed)
        original_hidden_states = hidden_states
        original_hidden_dim = hidden_states.shape[-1]

        # Apply transform for routed experts (e.g., latent projection for latent MoE)
        if self.routed_input_transform is not None:
            transformed_states = self.routed_input_transform(hidden_states)
            if isinstance(transformed_states, tuple):
                hidden_states = transformed_states[0]
            else:
                hidden_states = transformed_states

        # This is the dimension after transform (for routed expert output slicing)
        transformed_hidden_dim = hidden_states.shape[-1]
        if self.moe_config.hidden_dim != transformed_hidden_dim:
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        fused_output = self.moe_forward(
            hidden_states, router_logits, original_hidden_states
        )

        if isinstance(fused_output, tuple):
            orig_hidden_dims = [original_hidden_dim, transformed_hidden_dim]
        else:
            orig_hidden_dims = [transformed_hidden_dim]

        return self._reduce_output(fused_output, orig_hidden_dims)

    def forward_impl_chunked(
        self,
        layer: torch.nn.Module,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype, (
            f"{self.batched_hidden_states.dtype} == {full_hidden_states.dtype}"
        )
        assert self.batched_router_logits.dtype == full_router_logits.dtype, (
            f"{self.batched_router_logits.dtype} == {full_router_logits.dtype}"
        )
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # Matrix multiply.
            if self.quant_method.is_monolithic:
                final_hidden_states = self.quant_method.apply_monolithic(
                    layer=layer,
                    x=staged_hidden_states,
                    router_logits=staged_router_logits,
                )
            else:
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=staged_hidden_states,
                    router_logits=staged_router_logits,
                )

                final_hidden_states = self.quant_method.apply(
                    layer=layer,
                    x=staged_hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                shared_output = self.shared_experts(staged_hidden_states)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
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

        num_tokens = full_hidden_states.size(0)
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
            with ctx.dp_metadata.chunked_sizes(
                self.moe_config.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.quant_method is not None

        self.ensure_dp_chunking_init()

        has_separate_shared_experts = (
            not self.quant_method.mk_owns_shared_expert
            and self.shared_experts is not None
        )

        use_chunked_impl = self.use_dp_chunking

        use_shared_experts_stream, hidden_states_clone = (
            self._maybe_setup_shared_experts_stream(
                hidden_states,
                shared_input,
                has_separate_shared_experts,
                use_chunked_impl,
            )
        )

        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        if use_chunked_impl:
            return self.forward_impl_chunked(
                layer,
                hidden_states,
                router_logits,
                shared_input,
                has_separate_shared_experts,
            )

        # NOTE(rob): once we finish migrating all the quant methods to use
        # MKs, we can remove the naive dispatch/combine path from here.
        do_naive_dispatch_combine = (
            self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
        )

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            extra_tensors = None
            if do_naive_dispatch_combine:
                post_quant_allgather = (
                    self.quant_method is not None
                    and self.moe_config.dp_size > 1
                    and self.moe_config.use_ep
                    and getattr(self.quant_method, "do_post_quant_allgather", False)
                )
                if post_quant_allgather:
                    hidden_states_to_dispatch, extra_tensors = (
                        self.quant_method.prepare_dp_allgather_tensor(
                            self, hidden_states, router_logits
                        )
                    )
                else:
                    hidden_states_to_dispatch = hidden_states

                dispatch_res = get_ep_group().dispatch_router_logits(
                    hidden_states_to_dispatch,
                    router_logits,
                    self.moe_config.is_sequence_parallel,
                    extra_tensors=extra_tensors,
                )
                if extra_tensors is not None:
                    (
                        orig_hidden_states,
                        router_logits,
                        extra_tensors_combined,
                    ) = dispatch_res
                    hidden_states_combined = (
                        orig_hidden_states,
                        extra_tensors_combined[0],
                    )
                else:
                    hidden_states_combined, router_logits = dispatch_res
                    orig_hidden_states = hidden_states_combined
            else:
                orig_hidden_states = hidden_states

            # Run shared experts before matrix multiply.
            # because matrix multiply maybe modify the hidden_states.
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                shared_output = self.shared_experts(shared_input)

            # NOTE: Similar with DP, PCP also needs dispatch and combine. For
            # simplicity, AgRsAll2All was added separately for PCP here. Maybe
            # we should modify All2AllManager abstract to better support PCP.
            if self.moe_config.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # Matrix multiply.
            x = hidden_states_combined if do_naive_dispatch_combine else hidden_states

            # TODO(bnell): deal with fp4 flashinfer tuple hidden states hack (#30014).
            # Figure out nicer way to do this.
            x_orig = orig_hidden_states if do_naive_dispatch_combine else hidden_states

            if self.quant_method.is_monolithic:
                final_hidden_states = self.quant_method.apply_monolithic(
                    layer=layer,
                    x=x,
                    router_logits=router_logits,
                )
            else:
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=x_orig,
                    router_logits=router_logits,
                )

                final_hidden_states = self.quant_method.apply(
                    layer=layer,
                    x=x,  # The type signture of this is wrong due to the hack.
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                )

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(hidden_states_clone)
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(
                        states, self.moe_config.is_sequence_parallel
                    )

                if self.moe_config.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            if self.shared_experts is not None:
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            else:
                return combine_output(final_hidden_states)
