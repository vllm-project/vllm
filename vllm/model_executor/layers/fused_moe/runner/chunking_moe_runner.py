# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.forward_context import (
    get_forward_context,
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
from vllm.model_executor.layers.fused_moe.runner.moe_runner_base import MoERunnerBase
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.ubatching import dbo_current_ubatch_id
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


class ChunkingMoERunner(MoERunnerBase):
    """
    Specialized MoE runner that processes large batches by breaking them into smaller
    chunks.

    This runner is designed for scenarios where the input batch is too large to process
    in a single pass, typically due to memory constraints or when using data parallel
    (DP) chunking strategies. It provides:
    - Automatic chunking of large input batches into manageable sizes
    - Memory-efficient processing by reusing pre-allocated workspace tensors
    - Support for both hidden states and router logits chunking
    - Slice-and-copy operations to handle input/output tensor management
    - Integration with workspace managers for optimal memory utilization

    The chunking strategy allows processing of arbitrarily large batches by dividing
    the computation across multiple smaller chunks, then combining the results.
    This approach is particularly beneficial in distributed settings where memory
    per rank is limited or when the batch size exceeds hardware capabilities.

    Key differences from DefaultMoERunner:
    - Uses pre-allocated workspace tensors for intermediate computations
    - Implements chunked processing logic in forward_impl
    - Never reduces results (reduce_results always returns False)
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
        super().__init__(
            layer,
            moe_config,
            router,
            routed_input_transform,
            gate,
            shared_experts,
            quant_method,
            reduce_results,
            enable_dbo,
        )

        # Chunked all2all staging tensor
        # TODO rename these?
        # These need to exist ahead of time due to CUDAgraph construction
        # needing a fixed buffer address.
        self.batched_hidden_states, self.batched_router_logits = (
            self._init_dp_chunking()
        )

    @property
    def reduce_results(self) -> bool:
        return False

    def _init_dp_chunking(self) -> list[torch.Tensor]:
        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (2, moe.max_num_tokens, self.moe_config.num_logical_experts)
        else:
            states_shape = (moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (moe.max_num_tokens, self.moe_config.num_logical_experts)

        # Does this need some kind of profiling run check like modular_kernel.py?
        return current_workspace_manager().get_simultaneous(
            (states_shape, moe.in_dtype),
            (logits_shape, moe.router_logits_dtype),
        )

    def _allocate_dp_chunking_outputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
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

    def forward_impl(
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
