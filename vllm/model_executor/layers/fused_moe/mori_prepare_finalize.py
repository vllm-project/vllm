# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
mori prepare and finalize module for expert parallelism.
Migration from DeepEP to mori for AMD GPU support.
"""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

logger = init_logger(__name__)


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using mori kernels for AMD GPU expert parallelism.

    This class handles the dispatch and combine operations for
    expert parallelism using the mori library, which provides optimized
    All2All communication primitives for AMD GPUs.
    """

    def __init__(
        self,
        handle: Any,  # mori EpDispatchCombineOp from MoriAll2AllManager
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        json_config: dict | None = None,
    ):
        """
        Initialize MoriPrepareAndFinalize.

        Args:
            handle: mori EpDispatchCombineOp instance from All2AllManager
            max_num_tokens: Maximum number of tokens per rank
            num_local_experts: Number of experts on this rank
            num_dispatchers: Number of dispatcher ranks (world size)
            use_fp8_dispatch: Whether to use FP8 quantization during dispatch
            json_config: Optional JSON configuration with operation-specific parameters
        """
        super().__init__()
        assert max_num_tokens > 0
        assert num_local_experts > 0

        self.handle = handle  # mori EpDispatchCombineOp
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers
        self.use_fp8_dispatch = use_fp8_dispatch

        # Extract dispatch and combine specific parameters from JSON config
        self.dispatch_kwargs = {}
        self.combine_kwargs = {}

        if json_config:
            # Extract dispatch-specific parameters
            if "dispatch" in json_config:
                dispatch_config = json_config["dispatch"]

                if "block_num" in dispatch_config:
                    self.dispatch_kwargs["block_num"] = dispatch_config["block_num"]
                if "warp_num_per_block" in dispatch_config:
                    self.dispatch_kwargs["warp_per_block"] = dispatch_config[
                        "warp_num_per_block"
                    ]

            # Extract combine-specific parameters
            if "combine" in json_config:
                combine_config = json_config["combine"]

                if "block_num" in combine_config:
                    self.combine_kwargs["block_num"] = combine_config["block_num"]
                if "warp_num_per_block" in combine_config:
                    self.combine_kwargs["warp_per_block"] = combine_config[
                        "warp_num_per_block"
                    ]

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        """
        Prepare inputs for mori dispatch operation.
        Supports pre-dispatch quantization to reduce communication overhead.

        Args:
            a1: Input hidden states [num_tokens, hidden_dim]
            topk_weights: Top-k routing weights [num_experts, experts_per_token]
            topk_ids: Top-k expert indices [num_experts, experts_per_token]
            apply_router_weight_on_input: Whether to apply router weight
            quant_config: Quantization config

        Returns:
            Tuple of (dispatched_x, batched_scales, expert_tokens_meta,
                      dispatch_indices, dispatch_weights)
            where dispatched_x is in Standard format (2D tensor)
        """
        # Pre-dispatch quantization to reduce communication overhead
        dispatch_input = a1
        scales = None

        if self.use_fp8_dispatch:
            from aiter import QuantType, get_hip_quant

            if quant_config.block_shape is not None:
                assert not apply_router_weight_on_input, (
                    "apply_router_weight_on_input is not supported for block scaled moe"
                )
                quant_type = QuantType.per_1x128
            else:
                quant_type = QuantType.per_Tensor

            quant_func = get_hip_quant(quant_type)

            dispatch_input, scales = quant_func(
                a1,
                quant_dtype=quant_config.quant_dtype,
            )

        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = self.handle.dispatch(
            input=dispatch_input,
            weights=topk_weights,
            scales=scales,
            indices=topk_ids,
            **self.dispatch_kwargs,  # Apply dispatch-specific parameters from JSON
        )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_num_token,
            expert_num_tokens_cpu=None,
        )

        return (
            dispatch_output,
            dispatch_scales,
            expert_tokens_meta,
            dispatch_indices,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        extra_finalize_args: dict | None = None,
    ) -> None:
        """
        Finalize expert outputs using mori combine operation.

        Args:
            output: Output tensor to write results [num_original_tokens,
                                                    hidden_dim]
            fused_expert_output: Expert output activations in Standard format
                                 (2D tensor)
            topk_weights: Original top-k weights
            topk_ids: Original top-k indices
        """
        assert self.handle is not None

        num_original_tokens = output.size(0)  # Original number of tokens

        combined_output, combined_weights = self.handle.combine(
            input=fused_expert_output,
            weights=topk_weights,
            indices=topk_ids,
            **self.combine_kwargs,  # Apply combine-specific parameters from JSON
        )

        output.copy_(
            combined_output[:num_original_tokens],
            non_blocking=True,
        )

    def __repr__(self) -> str:
        return (
            f"MoriPrepareAndFinalize(max_tokens={self.max_num_tokens}, "
            f"num_local_experts={self.num_local_experts}, "
            f"num_dispatchers={self.num_dispatchers_})"
        )
