# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import dataclass
from math import prod

import torch

from vllm import _custom_ops as ops
from vllm.lora.punica_wrapper import PunicaWrapperBase
from vllm.model_executor.layers.fused_moe.config import (
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.mk_fused_experts_lora_support import (
    MkFusedExpertsSupportsLoRA,
    mk_fused_experts_supports_lora,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata,
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    TopKWeightAndReduce,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache


@dataclass
class ExpertsForwardState:
    base_output: torch.Tensor
    base_w13: torch.Tensor
    base_w2: torch.Tensor
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    num_experts: int
    expert_map: torch.Tensor | None
    apply_router_weight_on_input: bool
    # LoRA output buffer
    lora_gateup_output: torch.Tensor
    lora_down_output: torch.Tensor
    reduction_output: torch.Tensor
    # MoE lora align block size outputs
    sorted_token_ids_lora: torch.Tensor
    expert_ids_lora: torch.Tensor
    num_tokens_post_padded_lora: torch.Tensor
    # LoRA kernel configs
    config: dict[str, int]


class FusedMoEPermuteExpertsUnpermuteWithLoRA(
    FusedMoEPermuteExpertsUnpermute, MkFusedExpertsSupportsLoRA
):
    def __init__(self, base_experts: FusedMoEPermuteExpertsUnpermute):
        assert isinstance(base_experts, FusedMoEPermuteExpertsUnpermute)
        assert mk_fused_experts_supports_lora(base_experts)
        super().__init__(base_experts.quant_config)
        self.base_experts = base_experts
        self.punica_wrapper: PunicaWrapperBase | None = None
        self.w1_lora_a_stacked: torch.Tensor | None = None
        self.w1_lora_b_stacked: torch.Tensor | None = None
        self.w3_lora_a_stacked: torch.Tensor | None = None
        self.w3_lora_b_stacked: torch.Tensor | None = None
        self.w2_lora_a_stacked: torch.Tensor | None = None
        self.w2_lora_b_stacked: torch.Tensor | None = None
        self._inject_matmul_prologue_epilogue()

        # state recording
        self.experts_forward_state: ExpertsForwardState | None = None

    def _inject_matmul_prologue_epilogue(self):
        def activation_prologue(base_act_prologue_fn):
            def wrapper(*args, **kwargs):
                # Assert so we can be sure what args we are reading
                assert len(args) == 0  # self
                assert len(kwargs) == 1
                base_gateup_proj_output = kwargs["gateup_proj_output"]
                self.gateup_proj_lora_add(base_output=base_gateup_proj_output)

            return wrapper

        def activation_epilogue(base_act_epilogue_fn):
            def wrapper(*args, **kwargs):
                # Assert so we can be sure what args we are reading
                assert len(args) == 0  # self
                assert len(kwargs) == 1
                base_act_output = kwargs["activation_output"]
                self.down_proj_lora(input=base_act_output)

            return wrapper

        assert isinstance(self.base_experts, MkFusedExpertsSupportsLoRA)
        self.base_experts.activation_prologue = activation_prologue(
            self.base_experts.activation_prologue
        )
        self.base_experts.activation_epilogue = activation_epilogue(
            self.base_experts.activation_epilogue
        )

    def gateup_proj_lora(self):
        self._ensure_weights()

        assert self.experts_forward_state is not None
        assert self.w1_lora_a_stacked is not None
        hidden_states = self.experts_forward_state.hidden_states
        topk_ids = self.experts_forward_state.topk_ids
        topk_weights = self.experts_forward_state.topk_weights

        num_topk = topk_ids.size(-1)
        max_lora_rank = self.w1_lora_a_stacked.size(-2)

        w13_lora_a_stacked = [self.w1_lora_a_stacked, self.w3_lora_a_stacked]
        w13_lora_b_stacked = [self.w1_lora_b_stacked, self.w3_lora_b_stacked]

        # TODO (varun): Fix add_lora_fused_moe to overwrite output
        self.experts_forward_state.lora_gateup_output.fill_(0)
        assert self.punica_wrapper is not None
        self.punica_wrapper.add_lora_fused_moe(
            self.experts_forward_state.lora_gateup_output,
            hidden_states,
            w13_lora_a_stacked,
            w13_lora_b_stacked,
            topk_weights,
            self.experts_forward_state.sorted_token_ids_lora,
            self.experts_forward_state.expert_ids_lora,
            self.experts_forward_state.num_tokens_post_padded_lora,
            max_lora_rank,
            num_topk,
            self.experts_forward_state.config,
        )

    def gateup_proj_lora_add(self, base_output: torch.Tensor):
        assert self.experts_forward_state is not None
        lora_output = self.experts_forward_state.lora_gateup_output
        assert (
            base_output.dtype == lora_output.dtype
            and base_output.size() == lora_output.size()
        ), (
            "lora output does match base output: "
            f"lora output {lora_output.size()} {lora_output.dtype} vs"
            f"base_output {base_output.size()} {base_output.dtype}"
        )
        base_output += self.experts_forward_state.lora_gateup_output

    def down_proj_lora(self, input: torch.Tensor):
        self._ensure_weights()
        assert self.experts_forward_state is not None
        assert self.w1_lora_a_stacked is not None

        topk_weights = self.experts_forward_state.topk_weights
        topk_ids = self.experts_forward_state.topk_ids

        num_topk = topk_ids.size(-1)
        max_lora_rank = self.w1_lora_a_stacked.size(-2)

        # TODO (varun): Fix add_lora_fused_moe to overwrite output
        self.experts_forward_state.lora_down_output.fill_(0)
        assert self.punica_wrapper is not None
        self.punica_wrapper.add_lora_fused_moe(
            self.experts_forward_state.lora_down_output,
            input,
            [self.w2_lora_a_stacked],
            [self.w2_lora_b_stacked],
            topk_weights,
            self.experts_forward_state.sorted_token_ids_lora,
            self.experts_forward_state.expert_ids_lora,
            self.experts_forward_state.num_tokens_post_padded_lora,
            max_lora_rank,
            num_topk,
            self.experts_forward_state.config,
            True,
        )

        ops.moe_sum(
            self.experts_forward_state.lora_down_output,
            self.experts_forward_state.reduction_output,
        )

    def down_proj_lora_add(self):
        assert self.experts_forward_state is not None
        base_output = self.experts_forward_state.base_output
        lora_output = self.experts_forward_state.reduction_output
        assert (
            base_output.dtype == lora_output.dtype
            and base_output.size() == lora_output.size()
        ), (
            "lora output does match base output: "
            f"lora output {lora_output.size()} {lora_output.dtype} vs"
            f"base_output {base_output.size()} {base_output.dtype}"
        )
        base_output += lora_output

    def set_lora_weights(
        self,
        w1_lora_a_stacked: torch.Tensor,
        w1_lora_b_stacked: torch.Tensor,
        w3_lora_a_stacked: torch.Tensor,
        w3_lora_b_stacked: torch.Tensor,
        w2_lora_a_stacked: torch.Tensor,
        w2_lora_b_stacked: torch.Tensor,
    ):
        self.w1_lora_a_stacked = w1_lora_a_stacked
        self.w1_lora_b_stacked = w1_lora_b_stacked
        self.w3_lora_a_stacked = w3_lora_a_stacked
        self.w3_lora_b_stacked = w3_lora_b_stacked
        self.w2_lora_a_stacked = w2_lora_a_stacked
        self.w2_lora_b_stacked = w2_lora_b_stacked

    def _ensure_weights(self):
        assert all(
            x is not None
            for x in [
                self.w1_lora_a_stacked,
                self.w1_lora_b_stacked,
                self.w3_lora_a_stacked,
                self.w3_lora_b_stacked,
                self.w2_lora_a_stacked,
                self.w2_lora_b_stacked,
            ]
        )

    def _allocate_lora_output_buffers(
        self, M: int, num_topk: int, device: str, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.w1_lora_b_stacked is not None
        assert self.w3_lora_b_stacked is not None
        assert self.w2_lora_b_stacked is not None
        gateup_proj_hidden = self.w1_lora_b_stacked.size(
            -2
        ) + self.w3_lora_b_stacked.size(-2)
        down_proj_hidden = self.w2_lora_b_stacked.size(-2)
        gateup_output_shape = (M, num_topk, gateup_proj_hidden)
        down_output_shape = (M, num_topk, down_proj_hidden)
        reduction_output_shape = (M, down_proj_hidden)

        # LoRA op output buffer can be reused between gateup and down outputs.
        lora_op_output_buffer = torch.empty(
            (max(prod(gateup_output_shape), prod(down_output_shape))),
            device=device,
            dtype=dtype,
        )
        gateup_output = _resize_cache(lora_op_output_buffer, gateup_output_shape)
        down_output = _resize_cache(lora_op_output_buffer, down_output_shape)

        # Reduction needs a separate buffer
        reduction_output = torch.empty(
            (reduction_output_shape), device=device, dtype=dtype
        )

        return gateup_output, down_output, reduction_output

    def _lora_moe_config(
        self,
        hidden_states: torch.Tensor,
        num_topk: int,
        base_w1: torch.Tensor,
        base_w2: torch.Tensor,
        quant_block_shape: list[int] | None = None,
    ) -> dict[str, int]:
        dtype = hidden_states.dtype
        num_tokens = hidden_states.size(0)

        config_dtype = _get_config_dtype_str(
            dtype=dtype,
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
        )

        return try_get_optimal_moe_config(
            w1_shape=base_w1.size(),
            w2_shape=base_w2.size(),
            top_k=num_topk,
            dtype=config_dtype,
            M=num_tokens,
            block_shape=quant_block_shape,
        )

    @contextmanager
    def _lora_apply(
        self,
        base_output: torch.Tensor,
        base_w13: torch.Tensor,
        base_w2: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_lora_a_stacked is not None
        M = hidden_states.size(0)
        num_topk = topk_ids.size(1)
        max_loras = self.w1_lora_a_stacked.size(0)

        # MoE kernel config
        lora_moe_kernel_config = self._lora_moe_config(
            hidden_states,
            num_topk,
            base_w13,
            base_w2,
            self.base_experts.quant_config.block_shape,
        )

        # MoE LoRA align block size.
        assert self.punica_wrapper is not None
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
        ) = self.punica_wrapper.moe_lora_align_block_size(
            topk_ids,
            M,
            lora_moe_kernel_config["BLOCK_SIZE_M"],
            num_experts,
            max_loras,
            expert_map,
            lora_token_mapping_offset=self.get_lora_token_mapping_offset(),
        )
        expert_ids_lora = expert_ids_lora.view(max_loras, -1)
        sorted_token_ids_lora = sorted_token_ids_lora.view(max_loras, -1)

        # Allocate lora output buffers.
        gateup_output, down_output, reduction_output = (
            self._allocate_lora_output_buffers(
                M, num_topk, hidden_states.device, hidden_states.dtype
            )
        )

        # Populate layer_forward_state_dict
        self.experts_forward_state = ExpertsForwardState(
            base_output=base_output,
            base_w13=base_w13,
            base_w2=base_w2,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            # LoRA output buffer.
            lora_gateup_output=gateup_output,
            lora_down_output=down_output,
            reduction_output=reduction_output,
            # MoE lora align block size outputs.
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            # LoRA kernel configs.
            config=lora_moe_kernel_config,
        )

        # Trigger LoRA gateup
        self.gateup_proj_lora()

        yield

        # Trigger final LoRA add
        self.down_proj_lora_add()

        # Reset state
        self.experts_forward_state = None

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        with self._lora_apply(
            base_output=output,
            base_w13=w1,
            base_w2=w2,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        ):
            self.base_experts.apply(
                output=output,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=a1q_scale,
                a2_scale=a2_scale,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper = punica_wrapper

    ## FusedMoEPermuteExpertsUnPermute Forwarders ##
    @property
    def activation_formats(
        self,
    ) -> tuple[FusedMoEActivationFormat, FusedMoEActivationFormat]:
        return self.base_experts.activation_formats

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        return self.base_experts.moe_problem_size(a1, w1, w2, topk_ids)

    def supports_chunking(self) -> bool:
        return self.base_experts.supports_chunking()

    def supports_expert_map(self) -> bool:
        return self.base_experts.supports_expert_map()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return self.base_experts.workspace_dtype(act_dtype)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self.base_experts.workspace_shapes(
            M, N, K, topk, global_num_experts, local_num_experts, expert_tokens_meta
        )

    def activation(
        self,
        activation: str,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> None:
        return self.base_experts.activation(activation, output, input)

    def enable_chunking(self):
        return self.base_experts.enable_chunking()

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        return self.base_experts.finalize_weight_and_reduce_impl()
