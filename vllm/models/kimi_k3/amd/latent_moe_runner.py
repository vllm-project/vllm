# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, cast

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.model_executor.layers.quantization.online.mxfp4 import Mxfp4OnlineLinearMethod

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.linear import KimiRoutedInputLinear

logger = init_logger(__name__)


class ROCmLatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    Mirrors CUDA's LatentMoERunner, but currently only the up projection
    -sharded path is implemented. (Tier 2)

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        transform = self.routed_output_transform
        up_proj = getattr(transform, "up_proj", None)
        tp_size = get_tensor_model_parallel_world_size()

        self._up_proj_preshard = bool(getattr(transform, "row_sharded", False))
        self._up_proj_shard_size = 0
        self._tail_shardable = (
            up_proj is not None
            and tp_size > 1
            and (self._up_proj_preshard or up_proj.weight.shape[0] % tp_size == 0)
            and self._shared_experts is not None
            and not self.moe_config.is_sequence_parallel
            and self.routed_scaling_factor == 1.0
        )
        if self._tail_shardable:
            assert up_proj is not None
            self._up_proj_shard_size = (
                up_proj.weight.shape[0]
                if self._up_proj_preshard
                else up_proj.weight.shape[0] // tp_size
            )
        else:
            logger.warning_once(
                "K3 latent-MoE tail is not shardable under this config, "
                "falling back to the replicated up-projection.",
                scope="global",
            )
        self._logged_sharded_tail = False
        self._quantized_routed_input: bool | None = None

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Tier 2: column-parallel up-projection folded into the final reduce."""
        if not self._logged_sharded_tail:
            self._logged_sharded_tail = True
            logger.info_once(
                "Kimi-K3 latent-MoE tail: up-projecting only this rank's "
                "hidden shard into the shared output.",
                scope="global",
            )

        transform = self.routed_output_transform
        assert transform is not None

        latent = tensor_model_parallel_all_reduce(fused_output)
        if transform.norm is not None:
            latent = transform.norm(latent)

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        weight = transform.up_proj.weight
        up_proj_shard = (
            weight
            if self._up_proj_preshard
            else weight.narrow(0, shard_start, shard_size)
        )
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
        # beta-add epilogue so folding in the shared partial costs no kernel.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._tail_shardable and not self._fused_output_is_reduced:
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # The modular kernel is selected after weight loading. Resolve its
        # compatible prepare adapter once, during warmup before graph capture.
        if self._quantized_routed_input is None:
            self._quantized_routed_input = enable_quantized_routed_input(self)
        transform = self.routed_input_transform
        defer_projection = (
            self._quantized_routed_input
            and shared_experts_input is None
            and transform is not None
            and transform._token_group is not None
            and hidden_states.shape[0] >= transform.min_sharded_tokens
            and hidden_states.dtype == torch.bfloat16
        )
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if defer_projection:
            shared_experts_input = hidden_states
            og_hidden_dim_pre_xform: int | None = self.moe_config.hidden_dim
            og_hidden_dim_post_xform: int | None = hidden_states.shape[-1]
        elif shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        if not defer_projection:
            hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
                self._maybe_pad_hidden_states(
                    shared_experts_input,
                    hidden_states,
                )
            )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if defer_projection or self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = cast(tuple[torch.Tensor, torch.Tensor], result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        result = self._shard_up_proj_tail(
            fused_output, shared_output, og_hidden_dim_post_xform
        )

        return self._maybe_add_zero_expert_output(result)


class KimiLatentAiterExperts(AiterMxfp4Experts):
    def output_hidden_dim(self, hidden_dim: int) -> int:
        # Input may still be in model space when prepare owns the projection.
        return self.moe_config.hidden_dim

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        experts, rows, intermediate, width, topk = super().moe_problem_size(
            a1, w1, w2, topk_ids
        )
        if a1.dtype == torch.float4_e2m1fn_x2:
            width *= 2
        return experts, rows, intermediate, width, topk


class KimiLatentPrepare(MoEPrepareAndFinalizeNoDPEPModular):
    def __init__(self, projection: "KimiRoutedInputLinear"):
        self.projection = projection

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        if a1.shape[-1] == self.projection.input_size:
            assert a1.shape[0] >= self.projection.min_sharded_tokens
            assert not apply_router_weight_on_input
            values, scales = self.projection.forward_quantized(a1)
            return values, scales, None, None, None
        return super().prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )


def enable_quantized_routed_input(runner) -> bool:
    """Use the existing modular prepare/scale interface for Kimi A4 prefill."""
    # linear.py imports this module, so import the layer class here.
    from vllm.models.kimi_k3.amd.linear import KimiRoutedInputLinear

    projection = runner.routed_input_transform
    if (
        not isinstance(projection, KimiRoutedInputLinear)
        or projection._token_group is None
        # These methods return BF16 activations. Their existing A4 quantization
        # can happen on the local token shard before gathering values/scales.
        or (
            projection.weight.dtype != torch.bfloat16
            and not isinstance(
                projection.quant_method,
                (Mxfp4OnlineLinearMethod, Fp8PtpcOnlineLinearMethod),
            )
        )
        or get_current_vllm_config().lora_config is not None
        or runner.routed_experts.activation != MoEActivation.SITU
        or not rocm_aiter_ops.is_fused_moe_situv2_enabled()
        or rocm_aiter_ops.is_fused_moe_situv2_gate_up_interleaved()
        or runner.routed_experts.apply_router_weight_on_input
    ):
        return False
    method = runner._quant_method
    kernel = getattr(method, "moe_kernel", None)
    if (
        kernel is None
        or type(kernel.fused_experts) is not AiterMxfp4Experts
        or type(kernel.prepare_finalize) is not MoEPrepareAndFinalizeNoDPEPModular
        or kernel.fused_experts.post_gemm2_hook is not None
        or runner.moe_config.hidden_dim != projection.output_size
        or runner.moe_config.hidden_dim_unpadded != projection.output_size
    ):
        return False
    method.moe_kernel = FusedMoEKernel(
        KimiLatentPrepare(projection),
        KimiLatentAiterExperts(
            kernel.fused_experts.moe_config, kernel.fused_experts.quant_config
        ),
    )
    return True
