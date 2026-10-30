# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-threshold TRT-LLM execution over Muse MegaMoE's shared NVFP4 weights."""

import functools

import torch

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe.moe_output import (
    UnfinalizedMoEOutput,
    convert_flashinfer_moe_output,
)
from vllm.models.muse_spark.nvidia.mega_moe import MuseSparkMegaMoEExperts
from vllm.models.muse_spark.nvidia.ops.mnnvl_fusions import (
    maybe_create_moe_finalize_all_reduce,
)
from vllm.models.muse_spark.nvidia.ops.prepare_trtllm_nvfp4 import (
    norm_quant_trtllm_nvfp4,
)


@functools.cache
def _get_decode_workspace(hidden_size: int, top_k: int, max_tokens: int):
    workspace = maybe_create_moe_finalize_all_reduce(
        hidden_size, top_k, max_tokens, enable_pdl=True
    )
    if workspace is None:
        raise RuntimeError("Muse hybrid MoE requires the fused EP finalize workspace")
    return workspace


class MuseSparkHybridMoEExperts(MuseSparkMegaMoEExperts):
    """TRT-LLM for small replicated batches; MegaMoE for sequence shards."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._decode_workspace = None
        if self.is_nvfp4:
            if get_tensor_model_parallel_world_size() not in (4, 8):
                raise ValueError(
                    "Muse hybrid MoE currently supports single-node TEP4/8"
                )
            self._decode_workspace = _get_decode_workspace(
                self.hidden_size, self.top_k, self.trtllm_token_threshold
            )
            norm = self.post_expert_norm
            self._decode_workspace.compile(
                row_stride=self.hidden_size,
                weight_dtype=norm.weight.dtype,
                eps=norm.variance_epsilon,
                gain_center=norm.fused_gain_center,
                weight_before_norm=norm.applies_weight_before_norm,
            )

    def finalize_weights(self) -> None:
        if self._transformed_l1 is not None:
            return
        super().finalize_weights()
        if self.is_nvfp4:
            # Preserve MegaMoE's per-expert scales. TRT follows its existing
            # static-activation recipe, with one maximum per local expert bank.
            a13 = self.w13_input_scale.max().float().reshape(1)
            a2 = self.w2_input_scale.max().float().reshape(1)
            self._input_global_scale = a13.reciprocal()
            self._g1_gate = (a13 * self.w13_weight_scale_2[:, 0]).contiguous()
            self._g1_up = (a13 * self.w13_weight_scale_2[:, 1] / a2).contiguous()
            self._g2_scale = (a2 * self.w2_weight_scale_2).contiguous()

    def use_trtllm(self, num_tokens: int, sp_active: bool) -> bool:
        return (
            self.is_nvfp4
            and not sp_active
            and 0 < num_tokens <= self.trtllm_token_threshold
        )

    @property
    def nvfp4_input_global_scale(self) -> torch.Tensor:
        return self._input_global_scale

    def prepare_decode_inputs(self, pooled, norm, buffer, num_tokens: int):
        return norm_quant_trtllm_nvfp4(
            pooled,
            num_tokens,
            norm.weight,
            norm.variance_epsilon,
            norm.fused_gain_center,
            self._input_global_scale,
            out=(
                buffer.x[:num_tokens].view(torch.uint8),
                buffer.x_sf[:num_tokens].view(torch.uint8),
            ),
        )

    def forward_trtllm(
        self,
        quantized_input: tuple[torch.Tensor, torch.Tensor],
        buffer,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        import flashinfer
        from flashinfer.comm.mnnvl_cutedsl_moe_finalize import (
            moe_finalize_rmsnorm_allreduce,
        )

        packed, scales = quantized_input
        num_tokens = packed.shape[0]
        raw = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
            topk_ids=(
                topk_ids,
                buffer.topk_weights[:num_tokens],
            ),
            routing_bias=None,
            hidden_states=packed,
            hidden_states_scale=scales.view(torch.float8_e4m3fn),
            gemm1_weights=self.w13_weight,
            gemm1_weights_scale=self.w13_weight_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=self.w2_weight,
            gemm2_weights_scale=self.w2_weight_scale,
            gemm2_bias=None,
            output1_scale_scalar=self._g1_up,
            output1_scale_gate_scalar=self._g1_gate,
            output2_scale_scalar=self._g2_scale,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=0,
            topk_group=0,
            intermediate_size=self.intermediate_size,
            local_expert_offset=self.experts_start_idx,
            local_num_experts=self.num_local_experts,
            routed_scaling_factor=None,
            routing_method_type=7,
            do_finalize=False,
            tune_max_num_tokens=self.trtllm_token_threshold,
        )
        result = convert_flashinfer_moe_output(
            raw, do_finalize=False, num_tokens=num_tokens, top_k=self.top_k
        )
        assert isinstance(result, UnfinalizedMoEOutput)
        norm = self.post_expert_norm
        y = torch.empty(
            (num_tokens, self.hidden_size), device=packed.device, dtype=torch.bfloat16
        )
        moe_finalize_rmsnorm_allreduce(
            result.gemm2_permuted,
            result.expert_weights,
            result.expanded_idx_to_permuted_idx,
            norm.weight,
            y,
            self._decode_workspace,
            eps=norm.variance_epsilon,
            gain_center=norm.fused_gain_center,
            weight_before_norm=norm.applies_weight_before_norm,
        )
        return y
