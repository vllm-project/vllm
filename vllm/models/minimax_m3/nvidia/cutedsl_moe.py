# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 CuTeDSL NVFP4 MoE with a deferred (unfinalized) top-k reduce.

On TP1, ``MiniMaxM3CuteDSLExperts`` can stop after GEMM2 and hand the expanded
rows to the model as an ``UnfinalizedMoEOutput``; the next layer's input norm
consumes them with ``moe_finalize_add_norm_mxfp8`` (see
``deferred_finalize_runner.py``). The pipeline below is FlashInfer's
deterministic (non-atomic) W4A4 path -- ``moe_sort``, the gather/GEMM1
activation fusion, per-token FC2-input quantization and GEMM2 with
``use_fused_finalize=False`` -- driven through FlashInfer's
``CuteDslFusedMoERunner`` so autotuning and its tactic cache keep working. It
stops before ``moe_unpermute``, so the route weights are still unapplied; the
consuming tail multiplies them in, matching ``moe_unpermute``'s numerics
(fp32 FMA in ascending top-k order).
"""

import functools

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
    quantize_nvfp4_per_token_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.platforms import current_platform

logger = init_logger(__name__)

_OP_NAME = "MiniMaxM3CuteDSLMoE"


@functools.cache
def _identity_expanded_map(
    num_tokens: int, topk: int, device: torch.device | str
) -> torch.Tensor:
    """Identity row map: the deterministic GEMM2 writes rows in expanded
    (token, slot) order."""
    return torch.arange(num_tokens * topk, device=device, dtype=torch.int32).view(
        num_tokens, topk
    )


class MiniMaxM3CuteDSLExperts(FlashInferCuteDSLExperts):
    """FlashInferCuteDSLExperts that can stop after GEMM2 (TP1).

    Set by MiniMaxM3CuteDSLMethod: return the expanded GEMM2 rows for
    MiniMaxM3DeferredFinalizeRunner instead of running the finalize.
    """

    defer_finalize = False

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        per_token_activation: bool = False,
    ) -> None:
        super().__init__(moe_config, quant_config, per_token_activation)
        from flashinfer.fused_moe.cute_dsl.tuner import CuteDslFusedMoERunner

        # The unfinalized pipeline mirrors the non-localized per-token W4A4
        # path; MiniMaxM3CuteDSLMethod rejects DP, EP and SM107 up front.
        assert moe_config.dp_size == 1
        assert not moe_config.moe_parallel_config.use_ep
        assert per_token_activation
        assert None not in (self.gemm1_alpha, self.gemm1_beta, self.gemm1_clamp_limit)
        self._identity_expanded_map = _identity_expanded_map(
            moe_config.max_num_tokens, self.topk, moe_config.device
        )
        # The runner tunes sort -> GEMM1 -> GEMM2 per token-count bucket
        # through FlashInfer's AutoTuner, persisting picks to the autotune
        # cache; only the reduce differs from the finalized path.
        self._runner = CuteDslFusedMoERunner(
            forward_impl=self._moe_unfinalized,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            use_fused_finalize=False,
            output_dtype=moe_config.in_dtype,
            activation_type=activation_to_flashinfer_int(moe_config.activation),
            use_per_token_activation=True,
            swiglu_alpha=self.gemm1_alpha,
            swiglu_beta=self.gemm1_beta,
            swiglu_limit=self.gemm1_clamp_limit,
        )

    def _moe_unfinalized(
        self,
        *,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w1_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_sf: torch.Tensor,
        w2_alpha: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        gemm1_mma_tiler_mn: tuple[int, int],
        gemm1_cluster_shape_mn: tuple[int, int],
        gemm2_mma_tiler_mn: tuple[int, int],
        gemm2_cluster_shape_mn: tuple[int, int],
        output_dtype: torch.dtype,
        per_token_scale: torch.Tensor,
        enable_pdl: bool,
        activation_type: int,
        swiglu_alpha: float,
        swiglu_beta: float,
        swiglu_limit: float,
        gemm1_mma_tiler: tuple[int, int, int] | None = None,
        gemm2_mma_tiler: tuple[int, int, int] | None = None,
        **_ignored,
    ) -> UnfinalizedMoEOutput:
        """CuteDslFusedMoERunner forward_impl: FlashInfer's deterministic W4A4
        per-token pipeline, stopping before ``moe_unpermute``."""
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (  # noqa: E501
            blockscaled_contiguous_gather_grouped_gemm_act_fusion,
        )
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E501
            blockscaled_contiguous_grouped_gemm_finalize_fusion,
        )
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            moe_sort,
            normalize_cute_dsl_moe_activation_type,
        )
        from flashinfer.quantization.kernels.nvfp4_quantize import (
            SF_LAYOUT_128x4,
            nvfp4_quantize_per_token_cute_dsl,
        )

        # SM107 (Rubin) tactics select different kernels, which have no
        # non-atomic finalize; MiniMaxM3CuteDSLMethod rejects SM107.
        assert gemm1_mma_tiler is None and gemm2_mma_tiler is None
        activation, gated = normalize_cute_dsl_moe_activation_type(activation_type)
        num_tokens = token_selected_experts.size(0)
        hidden_size = w2_weight.shape[1]

        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            _expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            _total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_local_experts,
            tile_tokens_dim=tile_size,
            enable_pdl=enable_pdl,
        )

        intermediate, _ = blockscaled_contiguous_gather_grouped_gemm_act_fusion(
            a=x,
            b=w1_weight,
            a_scale=x_sf,
            b_scale=w1_weight_sf,
            alpha=w1_alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            token_id_mapping=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            out=None,
            out_scale=None,
            global_scale=None,
            a_per_token_scale=per_token_scale,
            c_dtype="bfloat16" if output_dtype == torch.bfloat16 else "float16",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            sf_vec_size=16,
            quantize_output=False,
            topk=top_k,
            mma_tiler_mn=gemm1_mma_tiler_mn,
            cluster_shape_mn=gemm1_cluster_shape_mn,
            enable_pdl=enable_pdl,
            activation_type=activation.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            gated=gated,
        )
        intermediate, intermediate_sf, intermediate_per_token_scale = (
            nvfp4_quantize_per_token_cute_dsl(
                intermediate,
                fc2_input_scale,
                sf_layout=SF_LAYOUT_128x4,
                enable_pdl=enable_pdl,
            )
        )
        intermediate_sf = convert_sf_to_mma_layout(
            intermediate_sf,
            m=intermediate.shape[0],
            k=intermediate.shape[1] * 2,
            num_groups=1,
            sf_vec_size=16,
        )

        # Deterministic (non-atomic) finalize: GEMM2 writes one expanded row
        # per (token, slot) and the route weights stay unapplied (the
        # finalized path applies them in moe_unpermute).
        gemm2_output = torch.empty(
            (num_tokens * top_k, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )
        blockscaled_contiguous_grouped_gemm_finalize_fusion(
            a=intermediate,
            b=w2_weight,
            a_scale=intermediate_sf,
            b_scale=w2_weight_sf,
            alpha=w2_alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            out=gemm2_output,
            a_per_token_scale=intermediate_per_token_scale,
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            sf_vec_size=16,
            out_dtype="bfloat16" if output_dtype == torch.bfloat16 else "float16",
            mma_tiler_mn=gemm2_mma_tiler_mn,
            cluster_shape_mn=gemm2_cluster_shape_mn,
            enable_pdl=enable_pdl,
            use_fused_finalize=False,
        )
        # Every expanded row is written (DP/EP are rejected, so all experts
        # are local), so the identity map covers all of them.
        return UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_output,
            expert_weights=token_final_scales,
            expanded_idx_to_permuted_idx=self._identity_expanded_map[:num_tokens],
            route_major=True,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> UnfinalizedMoEOutput | None:
        num_tokens = hidden_states.shape[0]
        if (
            not self.defer_finalize
            or num_tokens == 0
            or num_tokens > self._identity_expanded_map.shape[0]
        ):
            return super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
        from flashinfer.autotuner import AutoTuner

        assert self.expects_unquantized_inputs
        x, block_scale, per_token_scale = quantize_nvfp4_per_token_input(hidden_states)
        # Standard CuteDslFusedMoERunner input layout with
        # use_per_token_activation=True: per-token row scale at index 11 and
        # the output at index 12.
        inputs = [
            x,
            block_scale.unsqueeze(-1),
            topk_ids.to(torch.int32),
            topk_weights.float(),
            w1,
            self.w1_scale,
            self.g1_alphas,
            self.per_token_global_scale,
            w2,
            self.w2_scale,
            self.g2_alphas,
            per_token_scale,
            output,
        ]
        _, tactic = AutoTuner.get().choose_one(
            _OP_NAME,
            [self._runner],
            self._runner.tuning_config,
            inputs,
        )
        return self._runner(inputs, tactic=tactic)


class MiniMaxM3CuteDSLMethod(CompressedTensorsW4A4Nvfp4MoEMethod):
    def __init__(self, moe: FusedMoEConfig, layer_name: str) -> None:
        # Checked here, before weight loading, rather than when the experts are
        # built in process_weights_after_loading.
        if moe.moe_parallel_config.use_ep or moe.dp_size > 1:
            raise ValueError(
                "MiniMax-M3 moe_backend='flashinfer_cutedsl' targets TP/PP "
                "prefill workers and is not validated with DP or EP."
            )
        if current_platform.is_device_capability(107):
            # FlashInfer's SM107 CuTeDSL MoE GEMMs have no non-atomic finalize.
            raise ValueError(
                "MiniMax-M3 moe_backend='flashinfer_cutedsl' is not available "
                "on SM107 (Rubin); use moe_backend='auto' instead."
            )
        super().__init__(
            moe,
            layer_name=layer_name,
            # Nothing in the checkpoint says how its FC1 input was quantized
            # during QAT, so the model declares it (as in ModelOptNvFp4FusedMoE).
            activation_key=(moe.qat_activation_key or kNvfp4Dynamic),
        )
        assert self.nvfp4_backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL
        self.experts_cls = MiniMaxM3CuteDSLExperts
        # Defer the top-k reduce into the next layer's input norm on TP1 (no
        # TP all-reduce of the MoE output).
        self.defer_finalize = moe.tp_size == 1
        logger.info_once(
            "MiniMax-M3 CuTeDSL NVFP4 MoE engaged (deferred finalize fused "
            "into the next input norm: %s)",
            self.defer_finalize,
            scope="process",
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        super().process_weights_after_loading(layer)
        assert self.moe_kernel is not None
        experts = self.moe_kernel.fused_experts
        assert isinstance(experts, MiniMaxM3CuteDSLExperts)
        experts.defer_finalize = self.defer_finalize


class MiniMaxM3RoutedExperts(RoutedExperts):
    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> CompressedTensorsW4A4Nvfp4MoEMethod:
        return MiniMaxM3CuteDSLMethod(moe_config, layer_name=prefix)
