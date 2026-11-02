# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x MoE: routing, experts and the post-expert-norm finalize.

Every expert backend stops after GEMM2 and hands its rows to the finalizer.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch

from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsModular,
    TrtLlmBf16ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    UnfinalizedMoEOutput,
    convert_flashinfer_moe_output,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.utils import replace_parameter
from vllm.models.muse_spark.common.moe import (  # noqa: F401 re-exported
    MuseSpark1xRouter,
    MuseSparkConsolidatedRoutedExperts,
    needs_pre_norm_all_reduce,
)
from vllm.models.muse_spark.common.moe import (
    MuseSparkMoEConfig as CommonMuseSparkMoEConfig,
)
from vllm.models.muse_spark.common.ops.fused_finalize import fused_post_expert_finalize
from vllm.models.muse_spark.nvidia.ops.cute_dsl.moe_finalize_rmsnorm import (
    moe_finalize_with_rmsnorm,
    supports_moe_finalize,
)
from vllm.models.muse_spark.nvidia.ops.mnnvl_fusions import (
    maybe_create_moe_finalize_all_reduce,
    tp_group_spans_nodes,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class MuseSparkMoEConfig(CommonMuseSparkMoEConfig):
    """NVIDIA runner fallback for unquantized MegaMoE layers."""

    def __post_init__(self):
        super().__post_init__()
        if self.moe_backend == "deep_gemm_mega_moe":
            # FlashInfer TRT-LLM needs all2all dispatch.
            self.moe_backend = (
                "flashinfer_trtllm"
                if self.moe_parallel_config.use_all2all_kernels
                else "auto"
            )


def uses_modular_all2all(moe_config: FusedMoEConfig) -> bool:
    """Whether this deployment's all2all forces the modular experts path."""
    parallel_config = moe_config.moe_parallel_config
    return (
        parallel_config.use_all2all_kernels
        and not parallel_config.use_ag_rs_all2all_kernels
    )


def validate_deployment(moe_config: FusedMoEConfig, *, kernel_routes: bool) -> None:
    """Reject deployments the post-expert norm cannot be spliced into."""
    parallel_config = moe_config.moe_parallel_config
    routing_method = moe_config.routing_method
    if kernel_routes and routing_method != RoutingMethodType.MiniMax2:
        raise NotImplementedError(
            "MuseSpark1x on the monolithic TRT-LLM kernel needs it to run the "
            "router, which requires the routing to classify as MiniMax2 "
            "(sigmoid + expert bias + sum-renormalize, no expert groups, "
            f"route_scale == 1.0); this checkpoint classifies as "
            f"{routing_method.name}."
        )
    if moe_config.is_lora_enabled:
        raise NotImplementedError(
            "MuseSpark1x post-expert norm does not support MoE LoRA: the LoRA "
            "path applies the top-k weights alongside GEMM2."
        )
    if parallel_config.enable_eplb:
        raise NotImplementedError("MuseSpark1x post-expert norm does not support EPLB.")
    if parallel_config.is_sequence_parallel and not parallel_config.use_all2all_kernels:
        # Without an all2all combine the pre-norm all-reduce would mix tokens.
        raise NotImplementedError(
            "MuseSpark1x post-expert norm does not support sequence-parallel "
            "MoE without an all2all combine."
        )
    if moe_config.skip_final_all_reduce:
        raise NotImplementedError(
            "MuseSpark1x post-expert norm all-reduces TP partials before the "
            "norm, so it cannot hand back an un-reduced output for a fused "
            "downstream all-reduce."
        )


def _validate_route_major_deployment(moe_config: FusedMoEConfig) -> None:
    validate_deployment(moe_config, kernel_routes=False)
    parallel_config = moe_config.moe_parallel_config
    if parallel_config.dp_size != 1 or parallel_config.pcp_size != 1:
        raise NotImplementedError(
            "MuseSpark1x Triton/Marlin currently requires DP1 and PCP1."
        )


# Decode-sized batches; larger ones keep the finalize + separate all-reduce.
# Measured on GB300: TP4 gains up to 256 tokens, two-node TP8 only to 128.
_FUSED_ALL_REDUCE_MAX_TOKENS = 256
_MULTI_NODE_FUSED_ALL_REDUCE_MAX_TOKENS = 128
# One workspace serves every layer: calls are sequential and share the group.
_FUSED_ALL_REDUCE_WORKSPACES: dict[tuple[int, int], Any | None] = {}

# Caps the fp32 (num_tokens * top_k)-row finalize intermediate during prefill.
_FINALIZE_CHUNK_ELEMS = 32 * 1024 * 1024


class MusePostExpertFinalize:
    """Pre-norm TP all-reduce -> per-route RMSNorm -> fp32 weighted reduce."""

    def __init__(
        self,
        norm: torch.nn.Module,
        moe_config: FusedMoEConfig,
        fuse_all_reduce: bool = False,
    ) -> None:
        self._norm = norm
        self._fused_weight: torch.Tensor | None = None
        # GEMM2 rows are TP partial sums and must be summed before the norm.
        self.pre_norm_all_reduce = needs_pre_norm_all_reduce(moe_config)
        # Whether the last call already summed the EP partials across ranks.
        self.output_reduced = False
        self._all_reduce = (
            self._create_all_reduce(moe_config) if fuse_all_reduce else None
        )
        self._tp_finalize_enabled = False
        if (
            self.pre_norm_all_reduce
            and norm.hidden_size == 4096
            and moe_config.experts_per_token == 8
            and moe_config.in_dtype == torch.bfloat16
            and torch.cuda.is_available()
        ):
            try:
                from flashinfer.comm.tp_moe_finalize_rmsnorm import (
                    init_tp_moe_finalize_rmsnorm,
                )

                self._tp_finalize_enabled = init_tp_moe_finalize_rmsnorm(
                    get_tp_group().device_group
                )
            except (ImportError, AttributeError, RuntimeError):
                logger.warning_once(
                    "MuseSpark TP MoE finalize is unavailable; using the native "
                    "finalize.",
                )

    def _create_all_reduce(self, moe_config: FusedMoEConfig):
        """Workspace to fuse the EP combine all-reduce into the finalize."""
        pc = moe_config.moe_parallel_config
        norm = self._norm
        # Only the late all-reduce of replicated-token EP partials: TP partials
        # need their sum before the norm, and all2all already combines.
        if (
            self.pre_norm_all_reduce
            or pc.ep_size <= 1
            or pc.ep_size != get_tp_group().world_size
            or pc.dp_size != 1
            or pc.pcp_size != 1
            or pc.use_all2all_kernels
            or pc.is_sequence_parallel
            or moe_config.skip_final_all_reduce
            or norm.variance_size_override is not None
        ):
            return None
        key = (norm.hidden_size, moe_config.experts_per_token)
        if key not in _FUSED_ALL_REDUCE_WORKSPACES:
            _FUSED_ALL_REDUCE_WORKSPACES[key] = maybe_create_moe_finalize_all_reduce(
                *key, _FUSED_ALL_REDUCE_MAX_TOKENS, enable_pdl=True
            )
        workspace = _FUSED_ALL_REDUCE_WORKSPACES[key]
        if workspace is None:
            return None
        self._all_reduce_max_tokens = (
            _MULTI_NODE_FUSED_ALL_REDUCE_MAX_TOKENS
            if tp_group_spans_nodes()
            else _FUSED_ALL_REDUCE_MAX_TOKENS
        )
        # Weights arrive in the activation dtype when the kernel routes and in
        # fp32 from the router otherwise; compile both before capture.
        for weight_dtype in (moe_config.in_dtype, torch.float32):
            workspace.compile(
                row_stride=norm.hidden_size,
                weight_dtype=weight_dtype,
                eps=norm.variance_epsilon,
                gain_center=norm.fused_gain_center,
                weight_before_norm=norm.applies_weight_before_norm,
            )
        return workspace

    def __call__(self, unfinalized: UnfinalizedMoEOutput, out: torch.Tensor) -> None:
        """Finalize permuted GEMM2 rows into ``out``."""
        permuted_idx = unfinalized.expanded_idx_to_permuted_idx
        num_tokens, top_k = permuted_idx.shape
        rows_all = unfinalized.gemm2_permuted
        weights_all = unfinalized.expert_weights
        norm = self._norm

        self.output_reduced = False
        if self._all_reduce is not None and num_tokens <= self._all_reduce_max_tokens:
            from flashinfer.comm.mnnvl_cutedsl_moe_finalize import (
                moe_finalize_rmsnorm_allreduce,
                supports_moe_finalize_rmsnorm_allreduce,
            )

            weight = self._effective_weight(rows_all.dtype, rows_all.device)
            if supports_moe_finalize_rmsnorm_allreduce(
                self._all_reduce, rows_all, weights_all, permuted_idx, weight, out
            ):
                moe_finalize_rmsnorm_allreduce(
                    rows_all,
                    weights_all,
                    permuted_idx,
                    weight,
                    out,
                    self._all_reduce,
                    eps=norm.variance_epsilon,
                    gain_center=norm.fused_gain_center,
                    weight_before_norm=norm.applies_weight_before_norm,
                )
                self.output_reduced = True
                return

        if self._can_fuse_tp(rows_all, weights_all, num_tokens, top_k):
            self._finalize_tp(unfinalized, out)
            return
        # Padded permuted row counts can differ across ranks; only route-major
        # rows can be all-reduced in place.
        if norm.variance_size_override is not None or (
            self.pre_norm_all_reduce and not unfinalized.route_major
        ):
            self._warn_native_fallback(rows_all, weights_all)
            self._finalize_native(unfinalized, out)
            return

        if self.pre_norm_all_reduce:
            # In the activation dtype; see `_finalize_chunk`.
            rows_all = tensor_model_parallel_all_reduce(rows_all)
        if self._can_fuse_cutedsl(rows_all, weights_all):
            moe_finalize_with_rmsnorm(
                rows_all,
                weights_all,
                permuted_idx,
                self._effective_weight(rows_all.dtype, rows_all.device),
                out,
                topk=top_k,
                eps=norm.variance_epsilon,
                gain_center=norm.fused_gain_center,
                weight_before_norm=norm.applies_weight_before_norm,
            )
        else:
            fused_post_expert_finalize(
                rows_all,
                norm.weight if norm.has_weight else None,
                weights_all,
                norm.variance_epsilon,
                norm.fused_gain_center,
                norm.applies_weight_before_norm,
                out,
                permuted_idx=permuted_idx,
            )

    def _warn_native_fallback(self, rows: torch.Tensor, weights: torch.Tensor) -> None:
        logger.warning_once(
            "MuseSpark post-expert norm fell back to the torch finalize: "
            "rows=%s weights=%s norm_weight=%s hidden=%d pre_norm_all_reduce=%s",
            rows.dtype,
            weights.dtype,
            None if self._norm.weight is None else self._norm.weight.dtype,
            self._norm.hidden_size,
            self.pre_norm_all_reduce,
        )

    def _finalize_native(
        self, unfinalized: UnfinalizedMoEOutput, out: torch.Tensor
    ) -> None:
        """The permuted tail in torch: the kernels' reference, and the fallback."""
        permuted_idx = unfinalized.expanded_idx_to_permuted_idx
        num_tokens, top_k = permuted_idx.shape
        rows_all = unfinalized.gemm2_permuted
        weights_all = unfinalized.expert_weights

        chunk = self._chunk_size(top_k)
        # TP replicates tokens, so every rank runs the same number of collectives.
        for start in range(0, num_tokens, chunk):
            end = min(start + chunk, num_tokens)
            flat_idx = permuted_idx[start:end].reshape(-1)
            # EP marks remote experts with -1; read row 0 and zero it.
            rows = rows_all.index_select(0, flat_idx.clamp_min(0).to(torch.long))
            rows = rows.masked_fill((flat_idx < 0).unsqueeze(-1), 0)
            self._finalize_chunk(rows, weights_all[start:end], out[start:end])

    def _chunk_size(self, top_k: int) -> int:
        rows = max(top_k, _FINALIZE_CHUNK_ELEMS // max(self._norm.hidden_size, 1))
        return max(1, rows // top_k)

    def _can_fuse_tp(
        self,
        rows: torch.Tensor,
        weights: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> bool:
        norm = self._norm
        return (
            self._tp_finalize_enabled
            and num_tokens <= 8192
            and top_k == 8
            and rows.dtype == torch.bfloat16
            and weights.dtype in (torch.bfloat16, torch.float32)
            and norm.variance_size_override is None
            and (not norm.has_weight or norm.weight.dtype == rows.dtype)
        )

    def _finalize_tp(
        self, unfinalized: UnfinalizedMoEOutput, out: torch.Tensor
    ) -> None:
        """Run the TP4 collective directly on the permuted GEMM2 rows."""
        from flashinfer.comm.tp_moe_finalize_rmsnorm import (
            supports_tp_moe_finalize_rmsnorm,
            tp_moe_finalize_rmsnorm,
        )

        permuted_idx = unfinalized.expanded_idx_to_permuted_idx
        rows_all = unfinalized.gemm2_permuted
        weights_all = unfinalized.expert_weights
        norm_weight = self._effective_weight(rows_all.dtype, rows_all.device)

        if not supports_tp_moe_finalize_rmsnorm(
            rows_all,
            weights_all,
            permuted_idx,
            norm_weight,
            out,
        ):
            raise RuntimeError("TP MoE finalize contract changed after dispatch")
        tp_moe_finalize_rmsnorm(
            rows_all,
            weights_all,
            permuted_idx,
            norm_weight,
            out,
            eps=self._norm.variance_epsilon,
            gain_center=self._norm.fused_gain_center,
            weight_before_norm=self._norm.applies_weight_before_norm,
        )

    def _can_fuse_cutedsl(self, rows: torch.Tensor, weights: torch.Tensor) -> bool:
        """Whether CuTeDSL applies; it can't match torch's bf16 * fp32 promotion."""
        norm = self._norm
        return (
            current_platform.is_device_capability_family(100)
            and weights.is_contiguous()
            and (not norm.has_weight or norm.weight.dtype == rows.dtype)
            and supports_moe_finalize(rows.dtype, norm.hidden_size)
        )

    def _effective_weight(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Norm weight in the activation dtype; the gain centre is added in fp32."""
        if self._fused_weight is None:
            norm = self._norm
            if norm.has_weight:
                weight = norm.weight.detach()
                assert weight.shape == (norm.hidden_size,), (
                    f"post-expert norm weight is {tuple(weight.shape)}, expected "
                    f"({norm.hidden_size},)"
                )
            else:
                weight = torch.ones(norm.hidden_size)
            self._fused_weight = weight.to(dtype=dtype, device=device).contiguous()
        return self._fused_weight

    def _finalize_chunk(
        self,
        rows: torch.Tensor,
        expert_weights: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        num_tokens, top_k = expert_weights.shape

        if self.pre_norm_all_reduce:
            # GEMM2 already rounded the rows to bf16; fp32 would double traffic.
            rows = tensor_model_parallel_all_reduce(rows.contiguous())

        # An fp32 norm output costs nothing and keeps the top-k sum exact.
        contrib = self._norm.forward(rows, out_dtype=torch.float32)
        contrib.mul_(expert_weights.reshape(-1, 1))
        out.copy_(contrib.view(num_tokens, top_k, -1).sum(dim=1))


def _attach_post_expert_finalize(
    quant_method: "MuseSparkModelOptNvFp4FusedMoE | MuseSparkUnquantizedFusedMoE",
) -> None:
    """Hand the layer's finalizer to its experts, which exist only now."""
    assert quant_method.moe_kernel is not None
    quant_method.moe_kernel.fused_experts.post_gemm2_hook = (
        quant_method.muse_post_expert_finalize
    )


@dataclass
class MuseSparkNvFp4MoEQuantConfig(FusedMoEQuantConfig):
    """``FusedMoEQuantConfig`` that also carries the w3-side GEMM1 scale."""

    g1_scale_c: torch.Tensor | None = None


class _MuseSparkNvFp4Scales:
    """Split w1/w3 GEMM1 global scales (``w13_weight_scale_2`` is [E, 2])."""

    quant_config: FusedMoEQuantConfig
    gemm1_clamp_limit: torch.Tensor | None
    gemm1_alpha: torch.Tensor | None
    gemm1_beta: torch.Tensor | None

    def _compute_g1_scale_c(self) -> torch.Tensor:
        quant_config = self.quant_config
        if (
            isinstance(quant_config, MuseSparkNvFp4MoEQuantConfig)
            and quant_config.g1_scale_c is not None
        ):
            return quant_config.g1_scale_c
        return super()._compute_g1_scale_c()  # type: ignore[misc]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Scales are already folded; the base class's folding must not rerun.
        assert (
            self.gemm1_clamp_limit is None
            and self.gemm1_alpha is None
            and self.gemm1_beta is None
        ), "MuseSpark NVFP4 MoE does not support SwiGLU clamp/alpha/beta"
        layer.register_parameter(
            "g1_scale_c",
            torch.nn.Parameter(self._compute_g1_scale_c(), requires_grad=False),
        )
        self.g1_scale_c = layer.g1_scale_c


class MuseSparkTrtLlmNvFp4Experts(_MuseSparkNvFp4Scales, TrtLlmNvFp4ExpertsMonolithic):
    """Monolithic NVFP4 experts: TP, TP+EP and allgather_reducescatter."""

    def apply_routed(
        self,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
    ) -> torch.Tensor:
        """Experts on quantized input with routing computed by the caller."""
        import flashinfer

        num_tokens = a1q.shape[0]
        assert self._uses_post_gemm2_hook(num_tokens)
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None
        flashinfer_output = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk_ids, topk_weights),
            routing_bias=None,
            hidden_states=a1q,
            hidden_states_scale=a1q_scale.view(torch.float8_e4m3fn).reshape(
                num_tokens, -1
            ),
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=self.quant_config.g1_alphas,
            output2_scale_scalar=self.quant_config.g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=0,
            topk_group=0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=self.routing_method_type,
            do_finalize=False,
            activation_type=activation_to_flashinfer_int(activation),
            tune_max_num_tokens=fi_moe_largest_bucket(self.moe_config),
        )
        return self._run_post_gemm2_hook(
            convert_flashinfer_moe_output(
                flashinfer_output,
                do_finalize=False,
                num_tokens=num_tokens,
                top_k=self.topk,
            )
        )


class MuseSparkTrtLlmNvFp4ExpertsModular(
    _MuseSparkNvFp4Scales, TrtLlmNvFp4ExpertsModular
):
    """Modular NVFP4 experts for the all2all backends (DeepEP, FlashInfer)."""


class MuseSparkModelOptNvFp4FusedMoE(ModelOptNvFp4FusedMoE):
    """ModelOpt NVFP4 MoE preserving separate w1/w3 global weight scales.

    The name must contain "ModelOpt": ``RoutedExperts.weight_loader`` matches it.
    """

    muse_post_expert_finalize: MusePostExpertFinalize

    def __init__(
        self,
        quant_config: ModelOptNvFp4Config,
        moe_config: FusedMoEConfig,
    ) -> None:
        super().__init__(quant_config, moe_config)
        self.use_route_major = (
            self.nvfp4_backend == NvFp4MoeBackend.MARLIN
            and current_platform.is_device_capability_family(90)
        )
        if self.use_route_major:
            _validate_route_major_deployment(moe_config)
            if not self.moe.is_act_and_mul:
                raise NotImplementedError("MuseSpark Marlin requires SwiGLU.")
            logger.info_once(
                "MuseSpark NVFP4: Marlin W4A16 GEMMs with Triton finalization."
            )
            self.experts_cls = MarlinExperts
            return
        if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
            raise NotImplementedError(
                "MuseSpark NVFP4 MoE requires FLASHINFER_TRTLLM or Marlin "
                f"on Hopper; got {self.nvfp4_backend}."
            )
        if self.use_a16 or not self.moe.is_act_and_mul:
            raise NotImplementedError(
                "MuseSpark NVFP4 MoE requires W4A4 SwiGLU checkpoints."
            )
        modular = uses_modular_all2all(moe_config)
        validate_deployment(moe_config, kernel_routes=not modular)
        self.experts_cls = (
            MuseSparkTrtLlmNvFp4ExpertsModular
            if modular
            else MuseSparkTrtLlmNvFp4Experts
        )

    def process_weights_after_loading(self, layer: "MuseSparkRoutedExperts") -> None:
        if self.use_route_major:
            self._process_hopper_marlin_weights(layer)
            _attach_post_expert_finalize(self)
            return

        w13_weight_scale_2 = layer.w13_weight_scale_2.data
        assert w13_weight_scale_2.ndim == 2 and w13_weight_scale_2.shape[1] == 2

        (
            w13,
            w13_scale,
            _,
            a13_scale,
            w2,
            w2_scale,
            _,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            # Unused by the TRT-LLM conversion.
            w13_scale_2=w13_weight_scale_2[:, 0].contiguous(),
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=True,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)

        # Collapsed to the global max and repeated per expert.
        a13 = a13_scale[0:1]
        a2 = a2_scale[0:1]
        g1_alphas = (a13 * w13_weight_scale_2[:, 0]).contiguous()
        g1_scale_c = (a13 * w13_weight_scale_2[:, 1] / a2).contiguous()
        g2_alphas = (a2 * layer.w2_weight_scale_2.data).contiguous()

        base_config = nvfp4_moe_quant_config(
            g1_alphas=g1_alphas,
            g2_alphas=g2_alphas,
            a1_gscale=1.0 / a13,
            a2_gscale=1.0 / a2,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            # The TRT-LLM kernel rejects swizzled activation scale factors.
            is_scale_swizzled=False,
        )
        self.moe_quant_config = MuseSparkNvFp4MoEQuantConfig(
            **vars(base_config), g1_scale_c=g1_scale_c
        )
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
        )
        _attach_post_expert_finalize(self)
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def _process_hopper_marlin_weights(self, layer: "MuseSparkRoutedExperts") -> None:
        # Keep the SiLU gate input exact; the up-side gain is compensated in
        # GEMM2, not left to RMSNorm's approximate scale invariance.
        scales = layer.w13_weight_scale_2.data
        w13_weight_scale_2 = scales[:, 0].contiguous()
        w2_weight_scale_2 = (
            layer.w2_weight_scale_2.data * (scales[:, 1] / scales[:, 0])
        ).contiguous()

        (
            w13,
            w13_scale,
            w13_scale_2,
            _,
            w2,
            w2_scale,
            w2_scale_2,
            _,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=NvFp4MoeBackend.MARLIN,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=w13_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=True,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)

        self.moe_quant_config = make_nvfp4_moe_quant_config(
            backend=NvFp4MoeBackend.MARLIN,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_scale_2=w13_scale_2,
            w2_scale_2=w2_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=MarlinExperts,
            backend=NvFp4MoeBackend.MARLIN,
            routing_tables=layer._expert_routing_tables(),
        )

    def get_fused_moe_quant_config(
        self, layer: "MuseSparkRoutedExperts"
    ) -> FusedMoEQuantConfig:
        if self.moe_quant_config is None:
            raise RuntimeError(
                "MuseSparkModelOptNvFp4FusedMoE quant config is only available "
                "after process_weights_after_loading (the split w1/w3 scales "
                "are folded there)."
            )
        return self.moe_quant_config


class MuseSparkUnquantizedFusedMoE(UnquantizedFusedMoEMethod):
    """BF16 MoE method for layers a MuseSpark NVFP4 checkpoint excludes.

    Bypasses the oracle, whose TRT-LLM BF16 routing whitelist omits MiniMax2.
    """

    muse_post_expert_finalize: MusePostExpertFinalize

    def __init__(self, moe: FusedMoEConfig) -> None:
        self.use_route_major = current_platform.is_device_capability_family(90)
        if self.use_route_major:
            _validate_route_major_deployment(moe)
            FusedMoEMethodBase.__init__(self, moe)
            self.unquantized_backend = UnquantizedMoeBackend.TRITON
            self.experts_cls = TritonExperts
            return
        super().__init__(moe)
        modular = uses_modular_all2all(moe)
        validate_deployment(moe, kernel_routes=not modular)
        self.unquantized_backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM
        self.experts_cls = (
            TrtLlmBf16ExpertsModular if modular else TrtLlmBf16ExpertsMonolithic
        )

    def _setup_kernel(
        self,
        layer: "MuseSparkRoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        super()._setup_kernel(layer, w13, w2)
        _attach_post_expert_finalize(self)


class MuseSparkRoutedExperts(MuseSparkConsolidatedRoutedExperts):
    """Consolidated-tensor experts with the NVFP4 / BF16 quant methods."""

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        if isinstance(quant_config, ModelOptNvFp4Config) and not (
            quant_config.is_layer_excluded(prefix)
        ):
            return MuseSparkModelOptNvFp4FusedMoE(quant_config, moe_config)
        # MuseSpark NVFP4 keeps layers 0 and 61 in BF16.
        return MuseSparkUnquantizedFusedMoE(moe_config)


class MuseMoERunner(MoERunner):
    """MoE runner that splices MuseSpark1x's post-expert norm into the experts."""

    def __init__(
        self,
        *args,
        post_expert_norm: torch.nn.Module,
        fuse_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        routed_experts = self.routed_experts
        if not isinstance(routed_experts, MuseSparkRoutedExperts):
            raise TypeError(
                "MuseMoERunner requires routed_experts_cls=MuseSparkRoutedExperts, "
                f"got {type(routed_experts).__name__}."
            )
        self.post_expert_finalize = MusePostExpertFinalize(
            post_expert_norm, self.moe_config, fuse_all_reduce
        )
        quant_method = cast(
            MuseSparkModelOptNvFp4FusedMoE | MuseSparkUnquantizedFusedMoE,
            routed_experts.quant_method,
        )
        quant_method.muse_post_expert_finalize = self.post_expert_finalize

    @property
    def _prepared_nvfp4_experts(self) -> MuseSparkTrtLlmNvFp4Experts | None:
        quant_method = self.routed_experts.quant_method
        if not (
            isinstance(quant_method, MuseSparkModelOptNvFp4FusedMoE)
            and quant_method.moe_kernel is not None
        ):
            return None
        experts = quant_method.moe_kernel.fused_experts
        if (
            not isinstance(experts, MuseSparkTrtLlmNvFp4Experts)
            or experts.expects_unquantized_inputs
        ):
            return None
        return experts

    @property
    def nvfp4_input_global_scale(self) -> torch.Tensor | None:
        """The static NVFP4 input scale, if the experts take prepared inputs."""
        experts = self._prepared_nvfp4_experts
        return None if experts is None else experts.quant_config.a1_gscale

    def forward_prepared_nvfp4(
        self,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor,
        routing: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """MoE output from NVFP4 input quantized with ``nvfp4_input_global_scale``.

        ``routing`` is either router logits (the kernel routes) or prepared
        ``(topk_ids, topk_weights)``.
        """
        layer = self.routed_experts
        experts = self._prepared_nvfp4_experts
        assert experts is not None
        if isinstance(routing, tuple):
            topk_ids, topk_weights = routing
            output = experts.apply_routed(
                a1q,
                a1q_scale,
                topk_ids,
                topk_weights,
                layer.w13_weight,
                layer.w2_weight,
                layer.activation,
                layer.global_num_experts,
            )
        else:
            output = experts.apply(
                hidden_states=a1q,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                router_logits=routing,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                a1q_scale=a1q_scale,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                e_score_correction_bias=layer.e_score_correction_bias,
                routed_scaling_factor=layer.routed_scaling_factor,
            )
        return self._maybe_reduce_final_output(
            output, None, self._fused_output_is_reduced
        )

    @property
    def _fused_output_is_reduced(self) -> bool:
        finalize = self.post_expert_finalize
        return (
            finalize.pre_norm_all_reduce
            or finalize.output_reduced
            or super()._fused_output_is_reduced
        )
