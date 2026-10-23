# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x text model for AMD ROCm GPUs."""

import functools
import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm import envs
from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsLoRA
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    sequence_parallel_chunk,
)
from vllm.models.muse_spark.common.layers import (
    MulTanhSoftCap,
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xQKVOGateParallelLinear,
    MuseSpark1xResidualConnection,
    MuseSpark1xResidualStream,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    materialize_residual_path_constants,
    norm_then_gated_residual,
)
from vllm.models.muse_spark.common.moe import MuseSpark1xRouter, MuseSparkMoEConfig
from vllm.models.muse_spark.common.ops.fused_qk_norm_rope_kv import (
    supports_fused_kv_insert,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
    fused_norm_residual_norm_fwd,
    supports_norm_residual_norm,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.third_party.flash_linear_attention.ops.kda import (
    rms_norm_gated,
)
from vllm.transformers_utils.configs.muse_spark import (
    MuseSparkDSparkConfig,
    MuseSparkTextConfig,
)

from .muse_moe_runner import (
    MuseMoERunner,
    MuseSparkQuarkOCPMXFusedMoE,
    MuseSparkRoutedExperts,
    token_sharded_moe_output,
)
from .ops.fused_norm_expand import fused_rmsnorm_expand
from .ops.fused_qk_norm_rope_kv import rocm_fused_qk_norm_rope_kv_insert
from .ops.router_bf16x3 import (
    router_bf16x3_gemm,
    split_fp32_to_bf16x3,
    supports_router_bf16x3,
)

logger = init_logger(__name__)

# Below this the fp32 Triton router GEMM is at least as fast in serving.
_ROUTER_BF16X3_MIN_TOKENS = 16


def _supports_router_bf16x3_device() -> bool:
    if not current_platform.is_rocm():
        return False

    # Importing a foreign platform can reload torch extension libraries.
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def token_sharded_stream_supported(vllm_config: VllmConfig) -> bool:
    """Whether the residual stream may be sharded by token across the TP ranks."""
    pc = vllm_config.parallel_config
    config = vllm_config.model_config.hf_text_config
    return (
        envs.VLLM_ROCM_MUSE_SP_MIN_TOKENS > 0
        and not isinstance(config, MuseSparkDSparkConfig)
        and (
            pc.enable_expert_parallel or envs.VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE == "bf16"
        )
        and pc.tensor_parallel_size > 1
        and pc.data_parallel_size == 1
        and pc.pipeline_parallel_size == 1
        and pc.prefill_context_parallel_size == 1
        and pc.decode_context_parallel_size == 1
        and not pc.use_sequence_parallel_moe
        and all(t == "sparse" for t in config.mlp_layer_types)
    )


class TokenShard:
    """Current TP rank's rows of a token-sharded residual stream."""

    __slots__ = ("num_tokens", "rows", "start", "world")

    def __init__(self, num_tokens: int) -> None:
        self.world = get_tensor_model_parallel_world_size()
        self.num_tokens = num_tokens
        self.rows = -(-num_tokens // self.world)
        self.start = get_tensor_model_parallel_rank() * self.rows

    def local(self, x: torch.Tensor) -> torch.Tensor:
        part = x[self.start : self.start + self.rows]
        if part.shape[0] < self.rows:
            part = torch.cat(
                [part, part.new_zeros((self.rows - part.shape[0], *part.shape[1:]))]
            )
        return part

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_gather(x.contiguous(), 0)[: self.num_tokens]

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.rows * self.world - x.shape[0]
        if pad:
            x = torch.cat([x, x.new_zeros((pad, *x.shape[1:]))])
        return tensor_model_parallel_reduce_scatter(x.contiguous(), 0)


def norm_residual_then_norm(
    x: MuseSpark1xResidualStream,
    pre_norm: MuseSpark1xRMSNorm,
    *,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-norm, blend and pre-norm in one kernel; returns ``(out, new_stream)``."""
    post_norm, connection = x.post_norm, x.connection
    branch, stream = x.branch, x.stream
    if (
        isinstance(connection, MuseSpark1xNormPreservingResidualConnection)
        and getattr(connection, "_coefficients_ready", False)
        and post_norm.supports_fused_kernel(branch)
        and pre_norm.supports_fused_kernel(stream)
        and supports_norm_residual_norm(branch, stream)
    ):
        return fused_norm_residual_norm_fwd(
            branch,
            stream,
            post_norm.weight if post_norm.has_weight else None,
            pre_norm.weight,
            post_norm.variance_epsilon,
            pre_norm.variance_epsilon,
            alpha=connection.alpha,
            beta=connection.beta,
            post_gain_center=post_norm.fused_gain_center,
            pre_gain_center=pre_norm.fused_gain_center,
            post_input_scale=(
                post_norm.applies_weight_before_norm and post_norm.has_weight
            ),
            pre_input_scale=(
                pre_norm.applies_weight_before_norm and pre_norm.has_weight
            ),
            out_dtype=out_dtype,
        )
    new_stream = norm_then_gated_residual(x)
    return pre_norm(new_stream, out_dtype=out_dtype), new_stream


class MuseSparkTorchLinearMethod(UnquantizedLinearMethod):
    """Run Muse Spark's compression projections through PyTorch linear."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


class MuseSpark1xMoE(nn.Module):
    post_norm_routed_experts_cls = MuseSparkRoutedExperts
    post_norm_runner_cls = MuseMoERunner

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.defer_reduce = (
            token_sharded_stream_supported(vllm_config)
            and vllm_config.parallel_config.enable_expert_parallel
        )

        self.ep_size = get_ep_group().device_group.size()
        self.n_routed_experts = config.num_local_experts
        self.n_logical_experts = config.num_local_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.n_shared_experts = config.num_shared_experts

        self.router_dtype = torch.float32
        self.router = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            out_dtype=self.router_dtype,
            params_dtype=self.router_dtype,
            prefix=f"{prefix}.router",
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
        )
        self.layer_idx = layer_idx

        self.route_scale = config.routed_scaling_factor

        self.router_output_multiplier = config.output_multiplier
        self._router_multiplier_folded = False
        self.router.weight.weight_loader = self._load_router_weight

        self.moe_router = MuseSpark1xRouter(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_local_experts,
            expert_bias=self.expert_bias,
            route_scale=self.route_scale,
            router_output_multiplier=self.router_output_multiplier,
            router_score_normalization="sigmoid",
            router_score_division_eps=1e-15,
        )

        hidden_size = config.moe_hidden_size
        # Plain F.linear: AITER's gfx950 tuned-GEMM table picks lossy split-K
        # BF16-accumulating kernels for these shapes.
        self.pooler = ReplicatedLinear(
            config.hidden_size,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.pooler",
        )
        self.pooler.quant_method = MuseSparkTorchLinearMethod()
        self.unpooler = ReplicatedLinear(
            hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.unpooler",
        )
        self.unpooler.quant_method = MuseSparkTorchLinearMethod()

        self.pre_expert_norm = MuseSpark1xRMSNorm(
            hidden_size=hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            is_post_norm=False,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
            output_norm_gain_center_type="zero",
        )

        self.intermediate_size = config.moe_intermediate_size

        self.post_expert_norm = MuseSpark1xRMSNormWithInputScale(
            hidden_size=hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=None,
            output_norm_gain_center_type="zero",
        )
        self.experts: nn.Module = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=True,
            router=self.moe_router,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            is_sequence_parallel=self.is_sequence_parallel,
            router_logits_dtype=self.router_dtype,
            scoring_func="sigmoid",
            e_score_correction_bias=self.expert_bias,
            routed_scaling_factor=self.route_scale,
            routed_experts_cls=self.post_norm_routed_experts_cls,
            runner_cls=self.post_norm_runner_cls,
            runner_args={"post_expert_norm": self.post_expert_norm},
            moe_config_cls=MuseSparkMoEConfig,
            reduce_results=not self.defer_reduce,
        )

    def _load_router_weight(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        self.router.weight_loader(param, loaded_weight)
        self._router_multiplier_folded = False

    def fold_router_output_multiplier(self) -> None:
        """Fold the router output multiplier and split the bf16x3 planes."""
        if (
            not self._router_multiplier_folded
            and self.router_output_multiplier is not None
        ):
            with torch.no_grad():
                self.router.weight.mul_(self.router_output_multiplier)
            self._router_multiplier_folded = True
            self.moe_router.router_output_multiplier = None
        weight = self.router.weight.data
        # Configs are tuned and validated on gfx950 only.
        if (
            _supports_router_bf16x3_device()
            and weight.dtype == torch.float32
            and supports_router_bf16x3(*weight.shape)
        ):
            planes = split_fp32_to_bf16x3(weight)
            if getattr(self, "_router_planes", None) is None:
                self._router_planes = planes
            else:
                self._router_planes.copy_(planes)

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        planes = getattr(self, "_router_planes", None)
        if (
            planes is not None
            and hidden_states.shape[0] >= _ROUTER_BF16X3_MIN_TOKENS
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
        ):
            return router_bf16x3_gemm(hidden_states, planes)
        router_logits, _ = self.router(hidden_states)
        return router_logits

    @functools.cached_property
    def _experts_take_expanded_input(self) -> bool:
        """Whether the experts consume one input row per route (AITER MXFP4)."""
        norm = self.pre_expert_norm
        return norm.variance_size_override is None and isinstance(
            self.experts.routed_experts.quant_method, MuseSparkQuarkOCPMXFusedMoE
        )

    def _pool(
        self, hidden_states: torch.Tensor, shard: TokenShard | None = None
    ) -> torch.Tensor:
        hidden_states, _ = self.pooler(hidden_states)
        if shard is not None:
            hidden_states = shard.gather(hidden_states)
        norm = self.pre_expert_norm
        if self._experts_take_expanded_input and hidden_states.is_contiguous():
            return fused_rmsnorm_expand(
                hidden_states,
                norm.weight if norm.has_weight else None,
                norm.variance_epsilon,
                norm.fused_gain_center,
                norm.applies_weight_before_norm and norm.has_weight,
                self.top_k,
            )
        return norm(hidden_states)

    def forward(
        self, hidden_states: torch.Tensor, shard: TokenShard | None = None
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits = self._router_logits(hidden_states)
        hidden_states = self._pool(hidden_states, shard)
        if shard is not None:
            router_logits = shard.gather(router_logits)

        if shard is not None and not self.defer_reduce:
            with token_sharded_moe_output():
                experts_out = self.experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )
            experts_out = shard.local(experts_out)
        else:
            experts_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            if shard is not None:
                experts_out = shard.reduce_scatter(experts_out)
            elif self.defer_reduce:
                experts_out = tensor_model_parallel_all_reduce(experts_out)

        experts_out, _ = self.unpooler(experts_out)

        if self.is_sequence_parallel:
            experts_out = tensor_model_parallel_all_gather(experts_out, 0)
            experts_out = experts_out[:num_tokens]

        return experts_out


class MuseSpark1xAttention(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig | MuseSparkDSparkConfig,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        defer_reduce: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        is_draft = isinstance(config, MuseSparkDSparkConfig)
        if is_draft:
            self.use_rope = config.use_rope
            self.count_backward_layer_id = self.layer_idx
            if config.every_n_layers_nope is not None:
                self.count_backward_layer_id += (
                    config.every_n_layers_nope
                    - config.num_hidden_layers % config.every_n_layers_nope
                )
                self.use_rope = self.use_rope and (
                    (config.num_hidden_layers - self.layer_idx - 1)
                    % config.every_n_layers_nope
                    != 0
                )
            rope_theta = config.rope_theta
        else:
            rope_theta = config.layer_rope_theta[self.layer_idx]
            self.use_rope = rope_theta > 0
        self.use_qk_norm = config.use_qk_norm if is_draft else True
        self.use_o_norm = config.use_attn_o_norm if is_draft else True
        self.use_o_gate = config.use_attn_o_gate if is_draft else True
        norm_affine = config.norm_affine if is_draft else True
        zero_centered_gamma = config.norm_zero_centered_gamma if is_draft else True
        gain_center_type = config.post_norm_gain_center_type if is_draft else "one"

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qk_norm = (
            MuseSpark1xRMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,  # matches the training side
                is_post_norm=False,
                zero_centered_gamma=False,  # matches the training side
                post_norm_gain_center_type=gain_center_type,
            )
            if self.use_qk_norm
            else None
        )
        self.default_dtype = torch.get_default_dtype()
        self.o_norm = (
            MuseSpark1xRMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
                is_post_norm=False,
                zero_centered_gamma=False,
                post_norm_gain_center_type=gain_center_type,
            )
            if self.use_o_norm
            else None
        )
        self.fuse_o_norm_gate = self.use_o_norm and self.use_o_gate
        self.rms_norm_eps = config.rms_norm_eps

        self.scale_query_by = (
            (
                3.87
                if config.use_sqrt_attn_scaling
                else 43.7840518911 / math.sqrt(self.head_dim)
            )
            if is_draft
            else config.qk_scale_factor
        )
        if self.use_qk_norm:
            # Exact since RoPE is a rotation; keeps the QK-norm weightless.
            self.scaling *= self.scale_query_by

        self.pre_qkv_proj_norm = MuseSpark1xRMSNorm(
            hidden_size=self.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=norm_affine,
            is_post_norm=False,
            zero_centered_gamma=zero_centered_gamma,
            post_norm_gain_center_type=gain_center_type,
            residual_id=2 * self.layer_idx + 1,  # matches the reference indexing
        )

        if not self.use_o_gate:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=self.hidden_size,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        else:
            self.qkv_proj = MuseSpark1xQKVOGateParallelLinear(
                hidden_size=self.hidden_size,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        self.defer_reduce = defer_reduce
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=not defer_reduce,
            prefix=f"{prefix}.o_proj",
        )

        # GPT-J style: rotates interleaved adjacent channel pairs.
        self.rotary_emb = (
            RotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_theta,
                is_neox_style=False,
                dtype=(
                    torch.float
                    if not is_draft or config.rope_use_fp32_in_outer_product
                    else self.default_dtype
                ),
            )
            if self.use_rope
            else None
        )

        sliding_window = None
        if is_draft:
            if global_attn_cfg := config.global_attn_cfg:
                sw_entry = global_attn_cfg[
                    self.count_backward_layer_id % len(global_attn_cfg)
                ]
                if isinstance(sw_entry, int) and sw_entry > 0:
                    sliding_window = sw_entry
        elif config.layer_types[self.layer_idx] == "sliding_attention":
            sliding_window = config.sliding_window

        self.use_attn_temperature_tuning = (
            is_draft
            and config.attn_temperature_tuning_floor_scale is not None
            and config.attn_temperature_tuning_q_scale_constant is not None
            and config.attn_temperature_tuning_layers == "global"
            and sliding_window is None
        )
        if self.use_attn_temperature_tuning:
            self.attn_temperature_tuning_floor_scale = (
                config.attn_temperature_tuning_floor_scale
            )
            self.attn_temperature_tuning_q_scale = (
                config.attn_temperature_tuning_q_scale_constant
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

        # Needs a model-written KV cache (ROCM_AITER_FA without shuffle layout).
        self.fused_qk_norm_rope_kv_insert = (
            self.use_qk_norm
            and self.qk_norm is not None
            and not self.qk_norm.has_weight
            and not self.use_attn_temperature_tuning
            and self.head_dim in (64, 128, 256)
            and self.default_dtype in (torch.bfloat16, torch.float16)
            and supports_fused_kv_insert(self.attn)
        )
        self.attn.kv_cache_update_bypass = self.fused_qk_norm_rope_kv_insert

    @staticmethod
    def compute_attn_temperature_scale(
        positions: torch.Tensor,
        floor_scale: float,
        q_scale: float,
    ) -> torch.Tensor:
        """Query scale: 1.0 up to ``floor_scale``, logarithmic above it."""
        scaling = torch.log((positions.float() + 1.0) / floor_scale) * q_scale + 1.0
        return scaling.clamp_min(1.0).unsqueeze(-1)

    def _get_attn_temperature_scale(self, positions: torch.Tensor) -> torch.Tensor:
        assert self.attn_temperature_tuning_floor_scale is not None
        assert self.attn_temperature_tuning_q_scale is not None
        return self.compute_attn_temperature_scale(
            positions,
            self.attn_temperature_tuning_floor_scale,
            self.attn_temperature_tuning_q_scale,
        )

    def _qk_norm_rope_kv_insert(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """QK-norm, temperature scale and RoPE; returns ``(q, k, v, o_gate)``."""
        cos_sin_cache = None
        rotary_dim = 0
        if self.use_rope:
            assert self.rotary_emb is not None
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            rotary_dim = self.rotary_emb.rotary_dim

        if self.fused_qk_norm_rope_kv_insert:
            rocm_fused_qk_norm_rope_kv_insert(
                qkv,
                positions,
                cos_sin_cache,
                self.attn.layer_name,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                rotary_dim,
                self.rms_norm_eps,
            )

        if self.use_o_gate:
            q, k, v, og = qkv.split(
                [self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            og = None
        if self.fused_qk_norm_rope_kv_insert:
            return q, k, v, og

        if self.qk_norm is not None:
            q = self.qk_norm(
                q.reshape(-1, self.head_dim), out_dtype=torch.float32
            ).reshape(-1, self.q_size)
            k = self.qk_norm(
                k.reshape(-1, self.head_dim), out_dtype=torch.float32
            ).reshape(-1, self.kv_size)
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        if self.use_attn_temperature_tuning:
            q = (q * self._get_attn_temperature_scale(positions)).to(q.dtype)

        if self.rotary_emb is not None:
            # forward_cuda takes the flat layout and rotates in place.
            q, k = self.rotary_emb.forward_cuda(positions, q, k)
        return q, k, v, og

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        shard: TokenShard | None = None,
    ) -> torch.Tensor:
        """Attention over an input the caller has already normalized."""
        if shard is not None:
            hidden_states = shard.gather(hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)

        q, k, v, og = self._qk_norm_rope_kv_insert(qkv, positions)

        # Withhold k/v so the attention layer does not rewrite the KV cache.
        if self.fused_qk_norm_rope_kv_insert:
            attn_output = self.attn(q, None, None)
        else:
            attn_output = self.attn(q, k, v)

        if self.fuse_o_norm_gate:
            assert og is not None
            attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            og = og.reshape(-1, self.num_heads, self.head_dim)
            attn_output = rms_norm_gated(
                attn_output,
                og,
                None,
                None,
                "sigmoid",
                eps=self.rms_norm_eps,
            )
        else:
            if self.use_o_norm:
                assert self.o_norm is not None
                attn_output = self.o_norm(attn_output.reshape(-1, self.head_dim))
                attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            if self.use_o_gate:
                assert og is not None
                og = og.view(-1, self.num_heads, self.head_dim)
                if attn_output.dim() == 2:
                    attn_output = attn_output.view(-1, self.num_heads, self.head_dim)
                attn_output = og.sigmoid() * attn_output

        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        output, _ = self.o_proj(attn_output)
        if shard is not None:
            return shard.reduce_scatter(output)
        if self.defer_reduce:
            output = tensor_model_parallel_all_reduce(output)
        return output


class MuseSpark1xDecoderLayer(nn.Module):
    moe_cls = MuseSpark1xMoE

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        is_draft = isinstance(config, MuseSparkDSparkConfig)
        norm_affine = config.norm_affine if is_draft else True
        post_norm_affine = config.post_norm_affine if is_draft else False
        zero_centered_gamma = config.norm_zero_centered_gamma if is_draft else True
        gain_center_type = config.post_norm_gain_center_type if is_draft else "one"
        output_gain_center_type = (
            config.output_norm_gain_center_type if is_draft else "zero"
        )
        learnable_scales = config.use_ffn_learnable_scales if is_draft else True
        post_norm_eps = (
            (config.rms_post_norm_eps or config.rms_norm_eps)
            if is_draft
            else config.post_norm_eps
        )

        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size

        self.self_attn = MuseSpark1xAttention(
            config=config,
            max_position_embeddings=max(
                config.max_position_embeddings,
                vllm_config.model_config.max_model_len,
            ),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            defer_reduce=token_sharded_stream_supported(vllm_config),
        )
        self.is_moe_layer = (
            (
                config.moe.is_enabled
                and config.moe.interleave_moe_layer_step > 0
                and (self.layer_idx + 1) % config.moe.interleave_moe_layer_step == 0
                and self.layer_idx >= config.moe.n_dense_layers_before_moe
            )
            if is_draft
            else config.mlp_layer_types[self.layer_idx] == "sparse"
        )
        self.pre_ffn_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=norm_affine,
            is_post_norm=False,
            zero_centered_gamma=zero_centered_gamma,
            post_norm_gain_center_type=gain_center_type,
            residual_id=2 * self.layer_idx + 2,
            output_norm_gain_center_type=output_gain_center_type,
        )

        if self.is_moe_layer:
            self.feed_forward = self.moe_cls(
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=(
                    config.intermediate_size_mlp
                    if is_draft
                    else config.intermediate_size
                ),
                hidden_act="silu",
                quant_config=quant_config,
                bias=False,
                prefix=f"{prefix}.feed_forward",
            )
        self.post_attn_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=post_norm_eps,
            has_weight=post_norm_affine,
            is_post_norm=True,
            zero_centered_gamma=zero_centered_gamma,
            post_norm_gain_center_type=gain_center_type,
            residual_id=2 * self.layer_idx + 1,
            output_norm_gain_center_type=output_gain_center_type,
        )
        post_ffn_cls = (
            MuseSpark1xRMSNormWithInputScale if learnable_scales else MuseSpark1xRMSNorm
        )
        self.post_ffn_norm = post_ffn_cls(
            config.hidden_size,
            eps=post_norm_eps,
            has_weight=(learnable_scales if learnable_scales else post_norm_affine),
            is_post_norm=True,
            zero_centered_gamma=zero_centered_gamma,
            post_norm_gain_center_type=gain_center_type,
            residual_id=2 * self.layer_idx + 2,
            output_norm_gain_center_type=output_gain_center_type,
        )
        self.residual_dtype = config.residual_dtype if is_draft else torch.float32
        self.default_dtype = torch.get_default_dtype()

        if not is_draft or config.use_residual_gate:
            self.add_attn = MuseSpark1xNormPreservingResidualConnection(
                dim=config.hidden_size,
                activation="sigmoid",
                temperature=0.3 if is_draft else config.residual_gate_temperature,
                use_stream_gate=True,
                channel_wise=True,
            )
            self.add_ffn = MuseSpark1xNormPreservingResidualConnection(
                dim=config.hidden_size,
                activation="sigmoid",
                temperature=0.3 if is_draft else config.residual_gate_temperature,
                use_stream_gate=True,
                channel_wise=True,
            )
        else:
            self.add_attn = MuseSpark1xResidualConnection()
            self.add_ffn = MuseSpark1xResidualConnection()

    def forward(
        self,
        positions: torch.Tensor,
        x: MuseSpark1xResidualStream | torch.Tensor,
        shard: TokenShard | None = None,
    ) -> MuseSpark1xResidualStream:
        """Run attention and FFN; the FFN boundary is returned unblended.

        With ``shard``, the stream holds the current rank's tokens only.
        """
        if isinstance(x, torch.Tensor):
            stream = x
            attn_input = self.self_attn.pre_qkv_proj_norm(
                stream, out_dtype=self.default_dtype
            )
        else:
            attn_input, stream = norm_residual_then_norm(
                x, self.self_attn.pre_qkv_proj_norm, out_dtype=self.default_dtype
            )
        assert stream.dtype == self.residual_dtype

        attn_out = self.self_attn(
            positions=positions, hidden_states=attn_input, shard=shard
        )
        ffn_in, stream = norm_residual_then_norm(
            MuseSpark1xResidualStream(
                stream=stream,
                branch=attn_out,
                post_norm=self.post_attn_norm,
                connection=self.add_attn,
            ),
            self.pre_ffn_norm,
            out_dtype=self.default_dtype,
        )
        if shard is None:
            branch = self.feed_forward(ffn_in)
        else:
            branch = self.feed_forward(ffn_in, shard)
        return MuseSpark1xResidualStream(
            stream=stream,
            branch=branch,
            post_norm=self.post_ffn_norm,
            connection=self.add_ffn,
        )


def _drop_gate_up_stacking(base: WeightsMapper) -> WeightsMapper:
    """Keep Quark's per-expert gate/up names unstacked for ``RoutedExperts``."""
    return WeightsMapper(
        orig_to_new_renaming=base.orig_to_new_renaming,
        orig_to_new_regex=base.orig_to_new_regex,
        orig_to_new_substr=base.orig_to_new_substr,
        orig_to_new_stacked={
            source: target
            for source, target in base.orig_to_new_stacked.items()
            if source not in (".gate_proj", ".up_proj")
        },
        orig_to_new_prefix=base.orig_to_new_prefix,
        orig_to_new_suffix=base.orig_to_new_suffix,
    )


# Runs uncompiled under breakable cudagraphs; drop LlamaModel's torch.compile.
@ignore_torch_compile
class MuseSpark1xModel(LlamaModel):
    hf_to_vllm_mapper = _drop_gate_up_stacking(LlamaModel.hf_to_vllm_mapper)

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[MuseSpark1xDecoderLayer] = MuseSpark1xDecoderLayer,
    ):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

        config = self.config
        if get_pp_group().is_last_rank:
            self.norm = MuseSpark1xRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                is_post_norm=False,
                has_weight=True,
                zero_centered_gamma=True,
                post_norm_gain_center_type="one",
                residual_id=-1,
            )
        else:
            self.norm = PPMissingLayer()

        self.tok_embeddings_norm = MuseSpark1xRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

        self.default_dtype = torch.get_default_dtype()
        self.residual_dtype = torch.float32
        self.token_shard_min_tokens = (
            envs.VLLM_ROCM_MUSE_SP_MIN_TOKENS
            if token_sharded_stream_supported(vllm_config)
            else 0
        )
        # TP-sharded experts reduce-scatter the [T, top_k, H]. Padding to a
        # multiple of TP would require copying, so shard only when they split
        # evenly.
        self.token_shard_divisor = (
            1
            if vllm_config.parallel_config.enable_expert_parallel
            else get_tensor_model_parallel_world_size()
        )

        self.fuse_across_layers = get_pp_group().world_size == 1
        self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors

    def _make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size),
                    dtype=self.residual_dtype,
                    device=device,
                ),
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        return self.tok_embeddings_norm(hidden_states, out_dtype=self.residual_dtype)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        num_tokens = hidden_states.shape[0]
        x: MuseSpark1xResidualStream | torch.Tensor = hidden_states.to(
            self.residual_dtype
        )
        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)
        aux_hidden_states: list[torch.Tensor] = []
        if get_pp_group().is_first_rank:
            self._maybe_add_hidden_state(aux_hidden_states, self.start_layer, x, None)
        shard = None
        if (
            self.token_shard_min_tokens
            and num_tokens >= self.token_shard_min_tokens
            and num_tokens % self.token_shard_divisor == 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            shard = TokenShard(num_tokens)
            x = shard.local(x)
        for idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer], start=self.start_layer
        ):
            x = layer(positions, x, shard)
            if not self.fuse_across_layers and isinstance(x, MuseSpark1xResidualStream):
                x = norm_then_gated_residual(x)
            if idx + 1 in self.aux_hidden_state_layers:
                state = (
                    norm_then_gated_residual(x)
                    if isinstance(x, MuseSpark1xResidualStream)
                    else x
                )
                if shard is not None:
                    state = shard.gather(state)
                aux_hidden_states.append(state)

        if not get_pp_group().is_last_rank:
            assert isinstance(x, torch.Tensor)  # fuse_across_layers is off here
            return IntermediateTensors(
                {
                    "hidden_states": x,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )

        if isinstance(x, torch.Tensor):
            output = self.norm(x, out_dtype=self.default_dtype)
        else:
            output, _ = norm_residual_then_norm(
                x, self.norm, out_dtype=self.default_dtype
            )
        if shard is not None:
            output = shard.gather(output)
        aux_hidden_states = remote_aux + aux_hidden_states
        if aux_hidden_states:
            return output, aux_hidden_states
        return output


class MuseSpark1xForCausalLM(LlamaForCausalLM, SupportsLoRA, MixtureOfExperts):
    model_cls = MuseSpark1xModel
    decoder_layer_cls = MuseSpark1xDecoderLayer
    supported_quantization_configs: tuple[type[QuantizationConfig], ...] = (
        QuarkConfig,
    )
    supported_quantization_description = "Quark W8A8 FP8 or OCP-MXFP4 W4A4"

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper()

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[MuseSpark1xDecoderLayer] | None = None,
    ):
        config = vllm_config.model_config.hf_text_config
        gated_delta_net = getattr(config, "gated_delta_net", None)
        if gated_delta_net is not None and getattr(gated_delta_net, "enable", False):
            raise NotImplementedError(
                "MuseSpark1x gated_delta_net layers are not supported by the "
                "native vLLM text stack."
            )
        if vllm_config.quant_config is not None and not isinstance(
            vllm_config.quant_config, self.supported_quantization_configs
        ):
            raise NotImplementedError(
                "MuseSpark1x on ROCm only supports "
                f"{self.supported_quantization_description} quantized checkpoints; "
                "got quantization "
                f"{vllm_config.quant_config.get_name()!r}."
            )
        if layer_type is None:
            layer_type = self.decoder_layer_cls

        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

        # qkv_proj fuses Q+K+V+OG, so LoRA must treat it as one module.
        self.packed_modules_mapping = {  # type: ignore[misc]
            k: v for k, v in self.packed_modules_mapping.items() if k != "qkv_proj"
        }

        self.metap_output_multiplier = config.output_multiplier
        self.output_soft_cap_temp = config.final_logit_softcapping

        self.mul_tanh_soft_cap = (
            MulTanhSoftCap(
                multiplier=self.metap_output_multiplier,
                soft_cap_temp=self.output_soft_cap_temp,
            )
            if (
                self.metap_output_multiplier != 1.0
                and self.output_soft_cap_temp is not None
            )
            else None
        )

        if get_pp_group().is_last_rank:
            self.logits_processor.scale = 1.0

        self.set_moe_parameters()

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[MuseSpark1xDecoderLayer] = MuseSpark1xDecoderLayer,
    ):
        return self.model_cls(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )

    def set_moe_parameters(self):
        self.moe_layers: list[nn.Module] = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, MuseSpark1xDecoderLayer)
            if isinstance(layer.feed_forward, MuseSpark1xMoE):
                example_moe = layer.feed_forward
                self.moe_layers.append(layer.feed_forward.experts)

        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("No MuseSpark1xMoE layer found in model.layers.")
        else:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.feed_forward, MuseSpark1xMoE):
                moe = layer.feed_forward
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        if logits.dtype != torch.float32:
            logits = logits.to(torch.float32)
        if self.mul_tanh_soft_cap is not None:
            return self.mul_tanh_soft_cap(logits)
        if self.metap_output_multiplier != 1.0:
            logits = logits * self.metap_output_multiplier
        if self.output_soft_cap_temp is not None:
            logits = self.output_soft_cap_temp * torch.tanh(
                logits / self.output_soft_cap_temp
            )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def interleaved_weights() -> Iterable[tuple[str, torch.Tensor]]:
            config = self.model.config
            qk_rows = (
                config.num_attention_heads * config.head_dim,
                config.num_key_value_heads * config.head_dim,
            )
            for name, weight in weights:
                if (
                    name.startswith(
                        ("model.language_model.layers.", "model.layers.", "layers.")
                    )
                    and (".self_attn.q_proj." in name or ".self_attn.k_proj." in name)
                    and name.rsplit(".", 1)[-1].startswith("weight")
                    and weight.dim() >= 1
                    and weight.shape[0] in qk_rows
                ):
                    # HF stores Q/K rows as [even | odd] per head (NoPE too);
                    # restore interleaved pairs before TP slicing. Per-row
                    # scales move with their rows; 0-d scales stay.
                    shape = weight.shape
                    weight = (
                        weight.reshape(-1, 2, config.head_dim // 2, *shape[1:])
                        .transpose(1, 2)
                        .reshape(shape)
                    )
                yield name, weight

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(
            interleaved_weights(), mapper=self.hf_to_vllm_mapper
        )
        n_gates = materialize_residual_path_constants(self)
        logger.info(
            "Materialized residual-path constants for %d gated residual connections",
            n_gates,
        )
        return loaded


__all__ = [
    "MuseSpark1xAttention",
    "MuseSpark1xDecoderLayer",
    "MuseSpark1xForCausalLM",
    "MuseSpark1xMoE",
    "MuseSpark1xModel",
    "MuseSparkTorchLinearMethod",
    "norm_residual_then_norm",
]
