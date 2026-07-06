# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GigaChat 3.5 model."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import islice

import regex as re
import torch
import torch.nn.functional as F
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.gigachat3_5 import GigaChat35Config

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

_EXTRA_LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")


def _is_extra_decoder_layer(name: str, num_hidden_layers: int) -> bool:
    match = _EXTRA_LAYER_RE.match(name)
    return match is not None and int(match.group(1)) >= num_hidden_layers


def _apply_sigmoid_gate_fp32(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_mla_scaling_factors(config: GigaChat35Config) -> tuple[float, float]:
    if not getattr(config, "use_mla_scaling_factor", False):
        return 1.0, 1.0
    q_hidden_dim = (
        config.hidden_size if config.q_lora_rank is None else config.q_lora_rank
    )
    kv_hidden_dim = (
        config.hidden_size if config.kv_lora_rank is None else config.kv_lora_rank
    )
    return (
        math.sqrt(config.hidden_size / q_hidden_dim),
        math.sqrt(config.hidden_size / kv_hidden_dim),
    )


def _get_moe_scoring_func(config: GigaChat35Config) -> str:
    return config.scoring_func


def _get_gdn_gqa_interleaved_layout(config: GigaChat35Config) -> bool:
    # HF GigaChat stores GDN q/k/v/z and b/a grouped by key-head group.
    # QwenGatedDeltaNetAttention preserves that format via this layout flag.
    return not getattr(config, "linear_use_legacy_qkvz_layout", False)


class GigaChat35ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        # Output scale (MLA alpha_q/alpha_kv). Applied to the fp32 norm output
        # so it works for fp8 checkpoints too (cannot fold a scalar into fp8
        # weights). Base default 1.0; gated subclass uses its own _gate_scale.
        self._out_scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        if self._out_scale != 1.0:
            x = x * self._out_scale
        return x.to(dtype)


class GigaChat35ZeroCenteredGatedNorm(GigaChat35ZeroCenteredRMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(hidden_size, eps=eps)  # base _out_scale stays 1.0
        self.layernorm_gating_weight = layernorm_gating_weight
        self._gate_scale = scale
        self.gate_up_projection = nn.Linear(hidden_size, 16, bias=False)
        self.gate_down_projection = nn.Linear(16, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states)
        gate_hidden = F.linear(
            hidden_states.float(),
            self.gate_up_projection.weight.float(),
        )
        gate_hidden = F.silu(gate_hidden)
        gate_hidden = F.linear(
            gate_hidden,
            self.gate_down_projection.weight.float(),
        )
        output = (
            hidden_states.float()
            * self.layernorm_gating_weight
            * torch.sigmoid(gate_hidden.float())
        )
        if self._gate_scale != 1.0:
            output = output * self._gate_scale
        return output.to(hidden_states.dtype)


class GigaChat35ScaledRMSNorm(RMSNorm):
    """Plain RMSNorm whose output is scaled by ``scale`` (MLA alpha), keeping the
    ``.weight`` parameter name so checkpoints load unchanged."""

    def __init__(self, hidden_size: int, eps: float, scale: float) -> None:
        super().__init__(hidden_size, eps=eps)
        self._out_scale = scale

    def forward(self, x, residual=None):
        out = super().forward(x, residual)
        if isinstance(out, tuple):
            return out[0] * self._out_scale, out[1]
        return out * self._out_scale


def _build_norm(
    config: GigaChat35Config,
    hidden_size: int,
    scale: float = 1.0,
) -> nn.Module:
    norm_type = getattr(config, "norm_type", "ZeroCenteredGatedNorm")
    if norm_type in ("LlamaRMSNorm", "RMSNorm"):
        if scale != 1.0:
            return GigaChat35ScaledRMSNorm(hidden_size, config.rms_norm_eps, scale)
        return RMSNorm(hidden_size, eps=config.rms_norm_eps)
    if norm_type == "ZeroCenteredRMSNorm":
        return GigaChat35ZeroCenteredRMSNorm(
            hidden_size, eps=config.rms_norm_eps, scale=scale
        )
    if norm_type == "ZeroCenteredGatedNorm":
        return GigaChat35ZeroCenteredGatedNorm(
            hidden_size,
            eps=config.rms_norm_eps,
            layernorm_gating_weight=config.layernorm_gating_weight,
            scale=scale,
        )
    raise ValueError(f"Unsupported GigaChat 3.5 norm type: {norm_type!r}")


class GigaChat35GatedDeltaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str = "silu",
        zero_centered: bool = False,
    ) -> None:
        super().__init__()
        if activation not in ("silu", "sigmoid"):
            raise ValueError(f"Unsupported GDN gate activation: {activation!r}")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation
        self.zero_centered = zero_centered

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight = self.weight.float()
        if self.zero_centered:
            weight = 1.0 + weight
        hidden_states = hidden_states * weight
        if gate is not None:
            gate = gate.float()
            gate = torch.sigmoid(gate) if self.activation == "sigmoid" else F.silu(gate)
            hidden_states = hidden_states * gate
        return hidden_states.to(dtype)


class GigaChat35GatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def __init__(
        self,
        config: GigaChat35Config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            vllm_config=vllm_config,
            prefix=prefix,
            gqa_interleaved_layout=_get_gdn_gqa_interleaved_layout(config),
        )
        self.linear_gating_type = getattr(config, "linear_gating_type", "gated_rmsnorm")
        self.linear_sigmoid_gate_scale = getattr(
            config, "linear_sigmoid_gate_scale", 1.0
        )
        eps = config.linear_attn_o_norm_eps or config.rms_norm_eps

        if self.linear_gating_type == "gated_rmsnorm":
            self.norm = GigaChat35GatedDeltaRMSNorm(
                self.head_v_dim, eps=eps, activation="silu"
            )
        elif self.linear_gating_type == "gated_rmsnorm_sigmoid":
            self.norm = GigaChat35GatedDeltaRMSNorm(
                self.head_v_dim, eps=eps, activation="sigmoid"
            )
        elif self.linear_gating_type == "gated_rmsnorm_sigmoid_zero_centered":
            self.norm = GigaChat35GatedDeltaRMSNorm(
                self.head_v_dim,
                eps=eps,
                activation="sigmoid",
                zero_centered=True,
            )
        elif self.linear_gating_type == "sigmoid_gate":
            self.norm = None
        else:
            raise ValueError(
                f"Unsupported GigaChat 3.5 linear_gating_type: "
                f"{self.linear_gating_type!r}"
            )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        # ba_proj can't be block-wise fp8 quantized (output dim not
        # block-aligned), so keep it unquantized under block-fp8, like conv1d.
        if (
            isinstance(quant_config, Fp8Config)
            and quant_config.weight_block_size is not None
        ):
            quant_config = None
        return super().create_ba_proj(
            hidden_size=hidden_size,
            num_v_heads=num_v_heads,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor,
        num_tokens: int,
    ):
        z_shape = z.shape
        if self.linear_gating_type == "sigmoid_gate":
            core_attn_out = core_attn_out.reshape(num_tokens, -1)
            z = z.reshape(num_tokens, -1)
            core_attn_out = _apply_sigmoid_gate_fp32(
                core_attn_out, z * self.linear_sigmoid_gate_scale
            )
        else:
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            assert self.norm is not None
            core_attn_out = self.norm(core_attn_out, z)
            if self.linear_gating_type in (
                "gated_rmsnorm_sigmoid",
                "gated_rmsnorm_sigmoid_zero_centered",
            ):
                core_attn_out = core_attn_out * self.linear_sigmoid_gate_scale
            core_attn_out = core_attn_out.reshape(z_shape).flatten(-2)
        output[:num_tokens], _ = self.out_proj(core_attn_out)


class _GigaChat35PassthroughOProj(nn.Module):
    """Identity ``o_proj`` handed to the MLA wrapper.

    ``MultiHeadLatentAttentionWrapper.forward`` returns ``o_proj(attn_out)[0]``.
    Passing this stub makes the wrapper return the raw pre-``o_proj`` attention
    output so GigaChat can apply its sigmoid gated-attention before running the
    real ``o_proj``.
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x, None


class GigaChat35Attention(nn.Module):
    """DeepSeek-style MLA (compressed latent KV cache + weight absorption) with
    GigaChat's ``alpha_q``/``alpha_kv`` scaling applied at the q_a/kv_a layernorm
    output and an optional sigmoid gated-attention before ``o_proj``."""

    def __init__(
        self,
        config: GigaChat35Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.total_num_heads = config.num_attention_heads
        # alpha_q / alpha_kv are applied at the q_a/kv_a layernorm output (see
        # _build_norm scale=...), not folded into weights, so it also works for
        # fp8 checkpoints (a scalar cannot be folded into fp8 weights).
        self.alpha_q, self.alpha_kv = _get_mla_scaling_factors(config)
        tp_size = get_tensor_model_parallel_world_size()
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} must be divisible "
                f"by tensor_parallel_size={tp_size}"
            )
        self.num_local_heads = self.total_num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
            self.q_a_layernorm = _build_norm(
                config,
                self.q_lora_rank,
                scale=self.alpha_q,
            )
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )

        self.kv_a_layernorm = _build_norm(
            config,
            self.kv_lora_rank,
            scale=self.alpha_kv,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.gated_attention = getattr(config, "gated_attention", False)
        if self.gated_attention:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )

        rope_parameters = dict(config.rope_parameters)
        rope_type = rope_parameters.get("rope_type", "default")
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=not getattr(config, "rope_interleave", True),
        )

        if rope_type == "yarn":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", False)
            if mscale_all_dim:
                mscale = _yarn_get_mscale(
                    float(rope_parameters["factor"]), float(mscale_all_dim)
                )
                self.scaling *= mscale * mscale

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=_GigaChat35PassthroughOProj(),
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.mla_attn(positions, hidden_states)
        if self.gated_attention:
            gate, _ = self.gate_proj(hidden_states)
            attn_output = _apply_sigmoid_gate_fp32(attn_output, gate)
        output, _ = self.o_proj(attn_output)
        return output


class GigaChat35MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        swiglu_limit: float = 0.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act!r}")
        if swiglu_limit and swiglu_limit > 0:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GigaChat35MoE(nn.Module):
    def __init__(
        self,
        config: GigaChat35Config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act!r}")

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.is_fusion_moe_shared_experts_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        if (
            self.is_rocm_aiter_moe_enabled
            and self.gate.e_score_correction_bias is not None
        ):
            self.gate.set_out_dtype(self.gate.weight.dtype)

        if self.n_shared_experts == 0 or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None
        else:
            self.shared_experts = GigaChat35MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * self.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                is_sequence_parallel=self.is_sequence_parallel,
                swiglu_limit=config.swiglu_limit,
                prefix=f"{prefix}.shared_experts",
            )

        self.shared_expert_gate = None
        if getattr(config, "use_shared_expert_sigmoid", False):
            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.shared_expert_gate",
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=_get_moe_scoring_func(config),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            swiglu_limit=config.swiglu_limit,
            apply_routed_scale_to_output=not self.is_rocm_aiter_moe_enabled,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=self.n_shared_experts
            if self.is_fusion_moe_shared_experts_enabled
            else None,
            shared_expert_gate=self.shared_expert_gate,
            router_logits_dtype=self.gate.out_dtype,
        )

        if (
            self.is_rocm_aiter_moe_enabled
            and self.gate.e_score_correction_bias is not None
        ):
            self.gate.e_score_correction_bias.data = (
                self.gate.e_score_correction_bias.data.to(self.gate.out_dtype)
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            output = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            output = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        if self.is_sequence_parallel:
            output = tensor_model_parallel_all_gather(output, 0)
            output = output[:num_tokens]

        return output.view(orig_shape)


class GigaChat35DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        *,
        layer_type: str | None = None,
        is_nextn: bool = False,
        config: GigaChat35Config | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.layer_idx = extract_layer_index(prefix)
        if layer_type is not None:
            self.layer_type = layer_type
        elif self.layer_idx < len(config.layer_types):
            self.layer_type = config.layer_types[self.layer_idx]
        else:
            self.layer_type = "full_attention"
        self.is_nextn = is_nextn
        self.use_pre_layernorm = config._use_pre_layernorm
        self.use_post_layernorm = config._use_post_layernorm

        if self.layer_type == "linear_attention":
            self.self_attn = GigaChat35GatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = GigaChat35Attention(
                config=config,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid GigaChat 3.5 layer_type: {self.layer_type!r}")

        if is_nextn:
            use_moe = bool(getattr(config, "nextn_is_sparse", False))
        else:
            use_moe = self.layer_idx >= config.first_k_dense_replace
        if not use_moe:
            self.mlp = GigaChat35MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                swiglu_limit=config.swiglu_limit,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = GigaChat35MoE(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )

        if self.use_pre_layernorm:
            self.input_layernorm = _build_norm(config, config.hidden_size)
            self.post_attention_layernorm = _build_norm(config, config.hidden_size)
        if self.use_post_layernorm:
            self.post_self_attn_layernorm = _build_norm(config, config.hidden_size)
            self.post_feedforward_layernorm = _build_norm(config, config.hidden_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.use_pre_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            attn_output = torch.empty_like(hidden_states)
            self.self_attn(hidden_states=hidden_states, output=attn_output)
            hidden_states = attn_output
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )

        if self.use_post_layernorm:
            hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.use_pre_layernorm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.use_post_layernorm:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class GigaChat35Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: GigaChat35Config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        eplb_config = vllm_config.parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.vocab_size = config.vocab_size
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str) -> GigaChat35DecoderLayer:
            return GigaChat35DecoderLayer(vllm_config, prefix=prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = _build_norm(config, config.hidden_size)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to GigaChat35Model.forward"
                    )
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions=positions, hidden_states=hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return self.norm(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        num_experts = self.config.n_routed_experts
        if rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
            num_experts += self.config.n_shared_experts or 0
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        expert_params_mapping = self.get_expert_mapping()
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        is_fse = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        num_routed = self.config.n_routed_experts

        # Extra decoder layers (MTP/NextN) and ``mtp.*`` weights are already
        # filtered by GigaChat35ForCausalLM.load_weights before delegating here.
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            if is_fse and "mlp.shared_experts." in name:
                name = name.replace(
                    "mlp.shared_experts.",
                    f"mlp.experts.{num_routed}.",
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts." in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped_name, self):
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    mapped_name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(mapped_name, self):
                        continue
                    if (
                        mapped_name.endswith(".bias") or mapped_name.endswith("_bias")
                    ) and mapped_name not in params_dict:
                        continue
                    if mapped_name not in params_dict:
                        continue
                    param = params_dict[mapped_name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    name = mapped_name
                    break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            "Parameter %s not found in params_dict, skip loading",
                            name,
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class GigaChat35MixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, GigaChat35DecoderLayer) and isinstance(
                layer.mlp, GigaChat35MoE
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            logger.warning("No GigaChat35MoE layer found in model.layers.")
            return

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class GigaChat35ForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    GigaChat35MixtureOfExperts,
    IsHybrid,
):
    # ``in_proj_qkvz`` / ``in_proj_ba`` are the fused GDN input projections.
    # They map to themselves (a single checkpoint tensor, not split across
    # shards), mirroring the sibling Qwen3-Next GDN model
    # (``qwen3_next.py``) which registers the same entries for packed-module
    # bookkeeping (e.g. LoRA).
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: GigaChat35Config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "GigaChat 3.5 does not support 'all' mamba prefix caching. "
                "Use '--mamba-cache-mode=align'."
            )
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.scheduler_config = vllm_config.scheduler_config

        self.model = GigaChat35Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered_weights = (
            (name, weight)
            for name, weight in weights
            if not _is_extra_decoder_layer(name, self.config.num_hidden_layers)
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
            ignore_unexpected_prefixes=["model.layers."],
        )
        return loader.load_weights(filtered_weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
