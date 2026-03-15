# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Pacific-Prime / Complexity model for vLLM inference.

A decoder-only transformer with INL (Inertial Navigation Layer) dynamics
for numerical stability and smooth token generation.

Key innovations:
- INL Dynamics: PID-like control with velocity tracking (alpha, beta, gate, mu)
- Token-Routed MLP: Deterministic expert routing (token_id % num_experts)
- Mu-Guided Attention: Top-down influence from previous layer's equilibrium

GitHub: https://github.com/Complexity-ML/complexity-deep
HuggingFace: https://huggingface.co/Pacific-Prime/pacific-prime
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import ignore_torch_compile, support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.token_routed_i64 import (
    INLDynamics,
    TokenRoutedMLP,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# =============================================================================
# Standard MLP (fallback when token-routed is disabled)
# =============================================================================


class ComplexityMLP(nn.Module):
    """Standard SwiGLU MLP using vLLM parallel layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# =============================================================================
# Attention
# =============================================================================


class ComplexityAttention(nn.Module):
    """
    Complexity attention with mu-guidance and GQA support.

    Mu-guidance: the mu vector from INL Dynamics biases Q/K/V
    projections, providing top-down control from previous layers.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Separate Q/K/V projections (mirrors checkpoint structure)
        self.q_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        # Mu-guided projections (output matches TP-sharded Q/K/V sizes)
        self.mu_to_q = nn.Linear(
            hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.mu_to_k = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.mu_to_v = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        for proj in [self.mu_to_q, self.mu_to_k, self.mu_to_v]:
            nn.init.normal_(proj.weight, std=0.01)

        # Output projection
        self.o_proj = RowParallelLinear(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters={"base": rope_theta},
        )

        # QK Norm
        self.use_qk_norm = getattr(config, "use_qk_norm", True)
        if self.use_qk_norm:
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Attention (uses vLLM PagedAttention / FlashAttention)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mu_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # Mu-guidance: bias Q/K/V with previous layer's equilibrium
        if mu_prev is not None:
            q = q + self.mu_to_q(mu_prev)
            k = k + self.mu_to_k(mu_prev)
            v = v + self.mu_to_v(mu_prev)

        # QK Norm
        if self.use_qk_norm:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(-1, self.q_size)
            k = k.view(-1, self.kv_size)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Attention (PagedAttention via vLLM)
        attn_output = self.attn(q, k, v)

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output


# =============================================================================
# Decoder Layer
# =============================================================================


class ComplexityDecoderLayer(nn.Module):
    """Complexity decoder layer: Attention → INL Dynamics → MLP."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        # use_token_routed_mlp can be True, or config may have mlp_type="token_routed"
        _use_tr = getattr(config, "use_token_routed_mlp", None)
        _mlp_type = getattr(config, "mlp_type", None)
        self.use_token_routed_mlp = (
            _use_tr is True
            or _mlp_type == "token_routed"
            or (_use_tr is None and _mlp_type is None)  # default: True
        )

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

        self.self_attn = ComplexityAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            max_position_embeddings=getattr(config, "max_position_embeddings", 2048),
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )

        # Framework-trained models (mlp_type in config) use non-contextual error
        use_contextual_error = getattr(config, "mlp_type", None) is None
        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            controller_hidden=getattr(config, "dynamics_controller_hidden", 64),
            dt=getattr(config, "dynamics_dt", 0.1),
            use_contextual_error=use_contextual_error,
        )

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

        if self.use_token_routed_mlp:
            self.mlp = TokenRoutedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=getattr(config, "num_experts", 4),
                vocab_size=config.vocab_size,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = ComplexityMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        velocity_states: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        mu_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            mu_prev=mu_prev,
        )

        # INL Dynamics
        hidden_states, velocity_states, mu_current = self.dynamics(
            hidden_states, velocity_states
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_token_routed_mlp:
            hidden_states = self.mlp(hidden_states, token_ids=token_ids, mu=mu_current)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, velocity_states, mu_current


# =============================================================================
# Model
# =============================================================================


@ignore_torch_compile
@support_torch_compile
class ComplexityModel(nn.Module):
    """Complexity transformer model with INL dynamics threading."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: ComplexityDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=maybe_prefix(prefix, "layers"),
        )
        # Tag layers with index for debug
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "_layer_idx"):
                continue
            layer._layer_idx = i

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # Framework-trained models (mlp_type in config) do not cascade velocity
        self.cascade_velocity = getattr(config, "mlp_type", None) is None

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "velocity_states", "mu_prev"],
            config.hidden_size,
        )

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
                hidden_states = self.embed_tokens(input_ids)
            velocity_states = torch.zeros_like(hidden_states)
            mu_prev = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            velocity_states = intermediate_tensors["velocity_states"]
            mu_prev = intermediate_tensors.get("mu_prev")

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if isinstance(layer, PPMissingLayer):
                continue

            # Reset velocity each layer for framework-trained models
            layer_v = (
                velocity_states
                if self.cascade_velocity
                else torch.zeros_like(hidden_states)
            )

            hidden_states, velocity_states, mu_current = layer(
                positions=positions,
                hidden_states=hidden_states,
                velocity_states=layer_v,
                token_ids=input_ids,
                mu_prev=mu_prev,
            )

            if not self.cascade_velocity:
                velocity_states = torch.zeros_like(hidden_states)

            # Match i64 engine decode_step: simple mu pass-through
            if mu_current is not None:
                mu_prev = mu_current

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "velocity_states": velocity_states,
                    "mu_prev": mu_prev,
                }
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


# =============================================================================
# Causal LM
# =============================================================================


class ComplexityForCausalLM(nn.Module, SupportsPP):
    """
    Complexity model for causal language modeling.

    Compatible with vLLM inference engine. Uses:
    - LogitsProcessor for correct TP logits gathering
    - Pipeline Parallelism via SupportsPP interface
    - AutoWeightsLoader for standard weight loading
    """

    # No packed modules — Q/K/V are separate, MLP uses TokenRoutedMLP
    packed_modules_mapping = {}

    supported_lora_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = ComplexityModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            if getattr(config, "tie_word_embeddings", True):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.

        Handles:
        - No "model." prefix in checkpoint (saved from inner module)
        - Separate experts.E.{gate,up,down}_proj → merged gate_up_proj / down_proj
        - dynamics.controller.0/2 → controller_in / controller_out
        - Tied embeddings
        - mu_router (zero-init, not in checkpoint)
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # mu_router is zero-init and intentionally absent from checkpoint
        for pname in params_dict:
            if ".mlp.mu_router." in pname:
                loaded_params.add(pname)

        # Buffer per-expert weights: {layer_idx: {expert_idx: {proj: tensor}}}
        expert_buf: dict = {}

        for ckpt_name, loaded_weight in weights:
            # --- Normalize checkpoint key to model parameter name ---
            name = ckpt_name
            if not name.startswith("model.") and name != "lm_head.weight":
                name = "model." + name
            name = name.replace(".dynamics.controller.0.", ".dynamics.controller_in.")
            name = name.replace(".dynamics.controller.2.", ".dynamics.controller_out.")

            # Skip rotary_emb.inv_freq — vLLM recomputes it
            if "rotary_emb.inv_freq" in name:
                continue

            # Skip token_to_expert — buffer, not parameter
            if "token_to_expert" in name:
                continue

            # Tied embeddings: lm_head.weight → embed_tokens
            if ckpt_name == "lm_head.weight":
                if getattr(self.config, "tie_word_embeddings", True):
                    embed_name = "model.embed_tokens.weight"
                    if embed_name in params_dict:
                        param = params_dict[embed_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(embed_name)
                else:
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                continue

            # Skip embed_tokens if already loaded via lm_head tie
            if name == "model.embed_tokens.weight" and name in loaded_params:
                continue

            # Expert weights — buffer for merging
            # Pattern: model.layers.X.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight
            if ".mlp.experts." in name:
                layer_idx = int(name.split(".layers.")[1].split(".")[0])
                after = name.split(".mlp.experts.")[1]  # "E.gate_proj.weight"
                parts = after.split(".")
                expert_idx = int(parts[0])
                proj = parts[1]  # gate_proj | up_proj | down_proj
                expert_buf.setdefault(layer_idx, {}).setdefault(expert_idx, {})[
                    proj
                ] = loaded_weight.clone()
                continue

            # Standard parameter loading
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # --- Merge buffered expert weights into gate_up_proj / down_proj ---
        for layer_idx, experts in expert_buf.items():
            mlp = self.model.layers[layer_idx].mlp
            if not isinstance(mlp, TokenRoutedMLP):
                continue

            num_e = mlp.local_num_experts
            sample = next(iter(experts.values()))
            full_inter = sample["gate_proj"].shape[0]  # [out_feat, in_feat]
            hidden = sample["gate_proj"].shape[1]
            dtype = sample["gate_proj"].dtype

            # gate_up_proj: [E, hidden, 2*full_inter]
            gate_up_full = torch.zeros(num_e, hidden, 2 * full_inter, dtype=dtype)
            # down_proj:    [E, full_inter, hidden]
            down_full = torch.zeros(num_e, full_inter, hidden, dtype=dtype)

            for e_idx, e_w in experts.items():
                # Linear stores [out, in] → transpose for BMM [in, out]
                gate_up_full[e_idx, :, :full_inter] = e_w["gate_proj"].T
                gate_up_full[e_idx, :, full_inter:] = e_w["up_proj"].T
                down_full[e_idx] = e_w["down_proj"].T

            gu_name = f"model.layers.{layer_idx}.mlp.gate_up_proj"
            dn_name = f"model.layers.{layer_idx}.mlp.down_proj"
            if gu_name in params_dict:
                mlp.load_tp_weight("gate_up_proj", params_dict[gu_name], gate_up_full)
                loaded_params.add(gu_name)
            if dn_name in params_dict:
                mlp.load_tp_weight("down_proj", params_dict[dn_name], down_full)
                loaded_params.add(dn_name)

        return loaded_params


# HuggingFace compatibility alias
DeepForCausalLM = ComplexityForCausalLM
