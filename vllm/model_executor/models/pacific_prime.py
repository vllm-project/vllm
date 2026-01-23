# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

# =============================================================================
# INL Dynamics
# =============================================================================


class INLDynamics(nn.Module):
    """
    INL (Inertial Navigation Layer) Dynamics for numerical stability.

    Implements PID-like control with velocity tracking:
        mu(h) = mu_base + mu_proj(h)
        error = h - mu(h)
        v_next = alpha * v - beta * error
        h_next = h + dt * gate * v_next

    Parameters alpha, beta, gate are learned via controller MLP
    and clamped to [0, 1] via sigmoid for training stability.
    """

    def __init__(
        self,
        hidden_size: int,
        controller_hidden: int = 64,
        dt: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        # Learnable equilibrium
        self.mu = nn.Parameter(torch.zeros(hidden_size))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

        # Controller MLP
        self.controller_in = nn.Linear(hidden_size * 2, controller_hidden)
        self.controller_out = nn.Linear(controller_hidden, hidden_size * 3)

        with torch.no_grad():
            bias = self.controller_out.bias
            bias[:hidden_size].fill_(2.2)
            bias[hidden_size : hidden_size * 2].fill_(-2.2)
            bias[hidden_size * 2 :].fill_(0.0)
            self.controller_out.weight.normal_(0, 0.01)

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if v is None:
            v = torch.zeros_like(h)

        hv = torch.cat([h, v], dim=-1)
        ctrl = F.silu(self.controller_in(hv))
        ctrl_out = self.controller_out(ctrl)

        alpha_raw, beta_raw, gate_raw = torch.split(ctrl_out, self.hidden_size, dim=-1)
        alpha = torch.sigmoid(alpha_raw)
        beta = torch.clamp(F.softplus(beta_raw), max=2.0)
        gate = torch.sigmoid(gate_raw)

        mu_contextual = self.mu + self.mu_proj(h)
        error = h - mu_contextual
        v_next = alpha * v - beta * error
        v_next = torch.clamp(v_next, min=-10.0, max=10.0)
        h_next = h + self.dt * gate * v_next

        return h_next, v_next, mu_contextual


# =============================================================================
# Token-Routed MLP
# =============================================================================


class TokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP - Deterministic expert routing.
    Routes tokens to experts based on: expert_id = token_id % num_experts
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        vocab_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.expert_intermediate_size = intermediate_size // num_experts

        # Expert weights
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * self.expert_intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, self.expert_intermediate_size, hidden_size)
        )

        # Mu-guided routing
        self.mu_router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.mu_router.weight)

        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
        )

        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj, a=5**0.5)

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]

        if token_ids is None:
            expert_ids = torch.zeros(num_tokens, dtype=torch.long, device=x.device)
        else:
            token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
            base_expert_ids = self.token_to_expert.to(x.device)[token_ids_clamped]

            if mu is not None:
                mu_logits = self.mu_router(mu)
                base_one_hot = F.one_hot(base_expert_ids, self.num_experts).float()
                combined_logits = base_one_hot * 10.0 + mu_logits
                expert_ids = combined_logits.argmax(dim=-1)
            else:
                expert_ids = base_expert_ids

        # Memory-efficient: process by expert groups instead of per-token bmm
        output = torch.zeros(
            num_tokens, self.hidden_size, device=x.device, dtype=x.dtype
        )

        for expert_id in range(self.num_experts):
            # Find tokens routed to this expert
            mask = expert_ids == expert_id
            if not mask.any():
                continue

            # Get tokens for this expert
            x_expert = x[mask]  # [num_tokens_for_expert, hidden_size]

            # Get expert weights
            gate_up_w = self.gate_up_proj[expert_id]  # [hidden_size, 2*intermediate]
            down_w = self.down_proj[expert_id]  # [intermediate, hidden_size]

            # Forward through expert (matmul instead of bmm)
            gate_up_out = x_expert @ gate_up_w  # [n, 2*intermediate]
            gate_out = gate_up_out[..., : self.expert_intermediate_size]
            up_out = gate_up_out[..., self.expert_intermediate_size :]
            intermediate = F.silu(gate_out) * up_out  # [n, intermediate]
            expert_output = intermediate @ down_w  # [n, hidden_size]

            # Scatter back to output
            output[mask] = expert_output

        return output


# =============================================================================
# Standard MLP (fallback)
# =============================================================================


class ComplexityMLP(nn.Module):
    """Standard MLP with SiLU activation."""

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

        # Mu-guided projections
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

        # RoPE - vLLM v1 API
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

        # Attention
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

        # Mu-guidance
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

        # Attention
        attn_output = self.attn(q, k, v)

        # Output
        output, _ = self.o_proj(attn_output)
        return output


# =============================================================================
# Decoder Layer
# =============================================================================


class ComplexityDecoderLayer(nn.Module):
    """Complexity decoder layer with INL dynamics."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_token_routed_mlp = getattr(config, "use_token_routed_mlp", True)

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

        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            controller_hidden=getattr(config, "dynamics_controller_hidden", 64),
            dt=getattr(config, "dynamics_dt", 0.1),
        )

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

        if self.use_token_routed_mlp:
            self.mlp = TokenRoutedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=getattr(config, "num_experts", 4),
                vocab_size=config.vocab_size,
                quant_config=quant_config,
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


class ComplexityModel(nn.Module):
    """Complexity transformer model."""

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

        # Handle prefix correctly (avoid leading dot when prefix is empty)
        embed_prefix = f"{prefix}.embed_tokens" if prefix else "embed_tokens"
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=embed_prefix,
        )

        self.layers = nn.ModuleList(
            [
                ComplexityDecoderLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}" if prefix else f"layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.norm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        velocity_states = torch.zeros_like(hidden_states)

        mu_prev = None
        mu_residual = None

        for layer in self.layers:
            hidden_states, velocity_states, mu_current = layer(
                positions=positions,
                hidden_states=hidden_states,
                velocity_states=velocity_states,
                token_ids=input_ids,
                mu_prev=mu_prev,
            )

            if mu_residual is None:
                mu_residual = mu_current.clone()
            else:
                mu_residual = mu_residual + mu_current
            mu_prev = mu_current + 0.1 * mu_residual

        hidden_states = self.norm(hidden_states)
        return hidden_states


# =============================================================================
# Causal LM
# =============================================================================


class ComplexityForCausalLM(nn.Module):
    """
    Complexity model for causal language modeling.
    Compatible with vLLM inference engine.
    """

    # No packed modules - Q/K/V are separate, MLP uses TokenRoutedMLP
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

        # Handle prefix correctly (avoid leading dot when prefix is empty)
        model_prefix = f"{prefix}.model" if prefix else "model"

        self.model = ComplexityModel(
            vllm_config=vllm_config,
            prefix=model_prefix,
        )

        if getattr(config, "tie_word_embeddings", True):
            self.lm_head = self.model.embed_tokens
        else:
            lm_head_prefix = f"{prefix}.lm_head" if prefix else "lm_head"
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=lm_head_prefix,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens - required for vLLM v1 generate runner."""
        return self.model.embed_tokens(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states - vLLM v1 API."""
        # vLLM v1 uses simpler logits computation
        if isinstance(self.lm_head, VocabParallelEmbedding):
            # Tied embeddings case
            logits = F.linear(hidden_states, self.lm_head.weight)
        else:
            logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.

        Pacific-Prime checkpoint structure mirrors model exactly:
        - lm_head.weight -> model.embed_tokens.weight (tied embeddings)
        - self_attn.q_proj, k_proj, v_proj (separate, not fused)
        - layers.*.mlp.gate_up_proj -> TokenRoutedMLP expert weights
        - layers.*.mlp.down_proj -> TokenRoutedMLP expert weights
        - layers.*.mlp.token_to_expert -> buffer (not a parameter)
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            orig_name = name

            # Skip rotary_emb.inv_freq - vLLM computes this
            if "rotary_emb.inv_freq" in name:
                loaded_params.add(orig_name)
                continue

            # Handle tied embeddings: both lm_head.weight and model.embed_tokens.weight
            # map to the same parameter (model.embed_tokens.weight)
            if name == "lm_head.weight":
                embed_name = "model.embed_tokens.weight"
                if embed_name in params_dict:
                    param = params_dict[embed_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(orig_name)
                    # Mark embed_tokens as loaded too (tied embeddings)
                    loaded_params.add(embed_name)
                continue

            # Skip model.embed_tokens.weight if already loaded via lm_head.weight (tied)
            if name == "model.embed_tokens.weight":
                if name in loaded_params:
                    continue
                # If not yet loaded, load it directly
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(orig_name)
                continue

            # Handle token_to_expert buffer
            if "token_to_expert" in name:
                if name in buffers_dict:
                    buffers_dict[name].copy_(loaded_weight)
                    loaded_params.add(orig_name)
                continue

            # Handle TokenRoutedMLP weights (expert weights [num_experts, ...])
            if ".mlp.gate_up_proj" in name or ".mlp.down_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    with torch.no_grad():
                        param.copy_(loaded_weight)
                    loaded_params.add(orig_name)
                continue

            # Direct parameter loading (q_proj, k_proj, v_proj, o_proj, norms, etc.)
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(orig_name)

        return loaded_params


# Alias for HuggingFace compatibility
DeepForCausalLM = ComplexityForCausalLM
