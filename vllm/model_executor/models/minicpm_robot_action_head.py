# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn

DiTConfig = {
    "DiT-B": {
        "input_embedding_dim": 768,
        "attention_head_dim": 64,
        "num_attention_heads": 12,
    },
    "DiT-M": {
        "input_embedding_dim": 1024,
        "attention_head_dim": 64,
        "num_attention_heads": 16,
    },
    "DiT-L": {
        "input_embedding_dim": 1536,
        "attention_head_dim": 48,
        "num_attention_heads": 32,
    },
}


def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape (B, T, dim)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,).")
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, "
                "`num_positional_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        final_dropout: bool = True,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )
        self.gradient_checkpointing = False

        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim)

        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = (
                cross_attention_dim if not use_self_attn else None
            )
            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor | None = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class GR00tActionHead(nn.Module):
    """GR00T-style flow-matching action head (inference only).

    The DiT cross-attends to the VLM's last hidden state.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        action_model_type: str = "DiT-B",
        action_dim: int = 7,
        state_dim: int = 8,
        num_inference_timesteps: int = 4,
        num_target_vision_tokens: int = 32,
        max_seq_len: int = 1024,
        num_timestep_buckets: int = 1000,
        add_pos_embed: bool = True,
        action_horizon: int = 8,
        proprio_inject: str = "state_token",
        prediction_type: str = "velocity",
        diffusion_model_cfg: dict | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        action_model_cfg = DiTConfig[action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        default_diffusion_cfg = {
            **action_model_cfg,
            "cross_attention_dim": 1024,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        }
        if diffusion_model_cfg is not None:
            default_diffusion_cfg.update(diffusion_model_cfg)
        self.model = DiT(**default_diffusion_cfg)
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps

        if proprio_inject not in {"state_token", "concat"}:
            raise ValueError(f"Unsupported proprio_inject: {proprio_inject}")
        self.proprio_inject = proprio_inject
        self.state_dim = state_dim

        if proprio_inject == "concat":
            if not state_dim:
                raise ValueError("proprio_inject='concat' requires state_dim > 0")
            self.state_encoder = None
            encoder_action_dim = action_dim + state_dim
        else:
            encoder_action_dim = action_dim
            self.state_encoder = (
                MLP(
                    input_dim=state_dim,
                    hidden_dim=self.hidden_size,
                    output_dim=self.input_embedding_dim,
                )
                if state_dim
                else None
            )

        self.action_encoder = ActionEncoder(
            action_dim=encoder_action_dim, hidden_size=self.input_embedding_dim
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(
            num_target_vision_tokens, self.input_embedding_dim
        )
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if add_pos_embed:
            self.position_embedding = nn.Embedding(
                max_seq_len, self.input_embedding_dim
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.num_timestep_buckets = num_timestep_buckets
        self.add_pos_embed = add_pos_embed
        self.prediction_type = prediction_type

        if self.prediction_type not in {"velocity", "clean_action"}:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

    def _encode_action_tokens(
        self,
        noisy_trajectory: torch.Tensor,
        state: torch.Tensor | None,
        t_discretized: torch.Tensor,
    ) -> torch.Tensor:
        if self.proprio_inject == "concat":
            proprio = state.expand(-1, noisy_trajectory.shape[1], -1)
            noisy_trajectory = torch.cat([noisy_trajectory, proprio], dim=-1)
        return self.action_encoder(noisy_trajectory, t_discretized)

    def _build_sequence(
        self,
        action_features: torch.Tensor,
        state_features: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if state_features is not None:
            return torch.cat((state_features, future_tokens, action_features), dim=1)
        return torch.cat((future_tokens, action_features), dim=1)

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        noise = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        num_steps = self.num_inference_timesteps

        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state)
        else:
            state_features = None

        def _predict_from_noisy_actions(
            noisy_actions: torch.Tensor, t_discretized: int
        ) -> torch.Tensor:
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self._encode_action_tokens(
                noisy_actions, state, timesteps_tensor
            )
            if self.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = self._build_sequence(
                action_features, state_features, vl_embs.shape[0]
            )
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            return pred[:, -self.action_horizon :]

        if self.prediction_type == "velocity":
            actions = noise
            dt = 1.0 / num_steps
            for t in range(num_steps):
                t_cont = t / float(num_steps)
                t_discretized = int(t_cont * self.num_timestep_buckets)
                pred_velocity = _predict_from_noisy_actions(actions, t_discretized)
                actions = actions + dt * pred_velocity
            return actions

        actions = torch.zeros_like(noise)
        for step in range(num_steps, 0, -1):
            t_cont = step / float(num_steps)
            t_discretized = min(
                int(t_cont * self.num_timestep_buckets),
                self.num_timestep_buckets - 1,
            )
            noisy_trajectory = t_cont * noise + (1.0 - t_cont) * actions
            actions = _predict_from_noisy_actions(noisy_trajectory, t_discretized)
        return actions
