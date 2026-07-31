# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Kimi attention layers with explicit PyTorch math."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .layers import RMSNorm


def _load_a_log(parameter: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load either the old ``[1, 1, H, 1]`` or current ``[H]`` layout."""

    if loaded_weight.ndim == 4:
        loaded_weight = loaded_weight.flatten()
    rank = get_tensor_model_parallel_rank()
    shard_size = parameter.shape[0]
    loaded_weight = loaded_weight.narrow(0, rank * shard_size, shard_size)
    default_weight_loader(parameter, loaded_weight)


class MultiHeadLatentAttention(Attention):
    """NoPE MLA expanded to ordinary K/V and vLLM's registered attention."""

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        if not config.mla_use_nope:
            raise ValueError("The portable Kimi model supports NoPE MLA only")
        required = {
            "kv_lora_rank": config.kv_lora_rank,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
        }
        if any(value is None for value in required.values()):
            raise ValueError(f"Incomplete Kimi K3 MLA config: {required}")
        assert config.kv_lora_rank is not None
        assert config.qk_nope_head_dim is not None
        assert config.qk_rope_head_dim is not None
        assert config.v_head_dim is not None

        tp_size = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by TP size")
        num_heads = config.num_attention_heads // tp_size
        qk_head_dim = int(config.qk_nope_head_dim) + int(config.qk_rope_head_dim)
        quant_config = vllm_config.quant_config
        super().__init__(
            num_heads,
            qk_head_dim,
            qk_head_dim**-0.5,
            num_kv_heads=num_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = qk_head_dim

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        else:
            self.q_a_proj = ReplicatedLinear(
                config.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                config.num_attention_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = (
            ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            if config.mla_use_output_gate
            else None
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del positions
        if self.q_lora_rank is None:
            query, _ = self.q_proj(hidden_states)
        else:
            query_latent, _ = self.q_a_proj(hidden_states)
            query, _ = self.q_b_proj(self.q_a_layernorm(query_latent))

        compressed_kv, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, shared_key = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        key_value, _ = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
        key_value = key_value.view(
            -1,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key, value = key_value.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        shared_key = shared_key[:, None, :].expand(-1, self.num_heads, -1)
        key = torch.cat((key, shared_key), dim=-1)

        value = F.pad(value, (0, self.qk_head_dim - self.v_head_dim))
        output = super().forward(query, key.flatten(1), value.flatten(1))
        output = output.view(-1, self.num_heads, self.qk_head_dim)
        output = output[..., : self.v_head_dim].flatten(1)
        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            output = output * torch.sigmoid(gate)
        output, _ = self.o_proj(output)
        return output


class CausalDepthwiseConv1d(nn.Module):
    """PyTorch short convolution with its previous-input state passed explicitly."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if channels % tp_size:
            raise ValueError("Convolution channels must be divisible by TP size")
        self.channels = channels // tp_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(self.channels, 1, kernel_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        if loaded_weight.ndim == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        rank = get_tensor_model_parallel_rank()
        loaded_weight = loaded_weight.chunk(
            get_tensor_model_parallel_world_size(), dim=0
        )[rank]
        parameter.data.copy_(loaded_weight)

    def forward(
        self,
        inputs: torch.Tensor,
        sconv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.cat((sconv_state, inputs.transpose(0, 1)), dim=-1)
        output = F.conv1d(
            sequence.unsqueeze(0),
            self.weight,
            groups=self.channels,
        )
        state_length = self.kernel_size - 1
        next_state = sequence[:, -state_length:] if state_length else sequence[:, :0]
        return F.silu(output.squeeze(0).transpose(0, 1)), next_state


class KimiDeltaAttention(nn.Module, MambaBase):
    """KDA registered with vLLM, with convolution and recurrence in PyTorch."""

    supports_dcp = False

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Portable KDA does not support speculative decode"
            )
        kda_config = config.linear_attn_config
        if kda_config is None:
            raise ValueError("KDA requires linear_attn_config")

        self.prefix = prefix
        self.vllm_config = vllm_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = int(kda_config["head_dim"])
        self.total_heads = int(kda_config["num_heads"])
        if self.total_heads % self.tp_size:
            raise ValueError("KDA num_heads must be divisible by TP size")
        self.num_heads = self.total_heads // self.tp_size
        self.projection_size = self.total_heads * self.head_dim
        self.local_projection_size = self.num_heads * self.head_dim
        self.conv_size = int(kda_config["short_conv_kernel_size"])
        self.gate_lower_bound = kda_config.get("gate_lower_bound")
        self.use_full_rank_gate = kda_config.get("use_full_rank_gate", False)
        quant_config = vllm_config.quant_config

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.q_conv1d = CausalDepthwiseConv1d(self.projection_size, self.conv_size)
        self.k_conv1d = CausalDepthwiseConv1d(self.projection_size, self.conv_size)
        self.v_conv1d = CausalDepthwiseConv1d(self.projection_size, self.conv_size)
        self.f_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.b_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )
        if self.use_full_rank_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        else:
            self.g_a_proj = ReplicatedLinear(
                config.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_a_proj",
            )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                self.projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_b_proj",
            )

        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": _load_a_log})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        self.o_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.projection_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.vllm_config.model_config.dtype,
            self.vllm_config.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.total_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=0,
        )

    def _gate(self, raw_gate: torch.Tensor) -> torch.Tensor:
        raw_gate = raw_gate.float() + self.dt_bias.view(self.num_heads, self.head_dim)
        decay = self.A_log.float().exp()[None, :, None]
        if self.gate_lower_bound is None:
            return -decay * F.softplus(raw_gate)
        return self.gate_lower_bound * torch.sigmoid(decay * raw_gate)

    def _recurrent_kda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query.float()
        query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
        query = query * self.head_dim**-0.5
        key = key.float()
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
        output_dtype = value.dtype
        value = value.float()
        state = state.float()
        outputs = []
        for token_idx in range(query.shape[0]):
            q_t = query[token_idx]
            k_t = key[token_idx]
            v_t = value[token_idx]
            state = state * gate[token_idx].exp().unsqueeze(-1)
            prediction = torch.einsum("hk,hkv->hv", k_t, state)
            delta = beta[token_idx, :, None] * (v_t - prediction)
            state = state + torch.einsum("hk,hv->hkv", k_t, delta)
            outputs.append(torch.einsum("hk,hkv->hv", q_t, state))
        return torch.stack(outputs).to(output_dtype), state

    def _run_sequence(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        sconv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_state, k_state, v_state = sconv_state.chunk(3, dim=0)
        query, q_state = self.q_conv1d(query, q_state)
        key, k_state = self.k_conv1d(key, k_state)
        value, v_state = self.v_conv1d(value, v_state)
        shape = (-1, self.num_heads, self.head_dim)
        output, recurrent_state = self._recurrent_kda(
            query.view(shape),
            key.view(shape),
            value.view(shape),
            self._gate(raw_gate.view(shape)),
            torch.sigmoid(beta.float()),
            recurrent_state,
        )
        output = self.o_norm(output) * torch.sigmoid(output_gate.view(shape))
        sconv_state = torch.cat((q_state, k_state, v_state), dim=0)
        return output.flatten(1), sconv_state, recurrent_state

    def _metadata(self) -> LinearAttentionMetadata | None:
        if not is_forward_context_available():
            return None
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        layer_metadata = metadata[self.prefix]
        if not isinstance(layer_metadata, LinearAttentionMetadata):
            raise TypeError("KDA received incompatible attention metadata")
        return layer_metadata

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del positions
        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)
        f_a, _ = self.f_a_proj(hidden_states)
        raw_gate, _ = self.f_b_proj(f_a)
        beta, _ = self.b_proj(hidden_states)
        if self.use_full_rank_gate:
            output_gate, _ = self.g_proj(hidden_states)
        else:
            g_a, _ = self.g_a_proj(hidden_states)
            output_gate, _ = self.g_b_proj(g_a)

        metadata = self._metadata()
        if metadata is None:
            sconv_state = hidden_states.new_zeros(
                3 * self.local_projection_size,
                self.conv_size - 1,
            )
            recurrent_state = hidden_states.new_zeros(
                self.num_heads,
                self.head_dim,
                self.head_dim,
                dtype=torch.float32,
            )
            output, _, _ = self._run_sequence(
                query,
                key,
                value,
                raw_gate,
                beta,
                output_gate,
                sconv_state,
                recurrent_state,
            )
        else:
            output = torch.zeros_like(query)
            sconv_cache, recurrent_cache = self.kv_cache
            query_starts = metadata.query_start_loc
            for request_idx in range(query_starts.numel() - 1):
                start = int(query_starts[request_idx].item())
                end = int(query_starts[request_idx + 1].item())
                if start == end:
                    continue
                state_idx = int(metadata.state_indices_tensor[request_idx].item())
                if state_idx < 0:
                    continue
                query_length = end - start
                has_initial_state = int(metadata.seq_lens[request_idx].item()) > (
                    query_length
                )
                if has_initial_state:
                    sconv_state = sconv_cache[state_idx]
                    recurrent_state = recurrent_cache[state_idx]
                else:
                    sconv_state = sconv_cache[state_idx].new_zeros(
                        sconv_cache.shape[1:]
                    )
                    recurrent_state = recurrent_cache[state_idx].new_zeros(
                        recurrent_cache.shape[1:]
                    )
                if not is_conv_state_dim_first():
                    sconv_state = sconv_state.transpose(0, 1)
                request_output, sconv_state, recurrent_state = self._run_sequence(
                    query[start:end],
                    key[start:end],
                    value[start:end],
                    raw_gate[start:end],
                    beta[start:end],
                    output_gate[start:end],
                    sconv_state,
                    recurrent_state,
                )
                output[start:end] = request_output
                if not is_conv_state_dim_first():
                    sconv_state = sconv_state.transpose(0, 1)
                sconv_cache[state_idx].copy_(sconv_state)
                recurrent_cache[state_idx].copy_(recurrent_state)

        output, _ = self.o_proj(output)
        return output
