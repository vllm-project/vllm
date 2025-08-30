# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import (_ACTIVATION_REGISTRY,
                                                   GeluAndMul,
                                                   GeluAndMulSparse)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsQuant
from .utils import (AutoWeightsLoader, extract_layer_index,
                    is_pp_missing_parameter, make_layers, maybe_prefix)

logger = init_logger(__name__)


class Gemma3nAltUp(nn.Module):
    """Alternating updates (Altup)
    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.
    See more in the research paper:
    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        rms_norm_eps: float,
        altup_num_inputs: int,
        altup_coef_clip: float,
        altup_active_idx: int,
        quant_config: QuantizationConfig,
        prefix: str,
    ):
        super().__init__()

        self.altup_num_inputs = altup_num_inputs
        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip

        self.correction_coefs = ReplicatedLinear(
            altup_num_inputs,
            altup_num_inputs,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.correction_coefs",
            return_bias=False,
        )
        self.prediction_coefs = ReplicatedLinear(
            altup_num_inputs,
            altup_num_inputs**2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.prediction_coefs",
            return_bias=False,
        )
        self.modality_router = ReplicatedLinear(
            hidden_size,
            altup_num_inputs,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.modality_router",
            return_bias=False,
        )
        self.router_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=rms_norm_eps,
        )
        self.router_input_scale = torch.tensor(
            hidden_size**-1.0, dtype=self.modality_router.weight.dtype)
        self.correct_output_scale = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32))

    def _compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        return (corrected.type_as(self.correct_output_scale) *
                self.correct_output_scale).type_as(corrected)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden:       [altup_num_inputs, num_tokens, hidden_size]
        # modalities:   [num_tokens, num_altup_inputs]
        # all_coefs:    [num_tokens, num_altup_inputs ** 2]
        modalities = self._compute_router_modalities(
            hidden_states[self.altup_active_idx])
        all_coefs = self.prediction_coefs(modalities)

        # Reshape and transpose the 2D matrix for the matmul.
        # all_coefs_T:  [num_tokens, num_altup_inputs, num_altup_inputs]
        all_coefs_T = all_coefs.reshape(
            -1,
            self.altup_num_inputs,
            self.altup_num_inputs,
        ).permute(0, 2, 1)

        # hidden_states to [num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 0), all_coefs_T)
        # [altup_num_inputs, num_tokens, hidden_size]
        predictions = predictions.permute(2, 0, 1)
        predictions += hidden_states
        return predictions.contiguous()

    def correct(self, predictions: torch.Tensor,
                activated: torch.Tensor) -> torch.Tensor:
        # predictions:  [altup_num_inputs, num_tokens, hidden_size]
        # activated:    [num_tokens, hidden_size]
        # modalities:   [num_tokens, altup_num_inputs]
        modalities = self._compute_router_modalities(activated)
        # innovation:   [num_tokens, altup_num_inputs]
        innovation = activated - predictions[self.altup_active_idx]
        # innovation:   [altup_num_inputs, num_tokens, hidden_size]
        innovation = innovation.repeat(self.altup_num_inputs, 1, 1)

        # Permute to [altup_num_inputs, num_tokens] as the last dim
        # is a scalar applied to each altup input and expand on
        # num_tokens dim for broadcastability over hidden_size.
        # all_coefs:    [num_tokens, altup_num_inputs]
        all_coefs = self.correction_coefs(modalities) + 1.0
        # all_coefs:    [altup_num_inputs, num_tokens, 1]
        all_coefs = all_coefs.T.unsqueeze(-1)

        # Elementwise (broadcast over hidden_size).
        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions

        return corrected.contiguous()


class Gemma3nLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(
        self,
        hidden_size: int,
        laurel_rank: int,
        rms_norm_eps: float,
        *,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str,
    ) -> None:
        super().__init__()

        self.linear_left = ColumnParallelLinear(
            hidden_size,
            laurel_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_left",
            return_bias=False,
        )
        self.linear_right = RowParallelLinear(
            laurel_rank,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_right",
            return_bias=False,
        )
        self.post_laurel_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        laurel_x = self.linear_left(x)
        laurel_x = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        return x + normed_laurel_x


class Gemma3nMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        activation_sparsity: float = 0.0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
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
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`.")

        self.act_fn = GeluAndMulSparse(
            activation_sparsity=activation_sparsity,
            approximate="tanh") if activation_sparsity > 0.0 else GeluAndMul(
                approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma3nAttention(nn.Module):

    def __init__(self,
                 config: Gemma3nTextConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=config.rms_norm_eps,
                              has_weight=False)

        layer_idx = extract_layer_index(prefix)
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if is_sliding else None

        # Initialize the rotary embedding.
        if is_sliding:
            # Local attention. Override the values in config.json.
            rope_theta = config.rope_local_base_freq
            rope_scaling = {"rope_type": "default"}
        else:
            # Global attention. Use the values in config.json.
            rope_theta = config.rope_theta
            rope_scaling = config.rope_scaling

        first_kv_shared_layer_idx = (config.num_hidden_layers -
                                     config.num_kv_shared_layers)
        self.is_kv_shared = layer_idx >= first_kv_shared_layer_idx

        kv_sharing_target_layer_name = None
        if self.is_kv_shared:
            # Last full attention layer is 1 before sharing
            # Last sliding attention layer is 2 before sharing
            offset = 2 if self.sliding_window is not None else 1
            kv_shared_layer_index = first_kv_shared_layer_idx - offset
            if kv_shared_layer_index >= 0:
                # Only the greater layer is required to specify sharing.
                kv_sharing_target_layer_name = f"language_model.model.layers.{kv_shared_layer_index}.self_attn.attn"  # noqa: E501

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)
        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = self.v_norm(v)
        v = v.flatten(-2, -1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)
        return output


class Gemma3nDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma3nTextConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert isinstance(config, Gemma3nTextConfig)
        self.altup_active_idx = config.altup_active_idx
        assert config.altup_correct_scale

        self.altup = Gemma3nAltUp(
            hidden_size=config.hidden_size,
            rms_norm_eps=config.rms_norm_eps,
            altup_num_inputs=config.altup_num_inputs,
            altup_coef_clip=config.altup_coef_clip,
            altup_active_idx=config.altup_active_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.altup",
        )
        self.self_attn = Gemma3nAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma3nMLP(
            hidden_size=config.hidden_size,
            # NOTE: Matformer https://github.com/huggingface/transformers/blob/a52478253bbe522a420e88ea3940d4d98a935300/src/transformers/models/gemma3n/modular_gemma3n.py#L258 # noqa: E501
            intermediate_size=config.intermediate_size[extract_layer_index(
                prefix)],
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            activation_sparsity=config.activation_sparsity_pattern[
                extract_layer_index(prefix)],
            prefix=f"{prefix}.mlp",
        )
        self.laurel = Gemma3nLaurelBlock(
            hidden_size=config.hidden_size,
            laurel_rank=config.laurel_rank,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.laurel",
        )

        # NOTE(rob): should be ColumnParallelLinear and RowParallelLinear
        # But, we need to add per_layer_input_gate(x) to per_layer_input.
        # per_layer_input cannot be sharded, so we replicate for now.
        self.per_layer_input_gate = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size_per_layer_input,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.per_layer_input_gate",
            return_bias=False,
        )
        self.per_layer_projection = ReplicatedLinear(
            config.hidden_size_per_layer_input,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.per_layer_projection",
            return_bias=False,
        )

        # LayerNorms.
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_per_layer_input_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.act_fn = _ACTIVATION_REGISTRY[config.hidden_activation]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ActUp (predict).
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.altup_active_idx]
        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # Attention.
        attn = self.self_attn(
            positions=positions,
            hidden_states=active_prediction_normed,
            **kwargs,
        )
        attn = self.post_attention_layernorm(attn)
        attn_gated = attn + active_prediction
        attn_laurel = (attn_gated + laurel_output) / torch.sqrt(
            torch.tensor(2.0))

        # MLP.
        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        # ActUp (connect).
        corrected_predictions = self.altup.correct(predictions,
                                                   attn_ffw_laurel_gated)
        first_prediction = corrected_predictions[self.altup_active_idx]
        first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate adapted from jax.numpy.einsum("btd,dp->btp", ...)
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = torch.mul(first_prediction, per_layer_input)

        # per_layer_projection adapted from jax.numpy.einsum("btp,pd->btd", ...)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)
        corrected_predictions[1:] += first_prediction

        return corrected_predictions


@support_torch_compile
class Gemma3nTextModel(nn.Module, SupportsQuant):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.embed_scale = torch.tensor(
            config.hidden_size**0.5,
            dtype=self.embed_tokens.weight.dtype,
        )
        # Additional per-layer embeddings (PLE)
        self.embed_tokens_per_layer = VocabParallelEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            quant_config=quant_config,
            prefix=f"{prefix}.per_layer_embed_tokens",
        )
        self.embed_scale_per_layer = torch.tensor(
            config.hidden_size_per_layer_input**0.5,
            dtype=self.embed_tokens.weight.dtype,
        )
        self.per_layer_model_projection = ColumnParallelLinear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.per_layer_model_projection",
        )
        self.per_layer_projection_norm = RMSNorm(
            hidden_size=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )
        self.per_layer_input_scale = torch.rsqrt(torch.tensor(2.0)).to(
            self.embed_tokens.weight.dtype)
        self.per_layer_projection_scale = torch.tensor(
            config.hidden_size**0.5,
            dtype=self.embed_tokens.weight.dtype,
        )
        self.altup_projections = nn.ModuleList([
            ColumnParallelLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.altup_projections.{idx-1}",
            ) for idx in range(1, self.config.altup_num_inputs)
        ])
        self.altup_unembed_projections = nn.ModuleList([
            ColumnParallelLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.altup_unembed_projections.{idx-1}",
            ) for idx in range(1, self.config.altup_num_inputs)
        ])

        # Transformer blocks.
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma3nDecoderLayer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.eps = torch.tensor(torch.finfo().min)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embed_scale

    def get_per_layer_input_embeddings(
            self, input_ids: torch.Tensor) -> torch.Tensor:
        # Deal with the fact that vocab_size_per_layer_input < vocab_size
        # which causes us to have some out of vocab tokens by setting
        # those token ids to 0. This matches the HF implementation.
        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0, input_ids < self.config.vocab_size_per_layer_input)
        per_layer_inputs_tokens = torch.where(per_layer_inputs_mask, input_ids,
                                              torch.zeros_like(input_ids))
        return self.embed_tokens_per_layer(
            per_layer_inputs_tokens) * self.embed_scale_per_layer

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is not None:
            hidden_states_0 = inputs_embeds
        else:
            hidden_states_0 = self.get_input_embeddings(input_ids)

        per_layer_projection = self.per_layer_model_projection(hidden_states_0)
        per_layer_projection = per_layer_projection.reshape(
            *hidden_states_0.shape[:-1],
            self.config.num_hidden_layers,
            self.config.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(
            per_layer_projection)

        if per_layer_inputs is not None:
            # Profiling run does not compute per_layer_inputs
            per_layer_inputs = per_layer_projection + per_layer_inputs
            per_layer_inputs *= self.per_layer_input_scale
        else:
            per_layer_inputs = per_layer_projection

        # Altup embed.
        hidden_states = [hidden_states_0] * self.config.altup_num_inputs
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1,
                                      keepdim=True)**0.5
        for i in range(1, self.config.altup_num_inputs):
            hidden_states[i] = self.altup_projections[i - 1](hidden_states[i])
            new_magnitude = torch.mean(hidden_states[i]**2,
                                       dim=-1,
                                       keepdim=True)**0.5
            hidden_states[i] *= target_magnitude / torch.maximum(
                new_magnitude, self.eps)
        hidden_states = torch.stack(hidden_states, dim=0)

        # Transformer blocks.
        for layer_idx, layer in enumerate(self.layers):
            # [altup_num_inputs, num_tokens, hidden_size]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                per_layer_input=per_layer_inputs[:, layer_idx, :],
                **kwargs,
            )

        # Altup unembed.
        target_magnitude = torch.mean(hidden_states[0]**2,
                                      dim=-1,
                                      keepdim=True)**0.5
        for i in range(1, self.config.altup_num_inputs):
            hidden_states[i] = self.altup_unembed_projections[i - 1](
                hidden_states[i])
            new_magnitude = torch.mean(hidden_states[i]**2,
                                       dim=-1,
                                       keepdim=True)**0.5
            hidden_states[i] *= target_magnitude / torch.maximum(
                new_magnitude, self.eps)
        # [altup_num_inputs,num_tokens,hidden_size] -> [num_tokens,hidden_size]
        hidden_states = torch.mean(hidden_states, dim=0)

        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                # Avoid spurious match with ".up_proj".
                if "altup_projections" in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Gemma3nForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        del lora_config  # Unused.
        super().__init__()
        self.config = config
        self.cache_config = vllm_config.cache_config
        self.model = Gemma3nTextModel(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        per_layer_inputs: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        hidden_states = self.model(
            input_ids,
            positions,
            per_layer_inputs=per_layer_inputs,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata],
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.embed_tokens, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self,
                                   skip_substrs=([
                                       "embed_audio.", "embed_vision.",
                                       "audio_tower.", "vision_tower."
                                   ]))
        return loader.load_weights(weights)
