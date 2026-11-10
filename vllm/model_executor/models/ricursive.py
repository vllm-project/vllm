# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Ricursive (Ripple) model.

Dense decoder with PeriNorm blocks (pre/post norms around attention and MLP),
per-head Q/K RMSNorm, sigmoid-gated attention output, value residuals from the
scaled input embeddings, sliding-window RoPE layers and NoPE global layers.
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
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
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.ricursive import RicursiveConfig

from .interfaces import EagleModelMixin, SupportsEagle3, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class RicursiveMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        clamp_threshold: float | None,
        quant_config: QuantizationConfig | None = None,
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
        self.act_fn = (
            SiluAndMulWithClamp(clamp_threshold) if clamp_threshold else SiluAndMul()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class RicursiveAttention(nn.Module):
    def __init__(
        self,
        config: RicursiveConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError(
                f"Ricursive requires num_attention_heads ({self.total_num_heads}) "
                f"and num_key_value_heads ({self.total_num_kv_heads}) to be "
                f"divisible by the tensor parallel size ({tp_size})."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.use_gated_attention = config.use_gated_attention
        self.gate_full_precision = config.gate_full_precision
        self.use_valres = config.use_valres

        # V reads a different input than Q/K/G when value residuals are on,
        # so only Q, K and the output gate share a projection.
        qkg_sizes = [self.total_num_heads * self.head_dim]
        qkg_sizes.append(self.total_num_kv_heads * self.head_dim)
        if self.use_gated_attention:
            qkg_sizes.append(self.total_num_heads * self.head_dim)
        self.qkg_proj = MergedColumnParallelLinear(
            hidden_size,
            qkg_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkg_proj",
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if self.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.rotary_emb = None
        if is_sliding:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=True,
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window if is_sliding else None,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        value_residual: torch.Tensor | None,
    ) -> torch.Tensor:
        qkg, _ = self.qkg_proj(hidden_states)
        q, k, g = qkg.split(
            [self.q_size, self.kv_size, qkg.shape[-1] - self.q_size - self.kv_size],
            dim=-1,
        )
        value_input = hidden_states
        if self.use_valres:
            blend = torch.sigmoid(self.value_res_lambda).to(hidden_states.dtype)
            value_input = torch.lerp(value_residual, hidden_states, blend)
        v, _ = self.v_proj(value_input)

        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim)))
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        if self.use_gated_attention:
            if self.gate_full_precision:
                g = g.float()
            attn_output = attn_output * torch.sigmoid(g).to(attn_output.dtype)
        output, _ = self.o_proj(attn_output)
        return output


class RicursiveDecoderLayer(nn.Module):
    def __init__(
        self,
        config: RicursiveConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = RicursiveAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = RicursiveMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            clamp_threshold=config.ffn_clamp_threshold,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        value_residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, value_residual)
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class RicursiveModel(nn.Module, EagleModelMixin):
    supports_aux_hidden_states_over_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RicursiveDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_scale = config.embed_scale
        self.use_valres = config.use_valres

        tensor_names = ["hidden_states", "residual"]
        if self.use_valres:
            tensor_names.append("value_residual")
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            tensor_names, config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.embed_scale is not None:
            embeds = embeds * self.embed_scale
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            # Cloned because the fused add-norm updates the residual in place.
            value_residual = hidden_states.clone() if self.use_valres else None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            value_residual = (
                intermediate_tensors["value_residual"] if self.use_valres else None
            )

        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)

        aux_hidden_states: list[torch.Tensor] = []
        if get_pp_group().is_first_rank:
            self._maybe_add_hidden_state(
                aux_hidden_states, self.start_layer, hidden_states, residual
            )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions, hidden_states, residual, value_residual
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            tensors = {
                "hidden_states": hidden_states,
                "residual": residual,
                **self.pack_local_aux_hidden_states(aux_hidden_states),
            }
            if self.use_valres:
                tensors["value_residual"] = value_residual
            return IntermediateTensors(tensors)

        hidden_states, _ = self.norm(hidden_states, residual)

        aux_hidden_states = remote_aux + aux_hidden_states
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class RicursiveForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkg_proj", 0),
            ".k_proj": (".qkg_proj", 1),
            ".g_proj": (".qkg_proj", 2),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )
    packed_modules_mapping = {
        "qkg_proj": ["q_proj", "k_proj", "g_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = RicursiveModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
