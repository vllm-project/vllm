# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LFM3 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.lfm3 import Lfm3Config

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention")


def get_rotary_embedding(config: Lfm3Config, layer_type: str) -> RotaryEmbedding:
    """RoPE of one attention layer type: `proportional` rotates the first
    `partial_rotary_factor` of the head with frequencies spread over the whole
    head, as the released full-attention layers do."""
    rope_parameters = config.rope_parameters[layer_type]
    if (
        rope_parameters.get("partial_rotary_factor", 1.0) != 1.0
        and rope_parameters.get("rope_type", "default") != "proportional"
    ):
        # the other types read the factor from elsewhere or scale the frequencies
        raise NotImplementedError(
            "LFM3 partial rotation is only supported with the proportional RoPE type"
        )
    return get_rope(
        config.head_dim,
        max_position=config.max_position_embeddings,
        is_neox_style=True,
        rope_parameters=rope_parameters,
    )


class Lfm3MLP(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x


class Lfm3Attention(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        layer_type: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
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

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_proj = (
            ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )
            if config.use_attention_gate
            else None
        )
        self.rotary_emb = get_rotary_embedding(config, layer_type)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=(
                config.sliding_window if layer_type == "sliding_attention" else None
            ),
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.q_size)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        k = k.view(-1, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.gate_proj is not None:
            gate, _ = self.gate_proj(hidden_states)
            attn_output = attn_output.view(-1, self.num_heads, self.head_dim)
            attn_output = attn_output * torch.sigmoid(gate)[..., None]
            attn_output = attn_output.view(-1, self.q_size)
        output, _ = self.o_proj(attn_output)
        return output


class Lfm3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_attention_layer = layer_type in ATTENTION_LAYER_TYPES
        if self.is_attention_layer:
            self.self_attn = Lfm3Attention(
                config,
                layer_type,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.short_conv = ShortConv(
                config=config,
                dim=config.hidden_size,
                layer_idx=layer_idx,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.short_conv",
            )
        self.feed_forward = Lfm3MLP(
            config, quant_config=quant_config, prefix=f"{prefix}.feed_forward"
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)
        if self.is_attention_layer:
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            output = torch.empty_like(hidden_states)
            self.short_conv(hidden_states, output)
            hidden_states = output
        if self.use_sandwich_norm:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        if self.use_sandwich_norm:
            hidden_states = self.post_ffn_norm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Lfm3Model(nn.Module, EagleModelMixin):
    supports_aux_hidden_states_over_pp = True

    # HF uses .conv. but vLLM uses .short_conv. to avoid LoRA regex collision
    # with the inner .conv.conv child, see Lfm2Model.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".conv.": ".short_conv."},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".w1": (".w13", 0),
            ".w3": (".w13", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Lfm3Config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Lfm3DecoderLayer(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.embedding_norm = PPMissingLayer()

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
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

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
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        aux_hidden_states = remote_aux + aux_hidden_states
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def _short_conv_state_shape(
    vllm_config: VllmConfig, config: Lfm3Config
) -> tuple[tuple[int, int]]:
    return MambaStateShapeCalculator.short_conv_state_shape(
        tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
        intermediate_size=config.hidden_size,
        conv_kernel=config.conv_L_cache,
        num_spec=vllm_config.num_speculative_tokens,
    )


def _short_conv_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, ...]:
    return MambaStateDtypeCalculator.short_conv_state_dtype(
        vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
    )


class Lfm3ForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsQuant,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "w13": ["w1", "w3"],
        "in_proj": ["in_proj"],
    }

    # Reuse the backbone mapper so LoRA/quantization see the same name mapping.
    hf_to_vllm_mapper = Lfm3Model.hf_to_vllm_mapper

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return _short_conv_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        return _short_conv_state_shape(vllm_config, vllm_config.model_config.hf_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Lfm3Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = Lfm3Model(
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
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=(
                ["lm_head."] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(weights)
