# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import Lfm2Config

from vllm import envs
from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import (HasInnerState, IsHybrid, SupportsLoRA, SupportsPP,
                         SupportsQuant)
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class Lfm2MLP(nn.Module):

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: Optional[float],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[ff_dim] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.w2 = RowParallelLinear(
            input_size=ff_dim,
            output_size=dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w1(x)
        x = self.act_fn(gate_up)
        x, _ = self.w2(x)
        return x


class Lfm2Attention(nn.Module):

    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
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
        self.head_dim = self.hidden_size // self.total_num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(n_tokens, self.num_heads, self.head_dim).contiguous()
        k = k.view(n_tokens, self.num_kv_heads, self.head_dim).contiguous()
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(n_tokens, self.num_heads * self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads * self.head_dim)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class Lfm2AttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.layer_idx = layer_idx

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)

        self.self_attn = Lfm2Attention(
            config=config,
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.feed_forward = Lfm2MLP(
            dim=config.block_dim,
            ff_dim=config.block_ff_dim,
            multiple_of=config.block_multiple_of,
            auto_adjust_ff_dim=config.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=config.block_ffn_dim_multiplier,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        return self.feed_forward(hidden_states), residual


class Lfm2ShortConvDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.conv = ShortConv(
            config=config,
            dim=config.conv_dim,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            prefix=f"{prefix}.conv",
        )

        self.feed_forward = Lfm2MLP(
            dim=config.block_dim,
            ff_dim=config.block_ff_dim,
            multiple_of=config.block_multiple_of,
            auto_adjust_ff_dim=config.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=config.block_ffn_dim_multiplier,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(
                hidden_states, residual)
        output = torch.empty_like(hidden_states)
        self.conv(
            hidden_states,
            output,
            conv_metadata=None,
        )
        hidden_states, residual = self.ffn_norm(output, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Lfm2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size)

        def get_layer(prefix: str):
            layer_idx = extract_layer_index(prefix)
            is_attn = self.config.layer_types[layer_idx] == "full_attention"
            layer_class = (Lfm2AttentionDecoderLayer
                           if is_attn else Lfm2ShortConvDecoderLayer)
            return layer_class(
                config,
                layer_idx,
                model_config,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        if get_pp_group().is_last_rank:
            self.embedding_norm = RMSNorm(config.hidden_size,
                                          eps=config.norm_eps)
        else:
            self.embedding_norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".w1", ".w1", 0),
            (".w1", ".w3", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Lfm2ForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP,
                      IsHybrid, SupportsQuant):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "w1": [
            "w1",
            "w3",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, ...]:

        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
        use_v1: bool = True,
    ) -> tuple[tuple[int, int]]:
        """ Calculate shapes for LFM2's convolutional cache.

        Args:
            vllm_config: vLLM config
            use_v1: Get shapes for V1 (or V0)

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config

        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            intermediate_size=hf_config.conv_dim,
            conv_kernel=hf_config.conv_L_cache,
            use_v1=use_v1,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert (not cache_config.enable_prefix_caching
                ), "Lfm2 currently does not support prefix caching"
        assert envs.VLLM_USE_V1, (
            "Lfm2ForCausalLM doesn't support vLLM v0. Please enable v1")

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.scheduler_config = scheduler_config
        self.model_config = vllm_config.model_config

        self.model = Lfm2Model(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = self.config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)