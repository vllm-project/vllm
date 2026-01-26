# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral adaptation of the LLaMA architecture."""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .utils import AutoWeightsLoader


class MistralMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        gate_up_proj_bias: bool | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        gate_up_proj_bias = bias if gate_up_proj_bias is None else gate_up_proj_bias
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=gate_up_proj_bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class MistralAttention(LlamaAttention):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=prefix,
            attn_type=attn_type,
        )

        llama_4_scaling_config: dict[str, int | float | str] | None = getattr(
            config, "llama_4_scaling", None
        )
        self.do_llama_4_scaling = llama_4_scaling_config is not None
        if self.do_llama_4_scaling:
            assert llama_4_scaling_config is not None
            self.llama_4_scaling_original_max_position_embeddings = (
                llama_4_scaling_config["original_max_position_embeddings"]
            )
            self.llama_4_scaling_beta = llama_4_scaling_config["beta"]

    def _get_llama_4_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        # Llama4 scaling
        scaling = 1 + self.llama_4_scaling_beta * torch.log(
            1
            + torch.floor(
                positions / self.llama_4_scaling_original_max_position_embeddings
            )
        )
        # Broadcast over head_dim
        return scaling.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if self.do_llama_4_scaling:
            attn_scale = self._get_llama_4_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MistralDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            attn_layer_type=MistralAttention,
        )

        self.layer_idx = int(prefix.split(sep=".")[-1])
        quant_config = self.get_quant_config(vllm_config)
        config = config or vllm_config.model_config.hf_config

        do_fusion = getattr(
            quant_config, "enable_quantization_scaling_fusion", False
        ) and vllm_config.cache_config.cache_dtype.startswith("fp8")
        if do_fusion:
            self.input_layernorm.quant_scaling_from = self.self_attn.qkv_proj
            self.post_attention_layernorm.quant_scaling_from = self.mlp.gate_up_proj

        if getattr(config, "ada_rms_norm_t_cond", False):
            self.ada_rms_norm_t_cond = nn.Sequential(
                ColumnParallelLinear(
                    input_size=config.hidden_size,
                    output_size=config.ada_rms_norm_t_cond_dim,
                    bias=False,
                    return_bias=False,
                ),
                nn.GELU(),
                RowParallelLinear(
                    input_size=config.ada_rms_norm_t_cond_dim,
                    output_size=config.hidden_size,
                    bias=False,
                    return_bias=False,
                ),
            )
        else:
            self.ada_rms_norm_t_cond = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        t_cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.ada_rms_norm_t_cond is not None:
            assert t_cond is not None
            hidden_states = hidden_states * (1 + self.ada_rms_norm_t_cond(t_cond))

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class MistralModel(LlamaModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = MistralDecoderLayer,
    ):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        t_cond: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, t_cond=t_cond
        )


class MistralForCausalLM(LlamaForCausalLM):
    # Mistral: We don't support LoRA on the embedding layers.
    embedding_modules: dict[str, str] = {}

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "q_fake_quantizer.qscale_act": "attn.q_scale",
        "k_fake_quantizer.qscale_act": "k_scale",
        "v_fake_quantizer.qscale_act": "v_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = MistralDecoderLayer,
    ):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = MistralDecoderLayer,
    ):
        return MistralModel(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights
        )

    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        def permute(w: torch.Tensor, n_heads: int, attn_out: int):
            attn_in = self.config.head_dim * n_heads

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        # If using quantized model in mistral format,
        # quantization scales (qscale_weight) also need to be sliced
        if "wk" in modules and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.config.num_key_value_heads, self.config.hidden_size
            )
        elif (
            "wk" in modules
            and modules[-1] == "qscale_weight"
            and loaded_weight.numel() > 1
        ):
            loaded_weight = permute(loaded_weight, self.config.num_key_value_heads, 1)
        elif "wq" in modules and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.config.num_attention_heads, self.config.hidden_size
            )
        elif (
            "wq" in modules
            and modules[-1] == "qscale_weight"
            and loaded_weight.numel() > 1
        ):
            loaded_weight = permute(loaded_weight, self.config.num_attention_heads, 1)

        num_modules = len(modules)
        for i in range(num_modules):
            item = modules[i]
            next_item = modules[i + 1] if i < num_modules - 1 else None

            combined_item = f"{item}.{next_item}" if next_item is not None else None

            if combined_item in mapping:
                name = name.replace(combined_item, mapping[combined_item])
            elif item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
