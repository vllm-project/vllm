# Copyright 2024 The vLLM team.
# Copyright 2024 Meta Platforms, Inc. and affiliates. All rights reserved.
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
"""Llama model for fairseq2 weights."""

from typing import Any, Dict, Iterable, Optional, Set, Tuple, Type

import torch
from torch.nn.parameter import Parameter
from transformers import LlamaConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.llama import (LlamaAttention,
                                              LlamaDecoderLayer,
                                              LlamaForCausalLM, LlamaMLP,
                                              LlamaModel)
from vllm.platforms import current_platform

from .utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    extract_layer_index, maybe_prefix)

logger = init_logger(__name__)


class MergedColumnShardedLinear(MergedColumnParallelLinear):

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        if loaded_shard_id is None:
            raise ValueError(
                "Fairseq2 weights like gate_proj and up_proj are expected"
                "to be loaded separately, not already fused")

        assert loaded_shard_id is not None and loaded_shard_id < len(
            self.output_sizes)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            # possibly narrow the weight if not sharded
            start_idx = tp_rank * shard_size
            if loaded_weight.shape[output_dim] // tp_size == shard_size:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnShardedLinear, assume the weight is "
                    "the same for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class QKVShardedLinear(QKVParallelLinear):

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[str] = None):

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads +
                                self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.head_size

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            # possibly narrow the weight if not sharded
            start_idx = tp_rank * shard_size
            if loaded_weight.shape[output_dim] // tp_size == shard_size:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RowShardedLinear(RowParallelLinear):

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # possibly narrow the weight if not sharded
        input_dim = getattr(param, "input_dim", None)
        if input_dim is not None:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            if loaded_weight.shape[input_dim] // tp_size == shard_size:
                loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                     shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class VocabShardedEmbedding(VocabParallelEmbedding):

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)

        # If parameter does not have output dim, then it should
        # be copied onto all gpus (e.g. g_idx for act_order gptq).
        if output_dim is None:
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return

        # possibly narrow the weight if not sharded
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        shard_size = param.data.shape[output_dim]
        start_idx = tp_rank * shard_size
        if loaded_weight.shape[output_dim] // tp_size == shard_size:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

        if current_platform.is_hpu():
            # FIXME(kzawora): Weight copy with slicing bugs out on Gaudi here,
            # so we're using a workaround. Remove this when fixed in
            # HPU PT bridge.
            padded_weight = torch.cat([
                loaded_weight,
                torch.zeros(param.shape[0] - loaded_weight.shape[0],
                            *loaded_weight.shape[1:])
            ])
            param.data.copy_(padded_weight)
        else:
            param[:loaded_weight.shape[0]].data.copy_(loaded_weight)
            param[loaded_weight.shape[0]:].data.fill_(0)


class ShardedLMHead(VocabShardedEmbedding, ParallelLMHead):

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")


class Fairseq2LlamaMLP(LlamaMLP):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super(LlamaMLP, self).__init__()

        self.gate_up_proj = MergedColumnShardedLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.down_proj = RowShardedLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()


class Fairseq2LlamaAttention(LlamaAttention):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super(LlamaAttention, self).__init__()
        layer_idx = extract_layer_index(prefix)
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
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVShardedLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowShardedLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )


class Fairseq2LlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super(LlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = Fairseq2LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Fairseq2LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)


@support_torch_compile
class Fairseq2LlamaModel(LlamaModel):

    def __init__(
            self,
            *,
            vllm_config: VllmConfig,
            prefix: str = "",
            embed_type: Type[VocabParallelEmbedding] = VocabShardedEmbedding,
            layer_type: Type[LlamaDecoderLayer] = Fairseq2LlamaDecoderLayer):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            embed_type=embed_type,
            layer_type=layer_type,
        )


class Fairseq2LlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(LlamaForCausalLM, self).__init__()
        # override load_format here
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ShardedLMHead(
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
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        # For the model loader to read only the relevant checkpoint files
        self.allow_patterns_overrides = [
            # either the full checkpoint
            "model.pt",
            # or the tp-sharded checkpoint of the current rank
            f"model.{self.tp_rank}.pt",
        ]

    def _init_model(self, vllm_config: VllmConfig, prefix: str = ""):
        return Fairseq2LlamaModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        # fairseq2's serialization adds a wrapper to usual .pt state_dict's:
        # { "model_key": my_model_name, "my_model_name": state_dict }
        # which we first need to unpack
        weights_wrapped = dict(weights)
        weights = weights_wrapped[
            weights_wrapped["model_key"]].items()  # type: ignore

        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "decoder_frontend.embed.": "model.embed_tokens.",
                "decoder.": "model.",
                "final_proj.": "lm_head.",
            },
            orig_to_new_substr={
                ".self_attn_layer_norm.": ".input_layernorm.",
                ".ffn_layer_norm.": ".post_attention_layernorm.",
                ".self_attn.output_proj.": ".self_attn.o_proj.",
                ".ffn.gate_proj.": ".mlp.gate_proj.",
                ".ffn.inner_proj.": ".mlp.up_proj.",
                ".ffn.output_proj.": ".mlp.down_proj.",
                ".layer_norm.": ".norm.",
            },
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            (self.reshape_fairseq2_weights(name, loaded_weight)
             for name, loaded_weight in weights),
            mapper=hf_to_vllm_mapper,
        )

    def reshape_fairseq2_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        """Reshape fairseq2's weights."""

        def permute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
            attn_in = self.config.head_dim * n_heads
            # check for a sharded weight on dim 0
            if attn_in // self.tp_size == w.size()[0]:
                attn_in //= self.tp_size
                n_heads //= self.tp_size
            attn_out = self.config.hidden_size
            return (w.view(n_heads, attn_in // n_heads // 2, 2,
                           attn_out).transpose(1,
                                               2).reshape(attn_in, attn_out))

        modules = name.split(".")

        # rotary embeds should be sliced
        if "k_proj" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)

        elif "q_proj" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        return name, loaded_weight
