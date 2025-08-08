# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# Adapted from https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/main/modeling_moonshot_kimia.py
# This file is meant to be used in kimi_audio.py only
#
# The code is based on qwen2 (qwen2/modeling_qwen2.py) and DeepSeek-V2 (DeepSeek-V2/modeling_deepseek.py), but modified for KimiAudio.
#
# Licensing Information:
# - Code derived from qwen2 (qwen2/modeling_qwen2.py) and DeepSeek-V2 (DeepSeek-V2/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn

import transformers
from packaging import version

assert version.parse(transformers.__version__) >= version.parse("4.34.1")

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import (
    logging,
)
from ...transformers_utils.configs import KimiAudioConfig
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2PreTrainedModel,
)
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

if version.parse(transformers.__version__) >= version.parse("4.35.0"):
    from transformers.utils import is_flash_attn_2_available as is_flash_attn_available
else:
    from transformers.utils import is_flash_attn_available

if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
else:
    raise RuntimeError("flash attention must be installed")

from .interfaces import MixtureOfExperts, SupportsPP
from vllm.distributed import (get_ep_group, get_pp_group,
                              get_tensor_model_parallel_world_size)
from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
                         get_current_vllm_config)
from vllm.sequence import IntermediateTensors
from .utils import (PPMissingLayer, LayerFn,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory,
                    maybe_prefix, maybe_offload_to_cpu)
from .qwen2 import Qwen2MLP, Qwen2Attention
import vllm.envs as envs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.attention import AttentionType
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)


class MoonshotMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
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
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
    

class MoonshotDecoderLayer(nn.Module):
    def __init__(
        self, 
        config: KimiAudioConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)
        
        attn_type = AttentionType.DECODER
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = MoonshotMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    

def make_layers(
    num_hidden_layers: int,
    num_all_layers: int,
    has_mimo_layer: bool,
    layer_fn: LayerFn,
    prefix: str,
) -> tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    start_layer, end_layer = get_pp_indices_kimia(num_hidden_layers,
                                                  num_all_layers,
                                                  has_mimo_layer,
                                                  get_pp_group().rank_in_group,
                                                  get_pp_group().size)
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)] + [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
            for idx in range(start_layer, end_layer)
        ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
    return start_layer, end_layer, modules


def get_pp_indices_kimia(num_hidden_layers: int, num_all_layers: int,
                         mimo_layer_idx: int, pp_rank: int, pp_size: int) -> tuple[int, int]:
    """Calculate layer indices for pipeline parallelism in Kimia dual-stream architecture.
    
    Kimia model implements a dual-stream transformer architecture for audio-text processing:
    - Text stream: Processes text tokens throughout the entire model
    - MIMO (Multi-Input Multi-Output) stream: Processes audio features starting from a 
      specific layer where hidden states are copied from the text stream
    
    The total computational load (num_all_layers) is the sum of text layers and MIMO layers.
    This function distributes the load evenly across pipeline parallel ranks while respecting
    the architectural constraints.
    
    Args:
        num_hidden_layers: Number of layers to create for current stream.
                          Either text_layers or mimo_layers depending on which stream
                          is being instantiated.
        num_all_layers: Total computational load (text_layers + mimo_layers).
                       Represents the aggregate compute across both streams.
        mimo_layer_idx: Layer index in the text stream where hidden states are 
                       branched to initialize the MIMO stream. This defines the
                       boundary between text-only and dual-stream processing regions.
        pp_rank: Current pipeline parallel rank (0 to pp_size-1).
        pp_size: Total number of pipeline parallel ranks.
    
    Returns:
        tuple[int, int]: (start_layer, end_layer) - Half-open interval [start, end)
                        of layer indices to instantiate on this rank for the current stream.
    """
    num_remaining = num_all_layers - num_hidden_layers
    is_creating_mimo = (num_remaining > num_hidden_layers)
    
    if is_creating_mimo:
        mimo_layers = num_hidden_layers
        text_layers = num_remaining
    else:
        text_layers = num_hidden_layers
        mimo_layers = num_remaining
    layers_per_rank = num_all_layers // pp_size
    remainder = num_all_layers % pp_size
    
    if pp_rank < pp_size - remainder:
        global_start = pp_rank * layers_per_rank
        global_end = global_start + layers_per_rank
    else:
        global_start = pp_rank * layers_per_rank + (pp_rank - (pp_size - remainder))
        global_end = global_start + layers_per_rank + 1
    
    if is_creating_mimo:
        mimo_global_start = text_layers
        if global_end <= mimo_global_start:
            return (0, 0)
        elif global_start >= mimo_global_start:
            start_layer = global_start - mimo_global_start
            end_layer = global_end - mimo_global_start
        else:
            start_layer = 0
            end_layer = global_end - mimo_global_start
        
        start_layer = max(0, min(start_layer, mimo_layers))
        end_layer = max(0, min(end_layer, mimo_layers))
    else:
        if global_start >= text_layers:
            return (0, 0)
        elif global_end <= text_layers:
            start_layer = global_start
            end_layer = global_end
        else:
            start_layer = global_start
            end_layer = text_layers
        
        start_layer = max(0, min(start_layer, text_layers))
        end_layer = max(0, min(end_layer, text_layers))
    
    return (start_layer, end_layer)


class MoonshotKimiaModel(Qwen2PreTrainedModel):

    config_class = KimiAudioConfig

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: KimiAudioConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.kimia_mimo_transformer_from_layer_index = (
            config.kimia_mimo_transformer_from_layer_index
        )

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            config.num_hidden_layers + config.kimia_mimo_layers,
            self.kimia_mimo_transformer_from_layer_index,
            lambda prefix: MoonshotDecoderLayer(config=config,
                                                cache_config=cache_config,
                                                quant_config=quant_config,
                                                prefix=prefix),
            prefix=f"{prefix}.layers"
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.mimo_start_layer, self.mimo_end_layer, self.mimo_layers = make_layers(
            config.kimia_mimo_layers,
            config.kimia_mimo_layers + config.num_hidden_layers,
            self.kimia_mimo_transformer_from_layer_index,
            lambda prefix: MoonshotDecoderLayer(config=config,
                                                cache_config=cache_config,
                                                quant_config=quant_config,
                                                prefix=prefix),
            prefix=f"{prefix}.mimo_layers"
        )

        if get_pp_group().is_last_rank:
            self.mimo_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.mimo_norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: torch.Tensor = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        mimo_hidden_states, mimo_residual = None, None
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
            mimo_hidden_states = intermediate_tensors.get("mimo_hidden_states")
            mimo_residual = intermediate_tensors.get("mimo_residual")
        
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        for layer in self.mimo_layers[self.mimo_start_layer:self.mimo_end_layer]:
            mimo_hidden_states, mimo_residual = layer(
                positions,
                mimo_hidden_states,
                mimo_residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
                "mimo_hidden_states": mimo_hidden_states,
                "mimo_residual": mimo_residual,
            })
        
        hidden_states, _ = self.norm(hidden_states, residual)
        mimo_hidden_states, _ = self.mimo_norm(mimo_hidden_states, mimo_residual)
        return hidden_states, mimo_hidden_states

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
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_prefix, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_prefix)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                internal_name = f"mlp.{param_prefix}.weight"
                if internal_name not in params_dict:
                    continue

                param = params_dict[internal_name]
                loader = getattr(param, "weight_loader", None)
                if loader is None:
                    if shard_id is None:
                        param.data.copy_(loaded_weight.T)
                    else:
                        W = loaded_weight.T
                        chunk_size = W.size(1) // 2
                        start = shard_id * chunk_size
                        end   = (shard_id + 1) * chunk_size
                        param.data[:, start:end].copy_(W[:, start:end])
                else:
                    loader(param, loaded_weight, shard_id)
                loaded.add(internal_name)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                loader = getattr(param, "weight_loader", None) or default_weight_loader
                loader(param, loaded_weight)
                loaded.add(name)
        return loaded
    

class MoonshotKimiaVllmWrapper(nn.Module):
    """Wrapper for KimiAudioModel to integrate with VLLM"""

    def __init__(self, kimi_model: MoonshotKimiaModel, config: KimiAudioConfig):
        super().__init__()
        self.kimia = kimi_model
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                input_ids: torch.Tensor = None,
                positions: torch.Tensor = None,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: torch.Tensor = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        out = self.kimia(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(out, IntermediateTensors):
            return out

        main_hidden = out

        main_hidden, _ = self.norm(main_hidden, torch.zeros_like(main_hidden))
        return main_hidden


class MoonshotKimiaForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight", "mimo_output.weight"]
    config_class = KimiAudioConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MoonshotKimiaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mimo_output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: tuple | BaseModelOutputWithPast= self.model(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_input_feature,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            hidden_states, mimo_hidden_states = (
                outputs.last_hidden_state[0],
                outputs.last_hidden_state[1],
            )
        else:
            hidden_states, mimo_hidden_states = outputs[0], outputs[1]

        audio_logits = self.lm_head(hidden_states)
        text_logits = self.mimo_output(mimo_hidden_states)

        if not return_dict:
            output = (text_logits, audio_logits) + outputs[2:]
            return output
        return CausalLMOutputWithPast(
            loss=None,
            logits=(text_logits, audio_logits),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )