# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
"""PyTorch T5 model."""

import copy
import os
from typing import Iterable, List, Optional, Tuple

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch import nn  # type: ignore
from transformers import T5Config
from transformers.utils import logging

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata

from .utils import maybe_prefix

# from flash_attn import flash_attn_func

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_CHECKPOINT_FOR_DOC = "t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np  # type: ignore
        import tensorflow as tf  # type: ignore
    except ImportError:
        logger.error(
            "TensorFlow is to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from %s", tf_path)
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight name is %s", name)
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        if any(n in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
        ] for n in name):
            log_name = "/".join(name)
            logger.info("Skipping %s", log_name)
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            log_name = "/".join(name)
            logger.info("Skipping %s", log_name)
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight", None)
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer", None)
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer", None)
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer", None)
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm", None)
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm", None)
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight", None)
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias", None)
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier", None)
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head", None)
            elif (scope_names[0] == "wi" and len(scope_names) > 1
                  and scope_names[1].isdigit()):
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    log_name = "/".join(name)
                    logger.info("Skipping %s", log_name)
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight", None)
        if scope_names[0] != "embedding":
            logger.info(
                "Transpose weight of \
              shape %s for %s",
                str(array.shape),
                name,
            )
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError("Pointer and array shape mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight %s", name)
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)
    weight_not_copied = ", ".join(tf_weights.keys())
    logger.info(
        "Weights not copied to PyTorch \
        model: %s.",
        weight_not_copied,
    )
    return model


class T5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style.
        No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        T5 uses a layer_norm which only scales and doesn't
        shift, which is also known as Root Mean
        Square Layer Normalization https://arxiv.org/abs/1910.07467
        thus variance is calculated
        w/o mean and there is no bias. Additionally we want to
        make sure that the accumulation for half-precision
        inputs is done in fp32
        """

        variance = hidden_states.to(torch.float32).pow(2).mean(-1,
                                                               keepdim=True)
        adj_var = variance + self.variance_epsilon
        hidden_states = hidden_states * torch.rsqrt(adj_var)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Module):

    def __init__(
        self,
        config: T5Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model,
                                       config.d_ff,
                                       bias=False,
                                       quant_config=quant_config)
        self.wo = RowParallelLinear(config.d_ff,
                                    config.d_model,
                                    bias=False,
                                    quant_config=quant_config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (isinstance(self.wo.weight, torch.Tensor)
                and hidden_states.dtype != self.wo.weight.dtype
                and self.wo.weight.dtype != torch.int8):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)[0]
        return hidden_states


class T5DenseGatedActDense(nn.Module):

    def __init__(
        self,
        config: T5Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.wi_0 = ColumnParallelLinear(config.d_model,
                                         config.d_ff,
                                         bias=False,
                                         quant_config=quant_config)
        self.wi_1 = ColumnParallelLinear(config.d_model,
                                         config.d_ff,
                                         bias=False,
                                         quant_config=quant_config)
        self.wo = RowParallelLinear(config.d_ff,
                                    config.d_model,
                                    bias=False,
                                    quant_config=quant_config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl,
        # self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users
        # will force `_keep_in_fp32_modules` to be `None``
        if (isinstance(self.wo.weight, torch.Tensor)
                and hidden_states.dtype != self.wo.weight.dtype
                and self.wo.weight.dtype != torch.int8):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(
        self,
        config: T5Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config, quant_config)
        else:
            self.DenseReluDense = T5DenseActDense(config, quant_config)

        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):

    def __init__(
        self,
        config: T5Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        has_relative_attention_bias=False,
        prefix: str = "",
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        rel_num_bucket = config.relative_attention_num_buckets
        rel_max_dist = config.relative_attention_max_distance
        self.relative_attention_num_buckets = rel_num_bucket
        self.relative_attention_max_distance = rel_max_dist
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.inner_dim // self.n_heads,
            self.n_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            self.d_model,
            bias=False,
            quant_config=quant_config,
        )
        self.attn = Attention(
            self.n_heads,
            self.inner_dim // self.n_heads,
            scale=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Self-attention (if key_value_states is None) or
        attention over source sentence (provided by
        key_value_states).
        """
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([2048, 2048, 2048], dim=-1)
        if encoder_hidden_states is None:
            attn_output = F.scaled_dot_product_attention(q,
                                                         k,
                                                         v,
                                                         dropout_p=0.0)
        else:
            qkv_enc, _ = self.qkv_proj(encoder_hidden_states)
            _, k, v = qkv.split([2048, 2048, 2048], dim=-1)
            attn_output = F.scaled_dot_product_attention(q,
                                                         k,
                                                         v,
                                                         dropout_p=0.0)
        output, _ = self.out_proj(attn_output)
        present_key_value_state = (k, v) if self.is_decoder else None
        return output, present_key_value_state


class T5LayerSelfAttention(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        has_relative_attention_bias=False,
        prefix: str = ""
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            cache_config,
            quant_config,
            has_relative_attention_bias=has_relative_attention_bias,
            prefix=f"{prefix}.attn")
        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(hidden_states, kv_cache,
                                              attn_metadata)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states, ) + attention_output[1:]
        return outputs


class T5LayerCrossAttention(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config,
            cache_config,
            quant_config,
            has_relative_attention_bias=False,
            prefix=f"{prefix}.attn")
        
        self.layer_norm = T5LayerNorm(config.d_model,
                                      eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            hidden_states,
            kv_cache,
            attn_metadata,
            encoder_hidden_states,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output, ) + attention_output[1:]
        return outputs


class T5Block(nn.Module):

    def __init__(
        self,
        config: T5Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        has_relative_attention_bias=False,
        prefix: str = "",
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.self_attn = T5LayerSelfAttention(
            config,
            cache_config,
            quant_config,
            has_relative_attention_bias=has_relative_attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        if self.is_decoder:
            self.cross_attn = T5LayerCrossAttention(config, cache_config,
                                                    quant_config,
                                                    prefix=f"{prefix}.encoder_attn")
        self.fc = T5LayerFF(config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attention_outputs = self.self_attn(hidden_states, kv_cache,
                                                attn_metadata)
        hidden, _ = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]

        # clamp inf values to enable fp16 training
        if hidden.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden).any(),
                torch.finfo(hidden.dtype).max - 1000,
                torch.finfo(hidden.dtype).max,
            )
            hidden = torch.clamp(hidden, min=-clamp_value, max=clamp_value)

        do_cross_attention = (self.is_decoder
                              and encoder_hidden_states is not None)
        if do_cross_attention:
            cross_attention_outputs = self.cross_attn(hidden, kv_cache,
                                                      attn_metadata,
                                                      encoder_hidden_states)
            hidden = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden).any(),
                    torch.finfo(hidden.dtype).max - 1000,
                    torch.finfo(hidden.dtype).max,
                )
                hidden = torch.clamp(hidden, min=-clamp_value, max=clamp_value)

            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden = self.fc(hidden)

        # clamp inf values to enable fp16 training
        if hidden.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden).any(),
                torch.finfo(hidden.dtype).max - 1000,
                torch.finfo(hidden.dtype).max,
            )
            hidden = torch.clamp(hidden, min=-clamp_value, max=clamp_value)

        outputs = (hidden, ) + attention_outputs
        return outputs


class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5Stack(nn.Module):

    def __init__(
        self,
        config: T5Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        embed_tokens=None,
        prefix: str = "",
    ):
        super().__init__()
        self.cache_config = cache_config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList([
            T5Block(
                config,
                cache_config,
                quant_config,
                has_relative_attention_bias=bool(i == 0),
                prefix=f"{prefix}.block.{i}"
            ) for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model,
                                            eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.dropout(inputs_embeds)
        # print('t5 stack', type(hidden_states))
        for i, layer in enumerate(self.block):
            layer_outputs = layer(
                hidden_states,
                kv_caches[i],
                attn_metadata,
                encoder_hidden_states,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class T5Model(nn.Module):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    # def __init__(
    #     self,
    #     config: T5Config,
    #     cache_config: Optional[CacheConfig] = None,
    #     quant_config: Optional[QuantizationConfig] = None,
    #     lora_config: Optional[LoRAConfig] = None,
    # ):
        super().__init__()
        # self.shared = nn.Embedding(config.vocab_size, config.d_model)
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.padding_idx = config.pad_token_id
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.shared = VocabParallelEmbedding(
            self.vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size,
        )

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, cache_config, quant_config,
                               self.shared, prefix=f"{prefix}.encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, cache_config, quant_config,
                               self.shared, prefix=f"{prefix}.decoder")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        encoder_hidden_states = None

        if encoder_input_ids.numel() > 0:
            # Run encoder attention if a non-zero number of encoder tokens
            # are provided as input
            encoder_hidden_states = self.encoder(
                input_ids=encoder_input_ids,
                positions=encoder_positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        return decoder_outputs


class T5ForConditionalGeneration(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

    # def __init__(
    #     self,
    #     config: T5Config,
    #     cache_config: Optional[CacheConfig] = None,
    #     quant_config: Optional[QuantizationConfig] = None,
    #     lora_config: Optional[LoRAConfig] = None,
    # ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.model_dim = config.d_model
        self.model = T5Model(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))
        # self.model = T5Model(config,
        #                      cache_config,
        #                      quant_config,
        #                      lora_config=lora_config)
        print("lora_config", lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            num_embeddings=self.unpadded_vocab_size,
            embedding_dim=config.d_model,
            org_num_embeddings=config.vocab_size,
            bias=False,
        )

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                torch.Tensor of *decoder* input token ids.
            positions
                torch.Tensor of *decoder* position indices.
            encoder_input_ids
                torch.Tensor of *encoder* input token ids.
            encoder_positions
                torch.Tensor of *encoder* position indices
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Output torch.Tensor
        """
        return self.model(
            input_ids,
            positions,
            encoder_input_ids,
            encoder_positions,
            kv_caches,
            attn_metadata,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    stacked_params_mapping = {
        "q.weight": {
            "param_name": "qkv_proj.weight",
            "shard_id": "q",
        },
        "k.weight": {
            "param_name": "qkv_proj.weight",
            "shard_id": "k",
        },
        "v.weight": {
            "param_name": "qkv_proj.weight",
            "shard_id": "v",
        },
        "o.weight": {
            "param_name": "out_proj.weight",
            "shard_id": None,
        },
    }

    params_mapping = {
        "beta": "bias",
        "gamma": "weight",
        "LayerNorm": "layernorm",
    }

    def _rename_key(self, key: str):
        prefix = f"{self.base_model_prefix}."
        key = key[len(prefix):] if key.startswith(prefix) else key

        for src, dst in self.params_mapping.items():
            key = key.replace(src, dst)

        return key

    layer_type_mapping = {
        "encoder": {
            "layer.0": "self_attn",
            "layer.1": "fc",
        },
        "decoder": {
            "layer.0": "self_attn",
            "layer.1": "cross_attn",
            "layer.2": "fc",
        },
    }

    def _rename_layer_types(
        self,
        name: str,
    ) -> str:
        for enc_dec, mapping in self.layer_type_mapping.items():
            if enc_dec in name:
                for layer_num in mapping:
                    if layer_num in name:
                        name = name.replace(layer_num, mapping[layer_num])
        return name

    def _rename_stacked_param(
        self,
        name: str,
    ) -> Tuple[str, Optional[str]]:
        for key, mapping in self.stacked_params_mapping.items():
            if key in name and ".wo." not in name:
                name = name.replace(key, mapping["param_name"])
                return name, mapping["shard_id"]
        return name, None

    def match_weight_name(self, weights_tuple_list):
        out = set()
        for name, _ in weights_tuple_list:
            if "decoder" in name and "layer_norm" not in name:
                if (("layer.0" in name and "SelfAttention" not in name) or
                    ("layer.1" in name and "EncDecAttention" not in name) or
                    ("layer.2" in name and "DenseReluDense" not in name)):
                    print(name)
                    out.add(False)
                # elif 'layer.1' in name and \
                #   'EncDecAttention' not in name:
                #     print(name)
                #     out.add(False)
                # elif 'layer.2' in name and \
                #   'DenseReluDense' not in name:
                #     print(name)
                #     out.add(False)
                else:
                    out.add(True)
            elif "encoder" in name and "layer_norm" not in name:
                if ("layer.0" in name and "SelfAttention" not in name) or (
                        "layer.1" in name and "DenseReluDense" not in name):
                    print(name)
                    out.add(False)
                # elif 'layer.1' in name and \
                #   'DenseReluDense' not in name:
                #     print(name)
                #     out.add(False)
                else:
                    out.add(True)
            elif "decoder" not in name and "encoder" not in name:
                print(name)
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        model_params_dict = dict(self.model.named_parameters())
        weights_tuple_list = list(weights)

        shared_embedding_weight = None

        for name, loaded_weight in weights_tuple_list:
            name = self._rename_layer_types(name)
            name, shard_id = self._rename_stacked_param(name)
            if ("encoder.embed_tokens.weight" in name
                    or "decoder.embed_tokens.weight" in name
                    or "lm_head.weight" in name):
                assert (shared_embedding_weight is
                        None), "Conflicting embedding weights."
                shared_embedding_weight = loaded_weight
            else:
                # Skip the specific downstream task weight.
                if name.startswith("cls."):
                    continue
                # use Pooler instead.
                if name.startswith("pooler."):
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in model_params_dict:
                    continue
                if "bias.weight" in name and name not in model_params_dict:
                    continue

                param = model_params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if shard_id:
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader(param, loaded_weight)