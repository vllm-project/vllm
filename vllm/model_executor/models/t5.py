# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/t5/modeling_t5.py
# Copyright 2023 The vLLM team.
# Copyright 2020 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model."""
from typing import List, Optional, Tuple

import math, copy

import torch
from torch import nn
from transformers import T5Config

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.enc_dec_attention import (
    EncoderAttention,
    DecoderAttention,
    CrossAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states):
        hidden_states, _ = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states)[0])
        hidden_linear, _ = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states


class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        is_cross: bool,
        has_relative_attention_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        total_num_heads = config.num_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.n_heads = total_num_heads // tensor_model_parallel_world_size
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.k = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.v = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False)
        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

        self.is_cross = is_cross
        if self.is_decoder:
            if self.is_cross:
                self.attn = CrossAttention(self.n_heads, self.key_value_proj_dim, 1)
            else:
                self.attn = DecoderAttention(self.n_heads, self.key_value_proj_dim, 1)
        else:
            self.attn = EncoderAttention(self.n_heads, self.key_value_proj_dim, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device="cuda")[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device="cuda")[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # shape (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # print("hidden_states shape", hidden_states.shape)
        # print("hidden_states", hidden_states)
        q, _ = self.q(hidden_states)

        # print("q shape", q.shape)
        # print("q", q)
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        prompt_len = input_metadata.prompt_lens.max().item()
        context_len = input_metadata.context_lens.max().item()
        context_len = max(context_len, 1)
        # print("batch_size", batch_size)
        # print("seq_len", seq_len)
        # print("prompt_len", prompt_len)
        # print("context_len", context_len)

        block_size = 16

        if not self.is_decoder:
            # print("encoder self attention!")
            assert kv_cache is None
            # Encoder self attention, no cache operations
            k, _ = self.k(hidden_states)
            v, _ = self.v(hidden_states)



            if input_metadata.attn_bias is None:
                input_metadata.attn_bias = self.compute_bias(
                    prompt_len, (prompt_len + block_size - 1) // block_size * block_size
                ).repeat(batch_size, 1, 1, 1)
                for i in range(batch_size):
                    input_metadata.attn_bias[
                        i,
                        :,
                        :,
                        input_metadata.prompt_lens[i] :,
                    ] = torch.finfo(input_metadata.attn_bias.dtype).min

                # print("input_metadata.attn_bias shape", input_metadata.attn_bias.shape)
                # print("input_metadata.attn_bias", input_metadata.attn_bias)
            attn_output = self.attn(q, k, v, input_metadata)

        elif not self.is_cross:
            # print("decoder self attention!")
            # Decoder self attention
            k, _ = self.k(hidden_states)
            v, _ = self.v(hidden_states)

            if input_metadata.attn_bias is None:
                position_bias = self.compute_bias(
                    1 if input_metadata.is_prompt else context_len,
                    (context_len + block_size - 1) // block_size * block_size
                ).repeat(batch_size, 1, 1, 1)
                # print("position_bias shape", position_bias.shape)
                # print("position_bias", position_bias)
                input_metadata.attn_bias = position_bias[:, :, -seq_len:, :].contiguous()
                # print("input_metadata.attn_bias shape", input_metadata.attn_bias.shape)
                # print("input_metadata.attn_bias", input_metadata.attn_bias)

            key_cache, value_cache = kv_cache

            attn_output = self.attn(q, k, v, key_cache, value_cache, input_metadata)

        else:
            # print("cross attention!")
            # Cross attention

            key_cache, value_cache = kv_cache
            if input_metadata.is_prompt:
                assert encoder_hidden_states is not None
                k, _ = self.k(encoder_hidden_states)
                v, _ = self.v(encoder_hidden_states)
                # print("k shape", k.shape)
                # for i in range(k.shape[0]):
                #     for j in range(k.shape[1]):
                        # print(f"key at batch {i} and pos {j}: ", k[i, j, :].reshape(1, 8, 64))
                attn_output = self.attn(q, k, v, key_cache, value_cache, input_metadata)
            else:
                attn_output = self.attn(
                    q, None, None, key_cache, value_cache, input_metadata
                )

        attn_output, _ = self.o(attn_output)
        return attn_output


class T5LayerSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        has_relative_attention_bias,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            is_cross=False,
            has_relative_attention_bias=has_relative_attention_bias,
            linear_method=linear_method,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        # print("self attention input shape: ", normed_hidden_states.shape)
        # print("self_attention input: ", normed_hidden_states)
        attention_output = self.SelfAttention(
            hidden_states=normed_hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            encoder_hidden_states=None,
        )
        # print("self attention output shape: ", attention_output.shape)
        # print("self_attention output: ", attention_output)
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5LayerCrossAttention(nn.Module):
    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config,
            is_cross=True,
            has_relative_attention_bias=False,
            linear_method=linear_method,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        # print("cross attention input shape: ", normed_hidden_states.shape)
        # print("cross_attention input: ", normed_hidden_states)
        attention_output = self.EncDecAttention(
            hidden_states=normed_hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            encoder_hidden_states=encoder_hidden_states,
        )
        # print("cross attention output shape: ", attention_output.shape)
        # print("cross_attention output: ", attention_output)
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5Block(nn.Module):
    def __init__(
        self,
        config,
        has_relative_attention_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                linear_method=linear_method,
            )
        )
        if self.is_decoder:
            self.layer.append(
                T5LayerCrossAttention(config, linear_method=linear_method)
            )

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        hidden_states = self.layer[0](
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        if self.is_decoder:
            hidden_states = self.layer[1](
                hidden_states,
                kv_cache=kv_cache,
                input_metadata=input_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        return hidden_states


class T5Stack(nn.Module):
    def __init__(
        self,
        config: T5Config,
        embed_tokens: torch.Tensor,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.embed_tokens = embed_tokens

        self.block = nn.ModuleList(
            [
                T5Block(
                    config,
                    has_relative_attention_bias=(i == 0),
                    linear_method=linear_method,
                )
                for i in range(config.num_layers)
            ]
        )

        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # print("input_ids: ", input_ids)
        hidden_states = self.embed_tokens(input_ids)

        # print("hidden_states shape: ", hidden_states.shape)
        # print("hidden_states: ", hidden_states)
        for i, layer_module in enumerate(self.block):
            kv_cache = kv_caches[i] if self.is_decoder else None

            layer_outputs = layer_module(
                hidden_states,
                kv_cache=kv_cache,
                input_metadata=input_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )

            hidden_states = layer_outputs

        hidden_states = self.final_layer_norm(hidden_states)
        # if encoder_hidden_states is not None:
            # print("hidden_states shape:" , hidden_states.shape)
            # print("encoder_hidden_states shape:" , encoder_hidden_states.shape)
        #     # Attach encoder hidden states
        #     hidden_states = torch.cat(
        #         [encoder_hidden_states, hidden_states], dim=1
        #     )
        # print("final_hidden_states shape: ", hidden_states.shape)
        # print("final_hidden_states: ", hidden_states)
        return hidden_states


class T5ForConditionalGeneration(nn.Module):
    def __init__(
        self, config: T5Config, linear_method: Optional[LinearMethodBase] = None
    ):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared, linear_method)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared, linear_method)

        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # print("input_ids shape: ", input_ids.shape)
        # print("input_ids: ", input_ids)
        # print("input_metadata: ", input_metadata)
        if input_metadata.is_prompt:
            # prompt run, need to run encoder once
            hidden_states = self.encoder(input_ids, kv_caches, input_metadata, None)
            # Clear the attention bias
            input_metadata.attn_bias = None
            batch_size = input_ids.shape[0]
            input_ids = (
                torch.ones(batch_size, 1, dtype=torch.long)
                * self.config.decoder_start_token_id
            ).cuda()

        else:
            hidden_states = None

        if kv_caches[0][0] is not None:  # Skip decoder for profiling run
            hidden_states = self.decoder(
                input_ids, kv_caches, input_metadata, hidden_states
            )

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.model_dim**-0.5)

        return hidden_states

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata):
        # logger.info(f"decoder_outputs: {decoder_outputs}")
        next_tokens = self.sampler(self.shared.weight, hidden_states, sampling_metadata)
        # logger.info(f"next_tokens: {next_tokens}")
        return next_tokens

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "EncDecAttention.relative_attention_bias" in name:
                continue

            assert name in params_dict, f"{name} not in params_dict"
            param = params_dict[name]
            assert param.shape == loaded_weight.shape, (
                f"{name} shape mismatch between model and checkpoint: "
                f"{param.shape} != {loaded_weight.shape}"
            )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
