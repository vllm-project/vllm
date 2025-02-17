# Derived from T5 implementation posted on HuggingFace; license below:
#
# coding=utf-8
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

import math
from typing import List, Optional, Tuple, Union, Set, Iterable
import re

import torch
from torch import nn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.attention.layer import Attention, AttentionType, AttentionMetadata
from vllm.config import CacheConfig
from transformers import T5Config
from vllm.attention.backends.xformers import XFormersMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import get_sampler, SamplerOutput
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata
from .utils import maybe_prefix


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states)->torch.Tensor:
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
    def __init__(self, config: T5Config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False, quant_config=quant_config)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states)->torch.Tensor:
        hidden_states, _ = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        # if (
        #     isinstance(self.wo.weight, torch.Tensor)
        #     and hidden_states.dtype != self.wo.weight.dtype
        #     and self.wo.weight.dtype != torch.int8
        # ):
        #     hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.wi_0 = ColumnParallelLinear(config.d_model, config.d_ff, bias=False, quant_config=quant_config)
        self.wi_1 = ColumnParallelLinear(config.d_model, config.d_ff, bias=False, quant_config=quant_config)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False, quant_config=quant_config)
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states)->torch.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states)[0])
        hidden_linear, _ = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear

        # TODO
        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        # if (
        #     isinstance(self.wo.weight, torch.Tensor)
        #     and hidden_states.dtype != self.wo.weight.dtype
        #     and self.wo.weight.dtype != torch.int8
        # ):
        #     hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config, quant_config=quant_config)
        else:
            self.DenseReluDense = T5DenseActDense(config, quant_config=quant_config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states)->torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states


class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        attn_type: AttentionType, 
        has_relative_attention_bias=False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ):
        super().__init__()
        self.prefix = prefix
        self.attn_type =  attn_type
        # Cross-attention has no relative pos encoding anyway
        self.is_decoder = attn_type == AttentionType.DECODER
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        # No GQA in t5.
        self.n_kv_heads = self.n_heads

        self.qkv_proj = QKVParallelLinear(self.d_model, self.d_model // self.n_heads, self.n_heads, self.n_kv_heads, bias=False, quant_config=quant_config)

        # TODO refactor in utils shared with bart
        # if self.total_num_kv_heads >= tp_world_size:
        #     # Number of KV heads is greater than TP size, so we partition
        #     # the KV heads across multiple tensor parallel GPUs.
        #     assert self.total_num_kv_heads % tp_world_size == 0
        # else:
        #     # Number of KV heads is less than TP size, so we replicate
        #     # the KV heads across multiple tensor parallel GPUs.
        #     assert tp_world_size % self.total_num_kv_heads == 0
        # self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)

        # NOTE (NickLucche) T5 employs a scaled weight initialization scheme 
        # instead of scaling attention scores directly.
        self.attn = Attention(self.n_heads, config.d_kv, 1.0, cache_config=cache_config, quant_config=quant_config, prefix=f"{prefix}.attn")

        # Only the first SelfAttention block in encoder decoder has this 
        # embedding layer, the others re-use its output.
        if self.has_relative_attention_bias:
            print(f"EMBEDDING {self.relative_attention_num_buckets} TO {self.n_heads}, max dist {self.relative_attention_max_distance}")
            self.relative_attention_bias = VocabParallelEmbedding(self.relative_attention_num_buckets,\
                                                                   self.n_heads, org_num_embeddings=self.relative_attention_num_buckets, quant_config=quant_config)
            # self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            self.d_model,
            bias=False,
            quant_config=quant_config,
        )

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
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
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
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
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None)->torch.Tensor:
        """Compute binned relative position bias"""
        # TODO possible tp issue?
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # max_seq_len, nh
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        x = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor, # (num_tokens, d_model)
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    )->torch.Tensor:
        # TODO auto-selection of xformers backend when t5 is detected 
        assert isinstance(attn_metadata, XFormersMetadata)
        is_profile_run = kv_cache.numel() == 0
        if not is_profile_run:
            # TODO xformers only
            block_size = kv_cache.shape[2] // self.inner_dim
        num_seqs = len(attn_metadata.seq_lens) if attn_metadata.seq_lens else len(attn_metadata.encoder_seq_lens)
        qkv, _ = self.qkv_proj(hidden_states)
        # Projection of 'own' hidden state (self-attention). No GQA here. 
        q, k, v = qkv.split(self.inner_dim, dim=-1)

        # NOTE (NickLucche) Attn bias is computed once per encoder or decoder
        # forward, on the first call to T5Attention.forward. Subsequent 
        # *self-attention* layers will re-use it.
        # TODO func should be in backend interface
        from vllm.attention.backends.xformers import _get_attn_bias, _set_attn_bias
        attn_bias = _get_attn_bias(attn_metadata, self.attn_type)
        if self.attn_type == AttentionType.ENCODER_DECODER:
            # Projection of encoder's hidden states, cross-attention.
            if encoder_hidden_states is None:
                # Decode phase, kv already cached
                assert attn_metadata.num_prefills == 0
                k = None
                v = None
            else:
                assert attn_metadata.num_prefills > 0
                # Prefill phase (first decode forward), caching kv
                qkv_enc, _ = self.qkv_proj(encoder_hidden_states)
                _, k, v = qkv_enc.split(self.inner_dim, dim=-1)
            # No custom attention bias must be set when running cross attn. 
            assert attn_bias is None
        # Skip when profiling.
        # FIXME should be enabled on profiling run to assess memory of bias.
        # TODO NOT compatible with CP here, assumes homogeneous batch
        elif self.has_relative_attention_bias and not is_profile_run:
            assert attn_bias is None # to be recomputed
            # Self-attention. T5 relative positional encoding. 
            # Compute bias based on longest sequence in batch. Biases for
            # shorter sequences are subsets of the longest.
            if self.attn_type == AttentionType.ENCODER:
                seq_len = attn_metadata.max_encoder_seq_len
            else:
                # Decoder can receive both prefill and decoding requests  
                seq_len = attn_metadata.max_prefill_seq_len if attn_metadata.prefill_metadata else attn_metadata.max_decode_seq_len
            block_aligned_seq_len = (seq_len + block_size - 1) // block_size * block_size

            # TODO xformers-specific, attention bias are to be provided as a list.
            align_to = 8
            print("IN", hidden_states.shape)
            # TODO chunked prefill
            # what I want: (num_seqs, NH, L, L_pad) for prefill, (num_seqs, NH, 1, L_pad) for decodes
            if self.attn_type == AttentionType.ENCODER:
                # NOTE seq padding needed for xformers!
                # HINT: To use an `attn_bias` with a sequence length that is not a multiple of 8, you need to ensure memory is aligned by slicing a bigger tensor. Example: use `attn_bias = torch.zeros([1, 1, 5, 8])[:,:,:,:5]` instead of `torch.zeros([1, 1, 5, 5])`
                padded_seq_len = (seq_len + align_to - 1) // align_to * align_to
                # FIXME blockdiagonal mask with tensor bias??
                pb = self.compute_bias(seq_len, padded_seq_len).repeat(num_seqs, 1, 1, 1)
                print(f"[{self.prefix}][ENCODING, xformers aligned] Seq len compute bias/shape", seq_len, padded_seq_len)
                # print("SEQ LEN", attn_metadata.encoder_seq_lens, pb.shape) xformers needs a list, one matrix per sequence
                attn_metadata.encoder_attn_bias = [p[:, :sq, :sq].unsqueeze(0) for p, sq in zip(pb, attn_metadata.encoder_seq_lens)]
                print("Bias shape", attn_metadata.encoder_attn_bias[0].shape)
            elif attn_metadata.decode_metadata is None: # TODO join with statement above
                # NOTE first decoder step, prefill: seq len here is usually 1, but one can prepend different start tokens prior to generation. XFormers is used.    
                print(f"[{self.prefix}][DECODER but xformers prefill]", attn_metadata.max_decode_seq_len, attn_metadata.max_prefill_seq_len)
                padded_seq_len = (seq_len + align_to - 1) // align_to * align_to
                # position_bias = self.compute_bias(seq_len, seq_len).repeat(num_seqs, 1, 1, 1)
                # this is always 1 (seqlen) but needs filling!!
                # position_bias = self.compute_bias(seq_len, align_to).repeat(num_seqs, 1, 1, 1)
                position_bias = self.compute_bias(seq_len, padded_seq_len).repeat(num_seqs, 1, 1, 1)
                print(f"[{self.prefix}][DECODER but xformers prefill] Seq len compute bias/shape", seq_len, position_bias.shape)
                # ->align
                # position_bias = position_bias.repeat(1, 1, align_to, align_to)
                # print("DECODER RUN BUT NO METADATA", position_bias.shape)
                # TODO debug, can it be removed?
                # position_bias[:, :, 1:, 1:] = torch.finfo(position_bias.dtype).min
                # attn_metadata.attn_bias = [pb[None, :, :seq_len, :seq_len] for pb in position_bias]
                attn_metadata.attn_bias = [pb[None, :, :seq_len, :seq_len] for pb, sq in zip(position_bias, attn_metadata.seq_lens)]
                # attn_metadata.attn_bias = [position_bias[:, :, :1, :seq_len]]
                print("[DECODER but xformers prefill] Bias shape", attn_metadata.attn_bias[0].shape)
            else:
                # Repeat along dim0: (num_seqs, n_heads, 1, L)
                # TODO (NickLucche): allow single bias for whole batch to avoid extra-copy.
                # TODO this needs to be block aligned!!
                # position_bias = self.compute_bias(seq_len, block_aligned_seq_len).repeat(num_seqs, 1, 1, 1)
                position_bias = self.compute_bias(1, block_aligned_seq_len).repeat(num_seqs, 1, 1, 1)
                # position_bias = self.compute_bias(seq_len, seq_len).repeat(num_seqs, 1, 1, 1)
                print(f"[{self.prefix}][DECODING] Seq len compute bias/shape", seq_len, seq_len)
                # position_bias[:, :, seq_len:, seq_len:] = torch.finfo(position_bias.dtype).min
                attn_metadata.attn_bias = [position_bias]
                # attn_metadata.attn_bias = [position_bias[:, :, :seq_len, :seq_len]]
                print("Bias shape", attn_metadata.attn_bias[0].shape)
            # TODO set attn bias
        elif not self.has_relative_attention_bias and not is_profile_run:
            # Encoder/Decoder Self-Attention Layer, attn bias already cached.
            assert attn_bias is not None

            # masking extra bias entries is done in the kernel
            # mask = (seq_range >= prompt_lens).unsqueeze(1).unsqueeze(3)
            # position_bias *= mask
            # xformers masking
                # for i in range(batch_size):
                #     input_metadata.attn_bias[
                #         i, :, :,
                #         input_metadata.prompt_lens[i]:, ] = torch.finfo(
                #             input_metadata.attn_bias.dtype).min

        attn_output = self.attn(q,
                        k,
                        v,
                        kv_cache,
                        attn_metadata,
                        attn_type=self.attn_type)
        # if not self.is_decoder:
            # print("\n\nSAVING\n\n")
            # torch.save(attn_output, "tensor_vllm.pth")
            # FIXME all equal until the next line
        output, _ = self.out_proj(attn_output)
        return output

class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, cache_config: Optional[CacheConfig] = None, 
        quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "", ):
        super().__init__()
        self.SelfAttention = T5Attention(config, AttentionType.DECODER if "decoder" in prefix else AttentionType.ENCODER, 
                                         has_relative_attention_bias=has_relative_attention_bias, 
                                         cache_config=cache_config, 
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.SelfAttention")
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    )->torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            hidden_states=normed_hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            encoder_hidden_states=None,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, cache_config: Optional[CacheConfig] = None, 
        quant_config: Optional[QuantizationConfig] = None,
                 prefix: str=""):
        super().__init__()
        self.EncDecAttention = T5Attention(config, AttentionType.ENCODER_DECODER, has_relative_attention_bias=False, cache_config=cache_config, quant_config=quant_config,prefix=f"{prefix}.EncDecAttention")
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    )->torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            hidden_states=normed_hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, config: T5Config, is_decoder: bool, has_relative_attention_bias=False, 
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,

                 prefix: str = ""):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attn = T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, cache_config=cache_config, quant_config=quant_config,prefix=f"{prefix}.self_attn")
        
        if self.is_decoder:
            self.cross_attn = T5LayerCrossAttention(config, cache_config=cache_config, quant_config=quant_config, prefix=f"{prefix}.cross_attn")

        self.ffn = T5LayerFF(config, quant_config=quant_config)
        # TODO remove
        self.prefix = prefix

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor]=None,
    )->torch.Tensor:

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # print(f"[{self.prefix}] HIDDEN inblock", hidden_states.shape, hidden_states.mean())
        if self.is_decoder:
            hidden_states = self.cross_attn(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )
        # print(f"[{self.prefix}] HIDDEN inblock", hidden_states.shape, hidden_states.mean())

        # Apply Feed Forward layer
        hidden_states = self.ffn(hidden_states)
        # print(f"[{self.prefix}] HIDDEN inblock", hidden_states.shape, hidden_states.mean())
        return hidden_states


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, is_decoder: bool, n_layers: int, embed_tokens=None, 
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
                 prefix: str=""):
        super().__init__()
        self.embed_tokens = embed_tokens
        # Only the first block has relative positional encoding.
        self.blocks = nn.ModuleList(
            [T5Block(config, is_decoder=is_decoder, has_relative_attention_bias=i==0,
                      cache_config=cache_config,quant_config=quant_config,
                      prefix=f"{prefix}.blocks.{i}") for i in range(n_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.prefix = prefix # TODO remove


    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor]=None
    )-> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for idx, block in enumerate(self.blocks):
            # print(f"[{self.prefix}] HIDDEN", hidden_states.shape, hidden_states.mean())
            hidden_states = block(
                hidden_states=hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )
        # print(f"[{self.prefix}] HIDDEN out", hidden_states.shape, hidden_states.mean())
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class T5Model(nn.Module):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, *, vllm_config: VllmConfig, prefix:str=""):
        super().__init__()
        config: T5Config = vllm_config.model_config.hf_config
        # TODO lora
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.padding_idx = config.pad_token_id # TODO decoding token
        self.shared = VocabParallelEmbedding(config.vocab_size, config.d_model, org_num_embeddings=config.vocab_size)

        self.encoder = T5Stack(config, False, config.num_layers, self.shared, cache_config=cache_config,quant_config=quant_config,prefix=f"{prefix}.encoder")
        # assert config.num_layers == config.num_decoder_layers
        self.decoder = T5Stack(config, True, config.num_decoder_layers, self.shared, cache_config=cache_config,quant_config=quant_config, prefix=f"{prefix}.decoder")

    def get_input_embeddings(self, input_ids: torch.Tensor)->torch.Tensor:
        return self.shared(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor, 
        encoder_input_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata
    ) ->torch.Tensor:
        encoder_hidden_states = None

        if encoder_input_ids.numel() > 0:
            # Run encoder attention if a non-zero number of encoder tokens
            # are provided as input: on a regular generate call, the encoder
            # runs once, on the prompt. Subsequent decoder calls re-use output
            # `encoder_hidden_states`.
            print("Running on encoder input ids", encoder_input_ids.shape, "on this many sequences", len(attn_metadata.encoder_seq_lens))
            encoder_hidden_states = self.encoder(input_ids=encoder_input_ids,
                                                 kv_caches=kv_caches,
                                                 attn_metadata=attn_metadata)
            # Clear attention bias state.
            attn_metadata.attn_bias = None
            attn_metadata.encoder_attn_bias = None
            attn_metadata.cross_attn_bias = None
            print("ENC OUT HIDDEN", encoder_hidden_states.shape, encoder_hidden_states.mean())
        print("Running on decoder input ids (0 as input token)", input_ids)
        # decoder outputs consists of
        # (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata)
        print("DEC OUT HIDDEN", decoder_outputs.shape, decoder_outputs.mean())
        return decoder_outputs


class T5ForConditionalGeneration(nn.Module):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, *, vllm_config: VllmConfig, prefix:str=""):
        super().__init__()
        config: T5Config = vllm_config.model_config.hf_config
        self.model_dim = config.d_model
        self.config = config
        self.unpadded_vocab_size = config.vocab_size
        # TODO
        # if lora_config := vllm_config.lora_config:
        #     self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.model = T5Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # Although not in config, this is the default for hf models.
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.shared
            # in transformers this is smt more explicit, as in (after load)
            # self.lm_head.weight = self.model.shared.weight
        else:
            self.lm_head = ParallelLMHead(self.unpadded_vocab_size, config.d_model, org_num_embeddings=config.vocab_size)

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = get_sampler()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.model_dim**-0.5)
        print("hidden states input", hidden_states.shape)
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
    
    def get_input_embeddings(self, input_ids: torch.Tensor)->torch.Tensor:
        return self.model.shared(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        *,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        **kwargs,
) -> torch.Tensor:
        return self.model(input_ids, encoder_input_ids, kv_caches, attn_metadata)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]]
    ):
        model_params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        renamed_reg = [
            (re.compile(r'block\.(\d+)\.layer\.0'), r'blocks.\1.self_attn'),
            (re.compile(r'decoder.block\.(\d+)\.layer\.1'), r'decoder.blocks.\1.cross_attn'),
            (re.compile(r'decoder.block\.(\d+)\.layer\.2'), r'decoder.blocks.\1.ffn'),
            # encoder has no cross-attn, but rather self-attention+ffn.
            (re.compile(r'encoder.block\.(\d+)\.layer\.1'), r'encoder.blocks.\1.ffn'),
            (re.compile(r'\.o\.'), r'.out_proj.'),
        ]
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj.", ".q.", "q"),
            (".qkv_proj.", ".k.", "k"),
            (".qkv_proj.", ".v.", "v")
        ]

        for name, loaded_weight in weights:
            # No relative position attn bias on cross attention. 
            if name == "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight":
                continue

            # Handle some renaming
            for reg in renamed_reg:
                name = re.sub(*reg, name)

            top_module, _ = name.split('.', 1)
            if top_module != 'lm_head':
                name = f"model.{name}"

            # Split q/k/v layers to unified QKVParallelLinear
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = model_params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Not a q/k/v layer.
                param = model_params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params