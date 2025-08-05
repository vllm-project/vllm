# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, TypedDict, Union, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import transformers

from transformers import AutoTokenizer, BatchFeature
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from vllm.config import VllmConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.linear import (
    ReplicatedLinear, RowParallelLinear, ColumnParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (AudioProcessorItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2PreTrainedModel,
    apply_rotary_pos_emb
)
from ...transformers_utils.processors import KimiAudioProcessor, WhisperEncoder

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

from packaging import version

assert version.parse(transformers.__version__) >= version.parse("4.34.1")

from ...transformers_utils.tokenizers import Glm4Tokenizer
import torch.nn.functional as F
if version.parse(transformers.__version__) >= version.parse("4.35.0"):
    from transformers.utils import is_flash_attn_2_available as is_flash_attn_available
else:
    from transformers.utils import is_flash_attn_available

if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
else:
    raise RuntimeError("flash attention must be installed")

from vllm.logger import init_logger

logger = init_logger(__name__)

def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(query_layer, key_layer, value_layer, padding_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
    num_heads = query_layer.shape[2]

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        padding_mask = padding_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
            query_layer, padding_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MoonshotAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: KimiAudioConfig, prefix: str = ""): # type: ignore
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = ColumnParallelLinear(
            self.hidden_size, 
            self.num_heads * self.head_dim,
            bias=True,
            prefix=f"{prefix}.q_proj"
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=True,
            prefix=f"{prefix}.k_proj"
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=True,
            prefix=f"{prefix}.v_proj"
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias=False,
            prefix=f"{prefix}.o_proj"
        )

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos[position_ids]
        sin = sin[position_ids]
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            padding_mask,
            q_len,
            dropout=dropout_rate,
        )

        if input_dtype == torch.float32:
            attn_output = attn_output.to(torch.float32)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = _upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

        return attn_output


class MoonshotDecoderLayer(nn.Module):
    def __init__(self, config: KimiAudioConfig, prefix: str = ""): # type: ignore
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        logger.warning_once("using normal flash attention")
        self.self_attn = MoonshotAttention(config=config, prefix=f"{prefix}.self_attn")

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class VQAdaptor(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.lin1 = ReplicatedLinear(
            config.kimia_adaptor_input_dim, 
            config.hidden_size,
            bias=True, 
            prefix=f"{prefix}.layers.0"
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.0)
        self.lin2 = ReplicatedLinear(
            config.hidden_size, 
            config.hidden_size,
            bias=True, 
            prefix=f"{prefix}.layers.3"
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps,
            bias=True
        )

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return self.norm(x)
    

class MoonshotKimiaModel(Qwen2PreTrainedModel):
    config_class = KimiAudioConfig # type: ignore

    def __init__(self, config: KimiAudioConfig, prefix: str = ""): # type: ignore
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.kimia_mimo_transformer_from_layer_index = (
            config.kimia_mimo_transformer_from_layer_index
        )

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [MoonshotDecoderLayer(config, prefix=f"{prefix}.layers.{layer_idx}") 
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # extra 1B audio transformers
        self.mimo_layers = nn.ModuleList(
            [MoonshotDecoderLayer(config, prefix=f"{prefix}.mimo_layers.{idx}") 
             for idx in range(config.kimia_mimo_layers)]
        )
        self.mimo_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_whisper_feature = config.use_whisper_feature
        if self.use_whisper_feature:
            self.vq_adaptor = VQAdaptor(config, prefix=f"{prefix}.vq_adaptor")
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            # shape: batch, seq_len, hidden_size
            input_ids = input_ids.to(self.embed_tokens.weight.device)
            text_input_ids = text_input_ids.to(self.embed_tokens.weight.device)
            audio_emb = self.embed_tokens(input_ids)
            if self.use_whisper_feature and whisper_input_feature is not None:
                if not isinstance(whisper_input_feature, list):
                    whisper_input_feature = whisper_input_feature.squeeze(0)
                    whisper_input_feature = [whisper_input_feature]

                media_start_idx = (input_ids == self.kimia_media_begin).nonzero()
                media_end_idx = (input_ids == self.kimia_media_end).nonzero()
                # shape: batch, seq_len, hidden_size
                whisper_input_dim = whisper_input_feature[0].shape[-1]
                whisper_dtype = whisper_input_feature[0].dtype
                expanded_whisper = (
                    torch.zeros(audio_emb.shape[1], whisper_input_dim)
                    .to(self.embed_tokens.weight.device)
                    .to(whisper_dtype)
                )
                for (seg_idx, start_idx), (_, end_idx) in zip(
                    media_start_idx, media_end_idx
                ):
                    # assert whisper_emb.shape[1] == end_idx - (start_idx + 1)

                    feat_len = end_idx - (start_idx + 1)
                    whisper_input_feature_i = whisper_input_feature[seg_idx].squeeze(0)
                    assert feat_len == is_continuous_mask[seg_idx].sum()
                    expanded_whisper[start_idx + 1 : end_idx, :] = (
                        whisper_input_feature_i[:feat_len, :]
                    )

                expanded_whisper = expanded_whisper.unsqueeze(0)
                whisper_emb = self.vq_adaptor(
                    expanded_whisper.transpose(0, 1)
                ).transpose(0, 1)
                is_continuous_mask = is_continuous_mask.to(self.embed_tokens.weight.device)
                whisper_emb = whisper_emb.to(self.embed_tokens.weight.device)
                whisper_emb = whisper_emb * is_continuous_mask[:, :, None]

                encoder_input_addwith_discrete_token = (
                    audio_emb + whisper_emb
                ) * torch.sqrt(
                    torch.tensor(
                        2.0, dtype=whisper_emb.dtype, device=self.embed_tokens.weight.device
                    )
                )
                audio_emb = (
                    audio_emb * (~is_continuous_mask[:, :, None])
                    + encoder_input_addwith_discrete_token
                    * is_continuous_mask[:, :, None]
                )
            if text_input_ids is not None and text_input_ids.sum() != 0:
                inputs_embeds = audio_emb + self.embed_tokens(text_input_ids)
            else:
                inputs_embeds = audio_emb
        # embed positions
        # TODO kill attention_mask for prefill
        padding_mask = attention_mask

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            hidden_states = layer_outputs[0]
            if idx == self.kimia_mimo_transformer_from_layer_index:
                mimo_hidden_states = hidden_states.clone()

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # apply audio transformer layers
        for idx, decoder_layer in enumerate(self.mimo_layers):
            if output_hidden_states:
                all_hidden_states += (mimo_hidden_states,)

            past_key_value = (
                past_key_values[idx + len(self.layers)]
                if past_key_values is not None
                else None
            )
            layer_outputs = decoder_layer(
                mimo_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            mimo_hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        mimo_hidden_states = self.mimo_norm(mimo_hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (mimo_hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    mimo_hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=(hidden_states, mimo_hidden_states),
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MoonshotKimiaForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight", "mimo_output.weight"]
    config_class = KimiAudioConfig # type: ignore

    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.model = MoonshotKimiaModel(config, prefix=f"{prefix}.model")
        self.vocab_size = config.vocab_size
        self.lm_head = ReplicatedLinear(config.hidden_size, config.vocab_size, 
                                        bias=False, prefix=f"{prefix}.lm_head")
        self.mimo_output = ReplicatedLinear(config.hidden_size, config.vocab_size, 
                                            bias=False, prefix=f"{prefix}.mimo_output")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

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
        outputs = self.model(
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
    
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)


@dataclass
class ExtraTokens:
    msg_end: int
    user_msg_start: int
    assistant_msg_start: int

    media_begin: int
    media_end: int

    kimia_text_blank: int
    kimia_text_eos: int

    kimia_user_msg_start: int
    kimia_assistant_msg_start: int

    kimia_speech_ct_id: int
    kimia_speech_ctd_id: int

    pad: int


def instantiate_extra_tokens(tokenizer):
    if hasattr(tokenizer, "special_tokens"):
        map_fn = lambda x: tokenizer.special_tokens[x]
    elif hasattr(tokenizer, "convert_tokens_to_ids"):
        map_fn = lambda x: tokenizer.convert_tokens_to_ids(x)
    else:
        raise ValueError(f"Invalid tokenizer type: {type(tokenizer)}")
    return ExtraTokens(
        msg_end=map_fn("<|im_msg_end|>"),  # 0
        user_msg_start=map_fn("<|im_user_msg_start|>"),  # 1
        assistant_msg_start=map_fn("<|im_assistant_msg_start|>"),  # 2
        media_begin=map_fn("<|im_media_begin|>"),  # 13
        media_end=map_fn("<|im_media_end|>"),  # 15
        kimia_text_blank=map_fn("<|im_kimia_text_blank|>"),  # 18
        kimia_text_eos=map_fn("<|im_kimia_text_eos|>"),  # 19
        kimia_user_msg_start=map_fn("<|im_kimia_user_msg_start|>"),  # 22
        kimia_assistant_msg_start=map_fn("<|im_kimia_assistant_msg_start|>"),  # 23
        kimia_speech_ct_id=map_fn("<|im_kimia_speech_ct_id|>"),  # 27
        kimia_speech_ctd_id=map_fn("<|im_kimia_speech_ctd_id|>"),  # 28
        pad=tokenizer.pad_id,
    )


# === Audio Inputs === #
class KimiAudioInputs(TypedDict):
    whisper_input_feature: List[torch.Tensor]
    """Shape: `(num_audios, seq_len, feature_dim)`"""
    
    is_continuous_mask: torch.Tensor
    """Shape: `(num_audios, seq_len)`"""

    text_input_ids: Optional[torch.Tensor]


# === Processing Info === #
class KimiAudioProcessingInfo(BaseProcessingInfo):
    
    def get_hf_config(self) -> KimiAudioConfig: # type: ignore
        return self.ctx.get_hf_config(KimiAudioConfig) # type: ignore
    
    def get_tokenizer(self) -> AutoTokenizer:
        return super().get_tokenizer()
    
    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        return self.ctx.get_hf_processor(KimiAudioProcessor, **kwargs)
    
    def get_audio_tokenizer(self) -> Any:
        audio_tokenizer = Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
        return audio_tokenizer.to(torch.cuda.current_device())
    
    def get_feature_extractor(
            self,
            *,
            sampling_rate: Optional[int] = None
    ) -> WhisperEncoder:
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperEncoder)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # no limits on count of audio
        return {"audio": None}
    
    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        """每个音频项最多产生多少tokens"""
        # 基于whisper特征长度估算
        # 通常30秒音频 -> ~750 tokens
        return {"audio": 750}


# === Dummy Inputs Builder === #
class KimiAudioDummyInputsBuilder(BaseDummyInputsBuilder[KimiAudioProcessingInfo]):
    
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        
        tokenizer = self.info.get_tokenizer()
        config = self.info.get_hf_config()
        
        # 使用Kimi-Audio的特殊tokens
        # 从config中获取特殊token ids
        media_begin = tokenizer.decode([config.kimia_media_begin])
        media_end = tokenizer.decode([config.kimia_media_end])
        media_pad = tokenizer.decode([config.kimia_media_pad])
        
        # 构建dummy文本
        dummy_segments = []
        for i in range(num_audios):
            # 模拟音频占位符
            # media_begin + 一些padding + media_end
            audio_placeholder = f"{media_begin}{media_pad * 100}{media_end}"
            dummy_segments.append(audio_placeholder)
        
        # 添加一些文本提示
        if num_audios > 0:
            return "Please transcribe the following audio: " + " ".join(dummy_segments)
        return "Hello, how can I help you?"
    
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        
        # 创建dummy音频数据
        # Whisper需要16kHz采样率的音频
        audio_duration = 3.0  # 3秒dummy音频
        sample_rate = 16000
        audio_len = int(audio_duration * sample_rate)
        
        return {
            "audio": self._get_dummy_audios(
                length=audio_len, 
                num_audios=num_audios
            )
        }


# === Multi-Modal Processor === #
class KimiAudioMultiModalProcessor(BaseMultiModalProcessor[KimiAudioProcessingInfo]):

    @property
    def whisper_processor(self, **kwargs):
        if self._whisper_processor is None:
            self._whisper_processor = WhisperEncoder()
        return self._whisper_processor
    
    @property
    def audio_tokenizer(self, **kwargs):
        if self._audio_tokenizer is None:
            self._audio_tokenizer = self.info.get_audio_tokenizer()
        return self._audio_tokenizer
    
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)
    
    def _process_audio_to_features(
        self,
        audio_data: np.ndarray,
        sampling_rate: int = 16000
    ) -> torch.Tensor:
        """将音频转换为Whisper特征"""
        # 使用Whisper处理器提取特征
        inputs = self.whisper_processor(
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        # 获取mel特征
        mel_features = inputs.input_features
        
        # 这里需要通过Whisper编码器获取特征
        # 在实际实现中，这会调用WhisperEncoder
        # 返回shape: (1, seq_len, feature_dim)
        return mel_features
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data.get("audios", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), return_tensors="pt")
        
        feature_extractor = self.info.get_feature_extractor()
        mm_kwargs = dict(
            **mm_kwargs,
            Sampling_rate=feature_extractor.sampling_rate,
        )
        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
    
    def _create_continuous_mask(
        self, 
        input_ids: torch.Tensor,
        config: KimiAudioConfig # type: ignore
    ) -> torch.Tensor:
        """创建标记音频位置的mask"""
        # 找到media_begin和media_end之间的位置
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        media_begin_indices = (input_ids == config.kimia_media_begin).nonzero(as_tuple=True)[0]
        media_end_indices = (input_ids == config.kimia_media_end).nonzero(as_tuple=True)[0]
        
        for begin, end in zip(media_begin_indices, media_end_indices):
            mask[begin+1:end] = True
        
        return mask.unsqueeze(0)  # Add batch dimension
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_nums = hf_inputs.get("audio", {}).get("nums", [])
        if audio_nums > self.info.get_supported_mm_limits():
            raise ValueError(
                f"Audio count {audio_nums} exceeds the supported limit "
                f"{self.info.get_supported_mm_limits()}"
            )
        return dict(
            whisper_input_feature=MultiModalFieldConfig.batched("audio"),
            is_continuous_mask=MultiModalFieldConfig.batched("audio"),
            text_input_ids=MultiModalFieldConfig.batched("audio"),
        )
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # 获取特殊token
        media_begin_token = tokenizer.decode([vocab.kimia_media_begin])
        media_pad_token = tokenizer.decode([vocab.kimia_media_pad])
        media_end_token = tokenizer.decode([vocab.kimia_media_end])
        
        def get_replacement_kimi_audio(item_idx: int):
            # 获取这个音频的特征长度
            whisper_features = out_mm_kwargs.get("whisper_input_feature", [])
            if item_idx < len(whisper_features):
                feature_len = whisper_features[item_idx].shape[1]
            else:
                feature_len = 100  # 默认长度
            
            # 创建替换序列
            # media_begin + feature_len个media_pad + media_end
            replacement_ids = (
                [vocab.kimia_media_begin] + 
                [vocab.kimia_media_pad] * feature_len + 
                [vocab.kimia_media_end]
            )
            
            return PromptUpdateDetails.select_token_id(
                replacement_ids,
                # 使用media_pad作为embedding token
                embed_token_id=vocab.kimia_media_pad,
            )
        
        # 查找prompt中的音频占位符并替换
        return [
            PromptReplacement(
                modality="audio",
                target=f"{media_begin_token}.*?{media_end_token}",  # 正则匹配
                replacement=get_replacement_kimi_audio,
            )
        ]


# === Main Model Class === #
@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder
)
class KimiAudioForConditionalGeneration(nn.Module, SupportsMultiModal,
                               SupportsPP):
    
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<|im_media_begin|><|im_media_end|>"
        
        raise ValueError("Only audio modality is supported")
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiAudioConfig = model_config.hf_config # type: ignore
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        
        self.audio_tower = Glm4Tokenizer(config.audio_config)
        self.multi_modal_projector = WhisperEncoder()

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["MoonshotKimiaForCausalLM"],
        )
        
        self.lm_head = self.language_model.lm_head
        self.mimo_output = self.language_model.mimo_output
        
        self.unpadded_vocab_size = config.vocab_size
        self.vocab_size = config.vocab_size
        
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            self.vocab_size,
            logit_scale
        )
        
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)
    
    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)
    
    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[KimiAudioInputs]:
        whisper_input_feature = kwargs.pop('whisper_input_feature', None)
        is_continuous_mask = kwargs.pop('is_continuous_mask', None)
        
        if whisper_input_feature is None:
            return None
        
        whisper_input_feature = self._validate_and_reshape_mm_tensor(
            whisper_input_feature, 'whisper_input_feature'
        )
        
        if is_continuous_mask is not None:
            is_continuous_mask = self._validate_and_reshape_mm_tensor(
                is_continuous_mask, 'is_continuous_mask'
            )
        else:
            # Create a default continuous mask
            batch_size, seq_len = whisper_input_feature.shape[:2]
            is_continuous_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        return KimiAudioInputs(
            whisper_input_feature=whisper_input_feature,
            is_continuous_mask=is_continuous_mask
        )
    
    def _process_audio_input(self, audio_input: KimiAudioInputs) -> torch.Tensor:
        whisper_input_feature = audio_input["whisper_input_feature"]
        is_continuous_mask = audio_input["is_continuous_mask"]
        
        # Step 1: Process through Kimi-Audio's Whisper encoder
        # Input: raw audio features
        # Output: reshaped features (B, T//4, D*4)
        continuous_feature = self.audio_tower.tokenize_waveform(whisper_input_feature)
        continuous_feature = continuous_feature.reshape(
            continuous_feature.shape[0],
            int(continuous_feature.shape[1] // 4),
            continuous_feature.shape[2] * 4,
        )
        return continuous_feature
    
    def get_language_model(self) -> torch.nn.Module:
        return self.language_model
    
    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        
        processed_features = self._process_audio_input(audio_input)
        return processed_features
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            # Use Kimi-Audio's media begin token ID for merging
            kimia_media_begin_id = getattr(self.config, 'kimia_media_begin', 151661)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                kimia_media_begin_id
            )
        
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            input_ids = None
        
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds
        )
        
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        **kwargs: object,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata, **kwargs)
        return logits
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)