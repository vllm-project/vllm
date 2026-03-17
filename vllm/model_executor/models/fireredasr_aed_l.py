"""
修复后的 FireRedASR-AED 模型适配 vLLM

问题分析：
==========
vLLM 启动时报错: "This model does not support `--runner generate`"

根本原因：
- vLLM 在 ModelConfig 初始化时会检查模型是否支持 generate runner
- 检查逻辑在 /root/vllm/vllm/config/model.py:510 行：
  if self.runner_type == "generate" and not is_generative_model:
      raise ValueError("This model does not support `--runner generate`.")
  
- is_generative_model 通过 registry.is_text_generation_model() 判断
- 该方法检查模型类是否实现了 VllmModelForTextGeneration 接口

解决方案：
=========
FireRedASR 模型需要遵循 Whisper 的模式（因为两者都是音频转录模型）：

1. 继承自 nn.Module （不需要继承 VllmModelForTextGeneration，通过鸭子类型即可）
2. 实现必需的方法：
   - __init__(vllm_config, prefix=""): vLLM 标准初始化
   - forward(input_ids, positions, encoder_outputs=None): vLLM 标准前向传播
   - embed_multimodal(**kwargs): 编码音频特征
   - embed_input_ids(input_ids, ...): 获取 decoder token embeddings
   - compute_logits(hidden_states): 计算输出 logits
   
3. 设置类属性：
   - supports_transcription_only = True: 标记为仅支持转录任务

关键修改点：
==========
1. __init__ 方法签名必须是 (self, *, vllm_config, prefix="")
2. forward 方法签名必须是 (self, input_ids, positions, encoder_outputs=None, **kwargs)
3. 必须实现 compute_logits 方法（这是 is_text_generation_model 检查的关键）
4. 必须实现 embed_input_ids 方法
5. 对于多模态模型，需要实现 embed_multimodal 方法

替换步骤：
=========
将此文件复制到: /root/vllm/vllm/model_executor/models/fireredasr_aed.py

命令：
cp /root/.vscode-server/libin_test/fireredasr_aed_fixed.py /root/vllm/vllm/model_executor/models/fireredasr_aed.py
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Optional, Tuple, Literal, cast


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, BatchFeature
from transformers.utils import ModelOutput
from vllm.inputs.data import PromptType
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.config.multimodal import BaseDummyOptions
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsTranscription, MultiModalEmbeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)

from vllm.multimodal.processing import BaseDummyInputsBuilder
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.model_executor.models.whisper_utils import (
    ISO639_1_SUPPORTED_LANGS,
)
from vllm.model_executor.layers.attention import (
    Attention,
    CrossAttention,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.v1.attention.backend import (
    AttentionType,
)
# from vllm.attention.layer import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from .utils import make_layers
from vllm.model_executor.layers.activation import get_act_fn
from vllm.utils.jsontree import json_map_leaves
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.config import CacheConfig, ModelConfig, SpeechToTextConfig, VllmConfig


# Configuration class
class FireRedASRConfig(PretrainedConfig):
    """Configuration class for FireRedASR model"""
    model_type = "fireredasr_aed"

    def __init__(
        self,
        num_mel_bins: int = 80,
        vocab_size: int = 7832,
        sos_id: int = 3,
        eos_id: int = 4,
        pad_id: int = 2,
        encoder_layers: int = 16,
        decoder_layers: int = 16,
        encoder_attention_heads: int = 20,
        decoder_attention_heads: int = 20,
        d_model: int = 1280,
        residual_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
        activation_function: str = "gelu",
        max_target_positions: int = 448,
        max_source_positions: int = 1500,
        is_encoder_decoder: bool = True,
        scale_embedding: bool = False,
        **kwargs
    ):
        self.num_mel_bins = num_mel_bins
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.d_model = d_model
        self.residual_dropout = residual_dropout
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.pe_maxlen = pe_maxlen
        self.activation_function = activation_function
        self.max_target_positions = max_target_positions
        self.max_source_positions = max_source_positions
        self.is_encoder_decoder = is_encoder_decoder
        self.scale_embedding = scale_embedding
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(**kwargs)


class FireRedASRAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - nmb: Number of mel bins
        - t: Time frames (M)
    """

    input_features: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b", "nmb", "t"),
    ]


@dataclasses.dataclass
class FireRedASROutput(ModelOutput):
    """Output class for FireRedASR model"""
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ConformerEncoder(nn.Module):
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = ""
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.input_preprocessor = Conv2dSubsampling(config.num_mel_bins, config.d_model)
        self.positional_encoding = RelPositionalEncoding(config.d_model)
        self.dropout = nn.Dropout(config.residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(config.encoder_layers):
            block = RelPosEmbConformerBlock(config.d_model, config.encoder_attention_heads,
                        config.residual_dropout,
                        config.dropout_rate, config.kernel_size)
            self.layer_stack.append(block)

    def forward(self, input_features):
        input_lengths = torch.tensor([input_features.size(1)], device=input_features.device)
        padded_input_features = F.pad(input_features, (0, 0, 0, self.input_preprocessor.context - 1), 'constant', 0.0)
        src_mask = self.padding_position_is_0(padded_input_features, input_lengths)
        embed_output, input_lengths, src_mask = self.input_preprocessor(padded_input_features, src_mask)
        enc_output = self.dropout(embed_output)

        pos_emb = self.dropout(self.positional_encoding(embed_output))

        enc_outputs = []
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output = enc_layer(enc_output, pos_emb, slf_attn_mask=src_mask,
                                   pad_mask=src_mask)
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(self, padded_input, input_lengths):
        B, T, _ = padded_input.shape
        pos = torch.arange(T, device=input_lengths.device).unsqueeze(0)
        mask = pos < input_lengths.unsqueeze(1)
        return mask.unsqueeze(1).float()


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model, n_head,
                 residual_dropout=0.1,
                 dropout_rate=0.1, kernel_size=33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model,
                                             residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size,
                                         dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, slf_attn_mask=None, pad_mask=None):
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, num_mel_bins, d_model, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((num_mel_bins - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)

        self.subsampling = 4
        left_context = right_context = 3  # both exclude currect frame
        self.context = left_context + 1 + right_context  # 7

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)
        x = self.conv(x)
        N, C, T, D = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(N, T, C * D))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Tmax = 2 * max_len - 1
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T].clone().detach()
        return pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        pre_layer_norm = nn.LayerNorm(d_model)
        linear_expand = nn.Linear(d_model, d_model*4)
        nonlinear = Swish()
        dropout_pre = nn.Dropout(dropout_rate)
        linear_project = nn.Linear(d_model*4, d_model)
        dropout_post = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(pre_layer_norm,
                                 linear_expand,
                                 nonlinear,
                                 dropout_pre,
                                 linear_project,
                                 dropout_post)

    def forward(self, x):
        residual = x
        output = self.net(x)
        output = output + residual
        return output


class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size=33, dropout_rate=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model*4, kernel_size=1, bias=False)
        self.glu = F.glu
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(d_model*2, d_model*2,
                                        kernel_size, stride=1,
                                        padding=self.padding,
                                        groups=d_model*2, bias=False)
        self.batch_norm = nn.LayerNorm(d_model*2)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model*2, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)

        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)

        out = self.dropout(self.pointwise_conv2(out))
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = out.transpose(1, 2)
        return out + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        output, attn = self.attention(q, k, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn

    def forward_qkv(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_output(self, output, residual, sz_b, len_q):
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        fc_out = self.fc(output)
        output = self.dropout(fc_out)
        output = output + residual
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.0)
        self.INF = float('inf')

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        output, attn = self.forward_attention(attn, v, mask)
        return output, attn

    def forward_attention(self, attn, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)

        d_attn = self.dropout(attn)
        output = torch.matmul(d_attn, v)

        return output, attn


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__(n_head, d_model,
                         residual_dropout)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k ** 0.5)
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        # CRITICAL: Use torch.empty() instead of torch.FloatTensor()
        # torch.FloatTensor always creates on CPU, ignoring device context
        # torch.empty() respects 'with device:' context and creates on correct device
        self.pos_bias_u = nn.Parameter(torch.empty(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def _rel_shift(self, x):
        N, H, T1, T2 = x.size()
        zero_pad = torch.zeros((N, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(N, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(self, q, k, v, pos_emb, mask=None):
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn_scores = matrix_ac + matrix_bd
        attn_scores.mul_(self.scale)

        output, attn = self.attention.forward_attention(attn_scores, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn


class TransformerDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx = getattr(config, 'pad_token_id', config.pad_id)
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # Token embedding: checkpoint uses tgt_word_emb
        self.tgt_word_emb = nn.Embedding(
            config.vocab_size, config.d_model, self.padding_idx
        )
        # Positional encoding: checkpoint uses positional_encoding with buffer 'pe'
        # Use pe_maxlen (5000) from config, not max_target_positions (448)
        pe_maxlen = getattr(config, 'pe_maxlen', 5000)
        self.positional_encoding = PositionalEncoding(
            config.d_model, pe_maxlen
        )
        # Decoder layers: checkpoint uses layer_stack
        self.start_layer, self.end_layer, self.layer_stack = make_layers(
            config.decoder_layers,
            lambda prefix: DecoderLayer(
                vllm_config=vllm_config, prefix=f"{prefix}"
            ),
            prefix=f"{prefix}.layer_stack",
        )
        # Final layer norm: checkpoint uses layer_norm_out
        self.layer_norm_out = nn.LayerNorm(config.d_model)
        # Output projection: checkpoint uses tgt_word_prj
        self.tgt_word_prj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        inputs_embeds = self.embed_input_ids(input_ids)
        pos_emb = self.positional_encoding(positions)
        hidden_states = inputs_embeds * self.embed_scale + pos_emb
        for i, decoder_layer in enumerate(self.layer_stack):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        hidden_states = self.layer_norm_out(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tgt_word_emb(input_ids)


class DecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Self attention: checkpoint uses self_attn with w_qs/w_ks/w_vs/fc
        self.self_attn = DecoderSelfAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        # Checkpoint uses self_attn_norm (not self_attn_layer_norm)
        self.self_attn_norm = nn.LayerNorm(config.d_model)
        
        # Cross attention: checkpoint uses cross_attn (not encoder_attn)
        self.cross_attn = DecoderCrossAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
        )
        # Checkpoint uses cross_attn_norm (not encoder_attn_layer_norm)
        self.cross_attn_norm = nn.LayerNorm(config.d_model)
        
        # MLP: checkpoint uses mlp with w_1/w_2
        self.mlp = MLP(
            embed_dim=config.d_model,
            ffn_dim=config.decoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Checkpoint uses mlp_norm (not final_layer_norm)
        self.mlp_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



class DecoderSelfAttention(nn.Module):
    """Decoder self-attention with separate w_qs/w_ks/w_vs/fc to match checkpoint."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        per_layer_sliding_window: int | None = None,
        block_pool_size: int = 1,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            assert self.total_num_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        # Checkpoint uses separate w_qs, w_ks, w_vs (not qkv_proj)
        # Note: w_ks has no bias in checkpoint
        self.w_qs = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.w_qs",
        )
        self.w_ks = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=False,  # No bias for w_ks in checkpoint
            quant_config=quant_config,
            prefix=f"{prefix}.w_ks",
        )
        self.w_vs = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.w_vs",
        )
        # Checkpoint uses fc (not out_proj)
        self.fc = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        # if block_pool_size > 1:
        #     attn_cls = partial(
        #         WhisperAttentionWithBlockPooling, block_pool_size=block_pool_size
        #     )
        # else:
        #     attn_cls = Attention

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=self.attn_type,
            per_layer_sliding_window=per_layer_sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        q, _ = self.w_qs(hidden_states)
        k, _ = self.w_ks(hidden_states)
        v, _ = self.w_vs(hidden_states)

        attn_output = self.attn(q, k, v)

        output, _ = self.fc(attn_output)

        return output


class DecoderCrossAttention(nn.Module):
    """Decoder cross-attention with separate w_qs/w_ks/w_vs/fc to match checkpoint."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            assert self.total_num_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = AttentionType.ENCODER_DECODER

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        # Checkpoint uses w_qs for query projection
        self.w_qs = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.w_qs",
        )
        # Checkpoint uses w_ks (no bias) and w_vs for key/value
        self.w_ks = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=False,  # No bias for w_ks in checkpoint
            quant_config=quant_config,
            prefix=f"{prefix}.w_ks",
        )
        self.w_vs = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.w_vs",
        )
        # Checkpoint uses fc (not out_proj)
        self.fc = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        # Use vLLM's CrossAttention for KV cache support
        self.attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=self.attn_type,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        q, _ = self.w_qs(hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            k, _ = self.w_ks(encoder_hidden_states)
            v, _ = self.w_vs(encoder_hidden_states)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)

        output, _ = self.fc(attn_output)

        return output


class MLP(nn.Module):
    """MLP with w_1/w_2 naming to match checkpoint."""
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.activation_fn = get_act_fn(act_fn)
        # Checkpoint uses w_1 (not fc1)
        self.w_1 = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.w_1",
        )
        # Checkpoint uses w_2 (not fc2)
        self.w_2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.w_2",
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states, _ = self.w_1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.w_2(hidden_states)
        return hidden_states


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with 'pe' buffer to match checkpoint."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Get positional embeddings for given positions.
        
        Args:
            positions: Position indices [seq_len] or [batch, seq_len]
        Returns:
            Positional embeddings with same shape + d_model
        """
        # Handle both 1D and 2D position inputs
        if positions.dim() == 1:
            # [seq_len] -> [seq_len, d_model]
            return self.pe[0, positions]
        else:
            # [batch, seq_len] -> [batch, seq_len, d_model]
            return self.pe[0, positions]


class FireRedASRModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = ConformerEncoder(
            vllm_config=vllm_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = TransformerDecoder(
            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        enc_states = torch.cat(encoder_outputs, dim=0) if len(encoder_outputs) else None
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=enc_states,
        )
        return decoder_outputs


class FireRedASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config(PretrainedConfig)

    @property
    def skip_prompt_length_check(self) -> bool:
        return True  # Because the encoder prompt is padded

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_target_channels(self) -> int:
        return 1

    def get_data_parser(self) -> MultiModalDataParser:
        """Override to provide target_sr for audio resampling."""
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_num_audio_tokens(self) -> int:
        return self.get_hf_config().max_source_positions

    def get_audio_token_id(self) -> int:
        """获取 <|AUDIO|> 占位符的 token id"""
        hf_processor = self.get_hf_processor()
        return hf_processor.audio_token_id


class FireRedASRDummyInputsBuilder(BaseDummyInputsBuilder[FireRedASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|AUDIO|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class FireRedASRMultiModalProcessor(EncDecMultiModalProcessor[FireRedASRProcessingInfo]):
    """MultiModal processor for FireRedASR encoder-decoder model.
    
    Inherits from EncDecMultiModalProcessor to properly handle
    encoder-decoder architecture (like Whisper).
    """
    
    def build_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.info.get_target_channels(),
        )

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        # For encoder-decoder models, encoder only accepts audio features.
        # Return a dummy encoder prompt which will be padded to num_audio_tokens.
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            mm_data = dict(audio=mm_data.pop("audios"))
            mm_kwargs = dict(
                **mm_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if "labels" in processed_outputs:
            processed_outputs["input_ids"] = processed_outputs.pop("labels")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            speech_lengths=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_tokens = self.info.get_num_audio_tokens()
        # Use [0] as target to match the dummy encoder prompt created in create_encoder_prompt
        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    FireRedASRMultiModalProcessor,
    info=FireRedASRProcessingInfo,
    dummy_inputs=FireRedASRDummyInputsBuilder,
)
class FireRedAsrAedLForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    """
    FireRedASR model adapted for vLLM
    
    This follows the same pattern as WhisperForConditionalGeneration in vLLM.
    Key requirements for vLLM compatibility:
    1. __init__(vllm_config, prefix) signature
    2. forward(input_ids, positions, encoder_outputs) signature
    3. Implement: embed_multimodal, embed_input_ids, compute_logits
    4. Set supports_transcription_only = True
    """

    # Mark as transcription-only model (like Whisper)
    supports_transcription_only = True
    supports_segment_timestamp = True
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,  # not needed here
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        if language is None:
            raise ValueError(
                "Language must be specified when creating the Whisper prompt"
            )
        prompt = {
            "encoder_prompt": {
                # Whisper does not support encoder prompt.
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio, stt_config.sample_rate),
                },
            },
            "decoder_prompt": ""
        }
        return cast(PromptType, prompt)

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)

        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        
        # Create encoder and decoder
        self.model = FireRedASRModel(vllm_config=vllm_config, prefix=prefix)

        # Logits processor
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for vLLM.
        
        Args:
            input_ids: Decoder input token IDs [batch, seq_len]
            positions: Position IDs for decoder [batch, seq_len]
            encoder_outputs: Pre-computed encoder outputs (list of tensors)

        Returns:
            Decoder hidden states
        """
        if encoder_outputs is None:
            encoder_outputs = []

        decoder_outputs = self.model(input_ids, positions, encoder_outputs)
        return decoder_outputs

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        # Encoder returns (enc_output, input_lengths, src_mask)
        # We only need enc_output for multimodal embeddings

        enc_output, _, _ = self.model.encoder(audio_input["input_features"])
        return enc_output.unbind(dim=0)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> FireRedASRAudioInputs:
        input_features = kwargs.pop("input_features", None)

        if input_features is not None:
            input_features = json_map_leaves(lambda x: x.to(self.dtype), input_features)

        input_features = input_features.transpose(1, 2)
        return FireRedASRAudioInputs(input_features=input_features)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # inputs_embeds = self.model.decoder.embed_input_ids(input_ids)

        # return _merge_multimodal_embeddings(
        #     inputs_embeds=inputs_embeds,
        #     multimodal_embeddings=multimodal_embeddings,
        #     is_multimodal=_require_is_multimodal(is_multimodal),
        # )
        return multimodal_embeddings

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute output logits from hidden states.
        Required for VllmModelForTextGeneration interface.
        This method is critical - its presence is checked to determine
        if the model is a text generation model.
        """
        # tgt_word_prj is a standard nn.Linear, so we apply it directly
        # instead of using logits_processor which expects quant_method
        logits = self.model.decoder.tgt_word_prj(hidden_states)
        return logits
    
    def load_weights(self, weights):
        """Load model weights from checkpoint.
        
        The checkpoint uses FireRedASR naming convention which matches
        our model structure exactly.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded_params = set()

        for name, loaded_weight in weights:
            # Handle parameters
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)
                loaded_params.add(name)
            # Handle buffers (like positional_encoding.pe)
            elif name in buffers_dict:
                buffers_dict[name].data.copy_(loaded_weight)
                loaded_params.add(name)
        
        return loaded_params
    
    def get_language_model(self) -> nn.Module:
        """Return the decoder (language model) component."""
        return self.model.decoder
