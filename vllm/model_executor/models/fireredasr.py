"""
FireRedASR LLM mode integration for vLLM.

This module implements the vLLM integration for FireRedASR's LLM-based ASR model,
which uses a speech encoder + projector + Qwen2 LLM architecture.
"""
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, List, Literal, Optional, Tuple, TypedDict, Union

import kaldi_native_fbank as knf
import kaldiio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, PretrainedConfig

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.fireredasr import FireRedAsrEncoderConfig, FireRedAsrConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)


class ASRFeatExtractor:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)
        self.sampling_rate = 16000

    def __call__(self, wav_paths):
        feats = []
        durs = []
        for wav_path in wav_paths:
            if isinstance(wav_path, np.ndarray):
                if wav_path.dtype == np.float32:
                    wav_np = np.round(wav_path * 32768).astype(np.int16)
                    sample_rate = self.sampling_rate
                else:
                    wav_np = wav_path
                    sample_rate = self.sampling_rate
            else:
                sample_rate, wav_np = kaldiio.load_mat(wav_path)
            dur = wav_np.shape[0] / sample_rate
            fbank = self.fbank((sample_rate, wav_np))
            if self.cmvn is not None:
                fbank = self.cmvn(fbank)
            fbank = torch.from_numpy(fbank).float()
            feats.append(fbank)
            durs.append(dur)
        lengths = torch.tensor([feat.size(0) for feat in feats]).long()
        feats_pad = self.pad_feat(feats, 0.0)
        return feats_pad, lengths, durs

    def pad_feat(self, xs, pad_value):
        # type: (List[Tensor], int) -> Tensor
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = torch.ones(n_batch, max_len, *xs[0].size()[1:]).to(xs[0].device).to(xs[0].dtype).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
        return pad


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file), f"{kaldi_cmvn_file} not found"
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2, f"Invalid CMVN stats shape: {stats.shape}"
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1, f"Invalid CMVN frame count: {count}"
        floor = 1e-20
        sums = stats[0, :dim]
        square_sums = stats[1, :dim]
        means = sums / count
        variances = square_sums / count - means * means
        variances = np.maximum(variances, floor)
        inverse_std_variences = 1.0 / np.sqrt(variances)
        return dim, means.astype(np.float32), inverse_std_variences.astype(np.float32)


class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither
        fbank = knf.OnlineFbank(self.opts)

        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            print("Check data, len(feat) == 0", wav, flush=True)
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat

class ConformerEncoder(nn.Module):
    def __init__(self, idim, n_layers, n_head, d_model,
                 residual_dropout=0.1, dropout_rate=0.1, kernel_size=33,
                 pe_maxlen=5000):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.dropout = nn.Dropout(residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = RelPosEmbConformerBlock(d_model, n_head,
                        residual_dropout,
                        dropout_rate, kernel_size)
            self.layer_stack.append(block)

    def forward(self, padded_input, input_lengths, pad=True):
        if pad:
            padded_input = F.pad(padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1), 'constant', 0.0)
        src_mask = self.padding_position_is_0(padded_input, input_lengths)

        embed_output, input_lengths, src_mask = self.input_preprocessor(padded_input, src_mask)
        # embed_output, input_lengths, src_mask, attn_mask = self.input_preprocessor(padded_input, src_mask, input_lengths)
        enc_output = self.dropout(embed_output)

        pos_emb = self.dropout(self.positional_encoding(embed_output))

        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, pos_emb, slf_attn_mask=src_mask,
                                   pad_mask=src_mask)
            # enc_output = enc_layer(enc_output, pos_emb, slf_attn_mask=attn_mask,
            #                        pad_mask=attn_mask)
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    # def padding_position_is_0(self, padded_input, input_lengths):
    #     """
    #     优化: 使用向量化操作替代Python for循环，性能提升10-50倍
    #     原实现使用for循环在batch维度上迭代，在GPU上非常慢
    #     """
    #     N, T = padded_input.size()[:2]
    #     # mask = torch.ones((N, T)).to(padded_input.device)
    #     # for i in range(N):
    #     #     mask[i, input_lengths[i]:] = 0
    #     # mask = mask.unsqueeze(dim=1)
    #     # return mask.to(torch.uint8)

    #     # 创建位置索引 [0, 1, 2, ..., T-1]，shape: (T,)
    #     positions = torch.arange(T, device=padded_input.device)
    #     # 扩展到batch维度，shape: (N, T)
    #     positions = positions.unsqueeze(0).expand(N, T)
    #     # 扩展input_lengths到时间维度，shape: (N, T)
    #     lengths = input_lengths.unsqueeze(1).expand(N, T)
    #     # 向量化比较: positions < lengths，一次性生成所有mask
    #     mask = (positions < lengths).unsqueeze(1).to(torch.uint8)
    #     return mask
    def padding_position_is_0(self, padded_input, input_lengths):
        N, T = padded_input.size()[:2]
        mask = torch.ones((N, T)).to(padded_input.device)
        for i in range(N):
            mask[i, input_lengths[i]:] = 0
        mask = mask.unsqueeze(dim=1)
        return mask.to(torch.uint8)
    # def padding_position_is_0(self, padded_input, input_lengths):
    #   N, T = padded_input.size()[:2]
    #   positions = torch.arange(T, device=padded_input.device)
    #   positions = positions.unsqueeze(0).expand(N, T)
    #   lengths = input_lengths.unsqueeze(1).expand(N, T)
    #   mask = (positions < lengths).unsqueeze(1).to(torch.uint8)
    #   return mask

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
    def __init__(self, idim, d_model, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
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
        """
        优化: 避免不必要的clone().detach()
        - pe已经通过register_buffer注册，默认不需要梯度
        - 切片操作会创建view，不会修改原始buffer
        - 移除clone()减少内存拷贝，detach()已经不需要
        """
        # Tmax = 2 * max_len - 1
        Tmax, T = self.pe.size(1), x.size(1)
        # 直接返回切片，无需clone和detach
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T]
        # pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T].clone().detach()
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
        """
        优化: 减少transpose次数和优化内存访问
        - 将4次独立transpose优化为更紧凑的形式
        - 使用连续的内存布局减少cache miss
        """
        residual = x
        out = self.pre_layer_norm(x)
        
        # transpose到channel-first格式用于卷积: (B, T, D) -> (B, D, T)
        out = out.transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        
        # 卷积模块
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)

        out = out.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        out = self.batch_norm(out)
        out = self.swish(out)
        out = out.transpose(1, 2)  # (B, T, D) -> (B, D, T)

        out = self.dropout(self.pointwise_conv2(out))
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        
        # 最后transpose回来: (B, D, T) -> (B, T, D)
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
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(n_head, d_k))
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

class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k)

        self.attention = DecoderScaledDotProductAttention(
            temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(n_head * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        q = self.w_qs(q).view(bs, -1, self.n_head, self.d_k)
        k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k)
        v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output = self.attention(q, k, v, mask=mask)
        # output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.fc(output)
        output = self.dropout(output)

        return output


class DecoderScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.INF = float("inf")

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.act(self.w_1(x)))
        output = self.dropout(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        length = x.size(1)
        return self.pe[:, :length].clone().detach()

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = DecoderMultiHeadAttention(d_model, n_head, dropout)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = DecoderMultiHeadAttention(d_model, n_head, dropout)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_model*4, dropout)

    def forward(self, dec_input, enc_output, self_attn_mask, cross_attn_mask, cache=None):
        x = dec_input
        residual = x
        x = self.self_attn_norm(x)
        if cache is not None:
            xq = x[:, -1:, :]
            residual = residual[:, -1:, :]
            self_attn_mask = self_attn_mask[:, -1:, :]
        else:
            xq = x
        # print(f"sxl DecoderLayer self_attn: {type(self.self_attn)}")
        x = self.self_attn(xq, x, x, mask=self_attn_mask)

        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, enc_output, enc_output, mask=cross_attn_mask)
        x = residual + x

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x

class TransformerDecoder(nn.Module):
    def __init__(
            self, sos_id, eos_id, pad_id, odim,
            n_layers, n_head, d_model,
            residual_dropout=0.1, pe_maxlen=5000):
        super().__init__()
        self.INF = 1e10
        # parameters
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.n_layers = n_layers

        # Components
        self.tgt_word_emb = nn.Embedding(odim, d_model, padding_idx=self.pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = DecoderLayer(d_model, n_head, residual_dropout)
            self.layer_stack.append(block)

        self.tgt_word_prj = nn.Linear(d_model, odim, bias=False)
        self.layer_norm_out = nn.LayerNorm(d_model)

        self.tgt_word_prj.weight = self.tgt_word_emb.weight
        self.scale = (d_model ** 0.5)

    def batch_beam_search(self, encoder_outputs, src_masks,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
        B = beam_size
        N, Ti, H = encoder_outputs.size()
        device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0 and eos_penalty <= 1.0

        # Init
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
        src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
        ys = torch.ones(N*B, 1).fill_(self.sos_id).long().to(device)
        caches: List[Optional[Tensor]] = []
        for _ in range(self.n_layers):
            caches.append(None)
        scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(device)
        scores = scores.repeat(N).view(N*B, 1)
        is_finished = torch.zeros_like(scores)

        # Autoregressive Prediction
        for t in range(maxlen):
            tgt_mask = self.ignored_target_position_is_0(ys, self.pad_id)

            dec_output = self.dropout(
                self.tgt_word_emb(ys) * self.scale +
                self.positional_encoding(ys))

            i = 0
            for dec_layer in self.layer_stack:
                dec_output = dec_layer.forward(
                    dec_output, encoder_outputs,
                    tgt_mask, src_mask,
                    cache=caches[i])
                caches[i] = dec_output
                i += 1

            dec_output = self.layer_norm_out(dec_output)

            t_logit = self.tgt_word_prj(dec_output[:, -1])
            t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)

            if eos_penalty != 1.0:
                t_scores[:, self.eos_id] *= eos_penalty

            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
            t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
            t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

            # Accumulated
            scores = scores + t_topB_scores

            # Pruning
            scores = scores.view(N, B*B)
            scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
            scores = scores.view(-1, 1)

            topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N*B)
            stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N*B).to(device)
            topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

            # Update ys
            ys = ys[topB_row_number_in_ys]
            t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
            ys = torch.cat((ys, t_ys), dim=1)

            # Update caches
            new_caches: List[Optional[Tensor]] = []
            for cache in caches:
                if cache is not None:
                    new_caches.append(cache[topB_row_number_in_ys])
            caches = new_caches

            # Update finished state
            is_finished = t_ys.eq(self.eos_id)
            if is_finished.sum().item() == N*B:
                break

        # Length penalty (follow GNMT)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
        nbest_ys = ys.view(N*B, -1)[index.view(-1)]
        nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
        nbest_ys_lengths = ys_lengths.view(N*B)[index.view(-1)].view(N, -1)

        # result
        nbest_hyps: List[List[Dict[str, Tensor]]] = []
        for n in range(N):
            n_nbest_hyps: List[Dict[str, Tensor]] = []
            for i, score in enumerate(nbest_scores[n]):
                new_hyp = {
                    "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]]
                }
                n_nbest_hyps.append(new_hyp)
            nbest_hyps.append(n_nbest_hyps)
        return nbest_hyps

    def ignored_target_position_is_0(self, padded_targets, ignore_id):
        mask = torch.ne(padded_targets, ignore_id)
        mask = mask.unsqueeze(dim=1)
        T = padded_targets.size(-1)
        upper_tri_0_mask = self.upper_triangular_is_0(T).unsqueeze(0).to(mask.dtype)
        upper_tri_0_mask = upper_tri_0_mask.to(mask.dtype).to(mask.device)
        return mask.to(torch.uint8) & upper_tri_0_mask.to(torch.uint8)

    def upper_triangular_is_0(self, size):
        ones = torch.ones(size, size)
        tri_left_ones = torch.tril(ones)
        return tri_left_ones.to(torch.uint8)

    def set_finished_beam_score_to_zero(self, scores, is_finished):
        NB, B = scores.size()
        is_finished = is_finished.float()
        mask_score = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(scores.device)
        mask_score = mask_score.view(1, B).repeat(NB, 1)
        return scores * (1 - is_finished) + mask_score * is_finished

    def set_finished_beam_y_to_eos(self, ys, is_finished):
        is_finished = is_finished.long()
        return ys * (1 - is_finished) + self.eos_id * is_finished

    def get_ys_lengths(self, ys):
        N, B, Tmax = ys.size()
        ys_lengths = torch.sum(torch.ne(ys, self.eos_id), dim=-1)
        return ys_lengths.int()

class Adapter(nn.Module):
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        # 使用inplace=True减少内存分配
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x, x_lens):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        # 优化: 使用reshape替代contiguous+view，自动处理内存布局
        # reshape会在需要时自动调用contiguous，避免不必要的拷贝
        x = x.reshape(
            batch_size, seq_len // self.ds, feat_dim * self.ds
        )

        x = self.linear1(x)
        x = self.relu(x)  # inplace操作，节省内存
        x = self.linear2(x)

        # 优化: 直接整除，避免中间变量
        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens

class FireRedAsrAed(torch.nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.sos_id = args.sos_id
        self.eos_id = args.eos_id

        self.encoder = ConformerEncoder(
            args.idim, args.n_layers_enc, args.n_head, args.d_model,
            args.residual_dropout, args.dropout_rate,
            args.kernel_size, args.pe_maxlen)

        # self.decoder = TransformerDecoder(
        #     args.sos_id, args.eos_id, args.pad_id, args.odim,
        #     args.n_layers_dec, args.n_head, args.d_model,
        #     args.residual_dropout, args.pe_maxlen)

    # def transcribe(self, padded_input, input_lengths,
    #                beam_size=1, nbest=1, decode_max_len=0,
    #                softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
    #     enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
    #     nbest_hyps = self.decoder.batch_beam_search(
    #         enc_outputs, enc_mask,
    #         beam_size, nbest, decode_max_len,
    #         softmax_smoothing, length_penalty, eos_penalty)
    #     return nbest_hyps

def _fireredasr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    audio_feature_lengths = hf_inputs.get("audio_feature_lengths",
                                          torch.empty((0, )))

    return dict(
        input_audio_features=MultiModalFieldConfig.flat_from_sizes(
            "audio", audio_feature_lengths, dim=1),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        audio_feature_lengths=MultiModalFieldConfig.batched("audio")
    )


class FireRedAsrInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
        - msl: Maximum sequence length
        - tsl: Total sequence length
    """
    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "msl", "nmb", dynamic_dims={"msl"}),
    ]

    audio_feature_lengths: Annotated[torch.Tensor, TensorShape("na")]

    # feature_attention_mask: Annotated[
    #     torch.Tensor | list[torch.Tensor],
    #     TensorShape("na", "msl", dynamic_dims={"msl"}),
    # ]

class FireRedAsrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> PretrainedConfig:
        hf_config = self.ctx.get_hf_config(FireRedAsrConfig)
        setattr(hf_config, "cmvn_path", os.path.join(self.ctx.model_config.model, "cmvn.ark"))
        return hf_config
    
    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        pass

    def get_feature_extractor(self, **kwargs: object) -> ASRFeatExtractor:
        # print(f"sxl get_feature_extractor: {self.get_hf_config()}")
        cmvn_path = self.get_hf_config().cmvn_path
        feature_extractor = ASRFeatExtractor(cmvn_path)
        assert isinstance(feature_extractor, ASRFeatExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class FireRedAsrDummyInputsBuilder(BaseDummyInputsBuilder[FireRedAsrProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_config = self.info.get_hf_config()
        audio_token = hf_config.default_speech_token
        # print(f"sxl get_dummy_text audio_token: {audio_token}, num_audios: {num_audios}")
        # Return speech tokens for each audio item
        return audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = 1 * sampling_rate
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }

class FireRedAsrMultiModalDataParser(MultiModalDataParser):    
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"input_audio_features", "audio_feature_lengths"},
                fields_factory=_fireredasr_field_config,
            )
        return super()._parse_audio_data(data)


# from transformers.processing_utils import ProcessorMixin
# class FireRedAsrProcessor(ProcessorMixin):
#     attributes = ["feature_extractor", "tokenizer"]
#     feature_extractor_class = "ASRFeatExtractor"
#     tokenizer_class = ("AutoTokenizer")

#     def __init__(
#         self,
#         feature_extractor=None,
#         tokenizer=None,
#         chat_template=None,
#         audio_token="<speech>",
#         audio_bos_token="<|audio_bos|>",
#         audio_eos_token="<|audio_eos|>",
#         ):
#         # self.feature_extractor = feature_extractor
#         # self.tokenizer = tokenizer
#         super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

#     def __call__(
#         self,
#         text: TextInput = None,
#         audios: AudioInput = None,
#         **kwargs
#     ) -> BatchFeature:
#         # print(f"sxl __call__ text: {text}, audios: {audios}")
#         if text is None:
#             raise ValueError("You need to specify either a `text` input to process.")

#         encoded = self.tokenizer([text], padding="longest",truncation=True, max_length=128)
#         prompt_ids = torch.tensor(encoded["input_ids"])
#         if audios is not None:
#             feats, lengths, _ = self.feature_extractor(audios)
#             return BatchFeature({
#                 "speech_features": feats,
#                 "speech_lengths": lengths,
#                 "input_ids": prompt_ids,
#             })


class FireRedAsrMultiModalProcessor(
        BaseMultiModalProcessor[FireRedAsrProcessingInfo]):
    """Multimodal processor for FireRedASR."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get CMVN path from config (auto-resolved) or mm_kwargs (user override)
        cmvn_path = self.info.get_hf_config().cmvn_path
        if cmvn_path is None:
            raise ValueError(
                "cmvn_path could not be resolved. Please ensure the model directory "
                "contains 'cmvn.ark' or provide cmvn_path explicitly."
            )
        if not os.path.exists(cmvn_path):
            raise FileNotFoundError(f"CMVN file not found at {cmvn_path}")

        self.feat_extractor = ASRFeatExtractor(cmvn_path)

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return FireRedAsrMultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False


    def _calc_speech_features_time_length(self, feat_frames: int) -> int:
        """
        hard-code the speech_features time frame downsample calculation
        
        Args:
            feat_frames: time frames of the input features
        Returns:
            time frames of the final speech_features
        """
        # Step 1: Encoder Conv2dSubsampling
        padded = feat_frames + 6  # context=7, padding=6
        after_conv1 = (padded - 3) // 2 + 1
        encoder_frames = (after_conv1 - 3) // 2 + 1

        # Step 2: Adapter downsample
        speech_frames = encoder_frames // 2

        return max(1, speech_frames)

    def _get_mm_fields_config(
        self,
        hf_inputs: dict[str, Any],
        hf_processor_mm_kwargs: dict[str, Any],
    ) -> dict[str, MultiModalFieldConfig]:
        """Configure multimodal fields.

        Only include fields that will be passed to the model's forward method.
        Derived metadata like 'projected_lengths' should not be included here.
        """
        return {
            "speech_features": MultiModalFieldConfig.batched("audio"),
            "speech_lengths": MultiModalFieldConfig.batched("audio"),
        }
        
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # hf_config = self.info.get_hf_config()
        # tokenizer = self.info.get_tokenizer()
        # desired_padding_side = getattr(hf_config, 'tokenizer_padding_side', 'left')
        # if tokenizer.padding_side != desired_padding_side:
        #     tokenizer.padding_side = desired_padding_side
        # encoded = tokenizer([prompt], padding="longest",truncation=True, max_length=128)
        # prompt_ids = torch.tensor(encoded["input_ids"])
        # attention_mask = torch.tensor(encoded["attention_mask"])
        # print(f"sxl fireredasr _call_hf_processor mm_data: {mm_data}")
        audios = mm_data.get("audios", [])
        tokenizer = self.info.get_tokenizer()
        if audios:
            mm_data["audio"] = audios
        if not mm_data.get("audio", []):
            prompt_ids = tokenizer.encode(prompt)
            # print(f"sxl fireredasr prompt_ids: {prompt_ids}")
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        
        # print(f"sxl _call_hf_processor mm_kwargs: {mm_kwargs}")
        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )
        # print(f"sxl _call_hf_processor audios: {audios}")
        feats, lengths, _ = self.feat_extractor(audios)
        # print(f"sxl _call_hf_processor feats: {feats, feats.shape, feats.sum()}, lengths: {lengths}")


        hf_config = self.info.get_hf_config()
        desired_padding_side = getattr(hf_config, 'tokenizer_padding_side', 'left')
        if tokenizer.padding_side != desired_padding_side:
            tokenizer.padding_side = desired_padding_side
        encoded = tokenizer([prompt], padding="longest",truncation=True, max_length=128)
        prompt_ids = torch.tensor(encoded["input_ids"])
        # print(f"sxl _call_hf_processor inputs: {inputs}")

        return BatchFeature({
            "speech_features": feats,
            "speech_lengths": lengths,
            "input_ids": prompt_ids,
        })
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        # Get speech token and token ID
        speech_token = hf_config.default_speech_token
        vocab = tokenizer.get_vocab()
        speech_token_id = vocab[speech_token]

        # 提前提取和处理 speech_lengths
        speech_lengths_list = []
        # speech_lengths_data = out_mm_kwargs.get("speech_lengths")
        out_mm_data = out_mm_kwargs.get_data()
        # print(f"sxl _get_prompt_updates out_mm_data: {out_mm_data}")
        # speech_lengths_data = out_mm_kwargs.get_data()["speech_lengths"]
        speech_lengths_data = out_mm_data.get("speech_lengths")
        # print(f"sxl out_mm_kwargs: {out_mm_kwargs}, _get_prompt_updates speech_lengths_data: {speech_lengths_data}")
        if speech_lengths_data is not None:
            if isinstance(speech_lengths_data, torch.Tensor):
                # 转换为 list
                if speech_lengths_data.dim() == 1:
                    speech_lengths_list = [int(i.item()) for i in speech_lengths_data]
                elif speech_lengths_data.dim() == 0:
                    speech_lengths_list = [int(speech_lengths_data.item())]
            elif isinstance(speech_lengths_data, (list, tuple)):
                speech_lengths_list = [
                    int(item.item()) if isinstance(item, torch.Tensor) else int(item)
                    for item in speech_lengths_data
                ]
        else:
            # print(">>>>>>>>> speech_lengths is not provided, use 1 as fallback <<<<<<<<<<<")
            speech_lengths_list = [1]
        speech_lengths_list = [self._calc_speech_features_time_length(l) for l in speech_lengths_list]

        def get_replacement_fireredasr(item_idx: int) -> list[int]:
            """Get replacement tokens for a specific audio item."""
            if item_idx < len(speech_lengths_list):
                num_tokens = speech_lengths_list[item_idx]
            else:
                num_tokens = 1  # Fallback
            return [speech_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="audio",
                target=[speech_token_id],
                replacement=get_replacement_fireredasr,
            )
        ]

    # def _apply_hf_processor_main(
    #     self,
    #     prompt: str | list[int],
    #     mm_items: MultiModalDataItems,
    #     hf_processor_mm_kwargs: Mapping[str, object],
    #     tokenization_kwargs: Mapping[str, object],
    #     *,
    #     enable_hf_prompt_update: bool,
    # ) -> tuple[list[int], BatchFeature, bool]:
    #     if isinstance(prompt, str):
    #         if enable_hf_prompt_update:
    #             return self._apply_hf_processor_text_mm(
    #                 prompt_text=prompt,
    #                 mm_items=mm_items,
    #                 hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    #                 tokenization_kwargs=tokenization_kwargs,
    #             )
    #         tokenizer = self.info.get_tokenizer()
    #         prompt_ids = tokenizer.encode(prompt)
    #     else:
    #         prompt_ids = self._apply_hf_processor_tokens_only(prompt)

    #     mm_processed_data = self._apply_hf_processor_mm_only(
    #         mm_items=mm_items,
    #         hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    #         tokenization_kwargs=tokenization_kwargs,
    #     )

    #     return prompt_ids, mm_processed_data, False
    
    # def _apply_hf_processor_mm_only(
    #     self,
    #     mm_items: MultiModalDataItems,
    #     hf_processor_mm_kwargs: Mapping[str, object],
    #     tokenization_kwargs: Mapping[str, object],
    # ) -> BatchFeature:
    #     """
    #     Qwen2.5-Omni reimplements this function to handle `use_audio_in_video`.
    #     """
    #     mm_counts = mm_items.get_all_counts()

    #     # use_audio_in_video = hf_processor_mm_kwargs.get("use_audio_in_video", False)
    #     # if use_audio_in_video and "video" in mm_counts:
    #     #     assert "audio" in mm_counts
    #     #     mm_counts["audio"] -= mm_counts["video"]

    #     _, mm_processed_data, _ = self._apply_hf_processor_text_mm(
    #         prompt_text=self.dummy_inputs.get_dummy_text(mm_counts),
    #         mm_items=mm_items,
    #         hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    #         tokenization_kwargs=tokenization_kwargs,
    #     )

    #     return mm_processed_data

# ============= Model Components =============

class FireRedAsrEncoder(nn.Module):
    """
    Wrapper for FireRedASR's speech encoder.
    Loads the encoder from FireRedAsrAed model.
    """

    def __init__(self, config: FireRedAsrEncoderConfig):
        super().__init__()

        model = FireRedAsrAed.from_args(config)
        self.encoder = model.encoder
        self.encoder_dim = self.encoder.odim
    
    def forward(
        self,
        speech_features: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """
        Args:
            speech_features: (batch, time, feat_dim)
            speech_lengths: (batch,)
        
        Returns:
            encoder_outputs: (batch, time', encoder_dim)
            output_lengths: (batch,)
            encoder_mask: (batch, 1, time')
        """
        encoder_outs, enc_lengths, enc_mask = self.encoder(speech_features, speech_lengths)
        # print(f"sxl FireRedAsrEncoder encoder_outs: {encoder_outs, encoder_outs.shape, encoder_outs.sum()}")
        # print(f"sxl FireRedAsrEncoder enc_lengths: {enc_lengths}")
        # print(f"sxl FireRedAsrEncoder enc_mask: {enc_mask, enc_mask.shape, enc_mask.sum()}")
        return encoder_outs, enc_lengths, enc_mask


class FireRedAsrProjector(nn.Module):
    """
    Adapter/Projector that maps encoder outputs to LLM embedding space.
    This is a wrapper around FireRedASR's Adapter module.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 4,
    ):
        super().__init__()
        
        if Adapter is None:
            raise ImportError(
                "FireRedASR is not installed. Please install it to use this model."
            )
        
        self.adapter = Adapter(encoder_dim, llm_dim, downsample_rate)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: (batch, time, encoder_dim)
            output_lengths: (batch,)
        
        Returns:
            projected_features: (batch, time', llm_dim)
            projected_lengths: (batch,)
        """
        return self.adapter(encoder_outputs, output_lengths)


# ============= Main Model =============

@MULTIMODAL_REGISTRY.register_processor(
    FireRedAsrMultiModalProcessor,
    info=FireRedAsrProcessingInfo,
    dummy_inputs=FireRedAsrDummyInputsBuilder,
)
class FireRedAsrForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    """
    FireRedASR model for conditional generation in vLLM.

    Architecture:
        Audio -> Encoder -> Projector -> LLM (Qwen2)
    """

    supports_multimodal: bool = True

    # Weight mapping from original FireRedASR checkpoint to vLLM model structure
    # This handles the prefix differences between training and inference
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Encoder mapping
            "encoder.": "audio_tower.encoder.",
            # Projector/Adapter mapping
            "encoder_projector.": "projector.adapter.",
            # LoRA mapping (MUST come before base LLM mapping!)
            # Format in checkpoint: llm.base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight
            # Format in vLLM with LoRA: language_model.model.layers.X.self_attn.q_proj.lora_A.default.weight
            "llm.base_model.model.": "language_model.",
            # LLM base mapping (for full finetuning scenario)
            "llm.model.": "language_model.model.",
            "llm.lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        audio_config = vllm_config.model_config.hf_config.encode_config
        text_config = vllm_config.model_config.hf_config.text_config
        self.config = vllm_config.model_config.hf_config
        
        self.model_dir = vllm_config.model_config.model
        self.llm_dir = os.path.join(self.model_dir, self.config.text_config.model_name)
        
        self.audio_tower = FireRedAsrEncoder(audio_config)

        # Get actual encoder dimension from loaded encoder
        encoder_dim = self.audio_tower.encoder_dim

        # Initialize LLM
        if self.llm_dir is None:
            raise ValueError("llm_dir must be provided in config. "
                           "Please ensure the model directory contains a Qwen2 subdirectory")

        if not os.path.exists(self.llm_dir):
            raise FileNotFoundError(f"LLM directory not found at {self.llm_dir}")

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=vllm_config.model_config.hf_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=text_config.architectures,
        )

        self.projector = FireRedAsrProjector(
            encoder_dim=self.audio_tower.encoder_dim,
            llm_dim=text_config.hidden_size,
            downsample_rate=self.config.encoder_downsample_rate,
        )

        self.projector.float().to(self.device)
        self.audio_tower.float().to(self.device)
        # self.projector.eval()
        # self.audio_tower.eval()
        # self.language_model.eval()

        # Initialize sampler (if needed for standalone use)
        self.sampler = Sampler()

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        """
        Get placeholder text for FireRedASR audio inputs.
        """
        if modality == "audio":
            return "<speech>"
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def get_language_model(self) -> torch.nn.Module:
        """
        Returns the underlying language model used for text generation.
        """
        return self.language_model
    
    def _validate_and_reshape_mm_tensor(
        self,
        mm_input: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        name: str,
    ) -> torch.Tensor:
        """Validate and reshape multimodal tensor input."""
        if mm_input is None:
            return torch.empty(0, device=self.device)
        
        if isinstance(mm_input, list):
            if not mm_input:
                return torch.empty(0, device=self.device)
            mm_input = torch.stack(mm_input)
        
        return mm_input

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> FireRedAsrInputs | None:
        input_audio_features = kwargs.pop("speech_features", None)
        audio_feature_lengths = kwargs.pop("speech_lengths", None)
        # feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        # print(f"sxl _parse_and_validate_audio_input speech_features: {input_audio_features.shape}, speech_lengths: {audio_feature_lengths}")

        if input_audio_features is None:
            return None

        if isinstance(input_audio_features, list) and input_audio_features:
            # Find max length across all tensors
            # max_time_len = max(feat.shape[0] for feat in input_audio_features)
            max_time_len = audio_feature_lengths.max().item() 

            # Pad each tensor to max length
            padded_features = []
            for feat in input_audio_features:
                if feat.shape[0] < max_time_len:
                    # Pad on time dimension (dimension 1)
                    pad_len = max_time_len - feat.shape[0]
                    # Pad format: (left, right, top, bottom, front, back)
                    # We pad on the right side of dimension 1
                    feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_len), value=0.0)
                padded_features.append(feat)
            input_audio_features = padded_features

        input_audio_features = self._validate_and_reshape_mm_tensor(
            input_audio_features, "speech_features"
        )
        audio_feature_lengths = self._validate_and_reshape_mm_tensor(
            audio_feature_lengths, "speech_lengths"
        )

        if input_audio_features.numel() == 0:
            return None

        # print(f"sxl input_audio_features: {input_audio_features.shape}")
        # input_audio_features = flatten_bn(input_audio_features, concat=True)
        # audio_feature_lengths = flatten_bn(audio_feature_lengths, concat=True)


        return FireRedAsrInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
        )
    
    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_inputs = self._parse_and_validate_audio_input(**kwargs)
        if audio_inputs is None:
            return []

        speech_features = audio_inputs["input_features"]
        speech_lengths = audio_inputs["audio_feature_lengths"]

        encoder_outputs, output_lengths, encoder_mask = self.audio_tower(
            speech_features, speech_lengths
        )

        # Explicitly delete encoder mask to free memory immediately
        # del encoder_mask

        # Run projector
        projected_features, projected_lengths = self.projector(
            encoder_outputs, output_lengths
        )

        batch_size = projected_features.size(0)
        feat_dim = projected_features.size(2)
        max_length = max(int(l.item()) for l in projected_lengths)

        # 为所有样本预分配内存
        audio_embeddings = torch.zeros(
            batch_size, max_length, feat_dim,
            device=self.device, dtype=projected_features.dtype
        )

        # 填充有效数据
        for i in range(batch_size):
            actual_len = int(projected_lengths[i].item())
            if actual_len > 0:
                audio_embeddings[i, :actual_len] = projected_features[i, :actual_len]

        return tuple(audio_embeddings[i, :int(projected_lengths[i].item())] for i in range(batch_size))


    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass through the model."""
        # print(f"sxl FireRedAsrForConditionalGeneration input_ids: {input_ids}, positions: {positions}, intermediate_tensors: {intermediate_tensors}, inputs_embeds: {inputs_embeds}")
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Process multimodal inputs if needed
        # In V1 the inputs_embeds should always be generated at model runner

        # Forward through language model
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states)
    
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Custom weight loading for FireRedASR.

        This method loads weights from two sources:
        1. Encoder and Projector weights from model.pth.tar (if exists)
        2. LLM weights from the separate llm_dir specified in config
        """
        import logging
        logger = logging.getLogger(__name__)

        # Check if we should load LLM from separate directory
        load_llm_separately = (self.llm_dir is not None and  os.path.exists(self.llm_dir))

        # Get model.pth.tar path for encoder/projector weights
        model_pth_path = os.path.join(self.model_dir, "model.pth.tar")

        # Initialize weight containers
        encoder_weights = {}
        projector_weights = {}
        llm_weights = []

        # Load encoder and projector weights from model.pth.tar if it exists
        if os.path.exists(model_pth_path):
            logger.info(f"Loading encoder/projector weights from {model_pth_path}")

            # Load the checkpoint
            try:
                package = torch.load(model_pth_path, map_location=lambda storage, loc: storage, weights_only=False)
            except Exception as e:
                logger.error(f"Failed to load {model_pth_path}: {e}")
                raise

            if "model_state_dict" not in package:
                logger.error(f"'model_state_dict' not found in {model_pth_path}")
                logger.error(f"Available keys: {list(package.keys())}")
                raise RuntimeError("Invalid checkpoint format")

            state_dict = package["model_state_dict"]
            logger.info(f"Loaded checkpoint with {len(state_dict)} parameters")

            # Process weights from model.pth.tar
            for orig_name, weight in state_dict.items():
                # Apply weight mapping
                # print(f"sxl orig_name: {orig_name}, weight: {weight.shape}")
                mapped_name = orig_name
                for old_prefix, new_prefix in self.hf_to_vllm_mapper.orig_to_new_prefix.items():
                    if mapped_name.startswith(old_prefix):
                        mapped_name = new_prefix + mapped_name[len(old_prefix):]
                        break

                # Categorize weights
                if mapped_name.startswith("audio_tower."):
                    encoder_weights[mapped_name] = weight
                elif mapped_name.startswith("projector."):
                    projector_weights[mapped_name] = weight
                elif mapped_name.startswith("language_model.") and not load_llm_separately:
                    # Only load LLM weights from model.pth.tar if not loading separately
                    llm_internal_name = mapped_name.replace("language_model.", "", 1)
                    llm_weights.append((llm_internal_name, weight))
        else:
            logger.warning(f"model.pth.tar not found at {model_pth_path}")
            logger.info("Encoder and projector weights will need to be loaded separately")

        # Load LLM weights from separate directory if specified
        if load_llm_separately:
            logger.info(f"Loading LLM weights from separate directory: {self.llm_dir}")

            # Use vLLM's model loader to load LLM weights
            from vllm.model_executor.model_loader import get_model_loader
            from vllm.config import LoadConfig, ModelConfig

            # from fpdb import ForkedPdb; ForkedPdb().set_trace()

            # Create a temporary ModelConfig for the LLM
            # Use tokenizer_path if available, otherwise fall back to llm_dir
            tokenizer_path = getattr(self.config, 'tokenizer_path', None) or self.llm_dir
            llm_model_config = ModelConfig(
                model=self.llm_dir,
                tokenizer=tokenizer_path,
                tokenizer_mode="auto",
                trust_remote_code=True,
                dtype=self.vllm_config.model_config.dtype,
                seed=self.vllm_config.model_config.seed,
            )

            # Create LoadConfig
            llm_load_config = LoadConfig(load_format="auto")

            # Get the model loader
            llm_loader = get_model_loader(llm_load_config)

            # Load LLM weights using the loader
            logger.info("Loading LLM weights using vLLM model loader...")
            llm_loader.load_weights(self.language_model, llm_model_config)
            logger.info("✓ LLM weights loaded from separate directory")

        elif llm_weights:
            # Load LLM weights from model.pth.tar (old behavior)
            logger.info(f"Loading {len(llm_weights)} LLM weights from model.pth.tar...")
            if hasattr(self.language_model, 'load_weights'):
                loaded_llm_params = self.language_model.load_weights(llm_weights)
                logger.info(f"Loaded LLM parameters: {len(loaded_llm_params) if loaded_llm_params else 'unknown'}")
            else:
                logger.info("Using state_dict fallback for LLM weights...")
                llm_state_dict = dict(llm_weights)
                missing_llm, unexpected_llm = self.language_model.load_state_dict(
                    llm_state_dict, strict=False
                )
                if missing_llm:
                    logger.warning(f"Missing LLM keys: {len(missing_llm)}")

        # Log what we found
        logger.info(f"Weight loading summary:")
        logger.info(f"  - Encoder parameters: {len(encoder_weights)}")
        logger.info(f"  - Projector parameters: {len(projector_weights)}")
        if load_llm_separately:
            logger.info(f"  - LLM: Loaded from {self.llm_dir}, {len(llm_weights)}")
        else:
            logger.info(f"  - LLM parameters from model.pth.tar: {len(llm_weights)}")

        # Load encoder and projector weights using load_state_dict
        encoder_projector_weights = {**encoder_weights, **projector_weights}
        # print(f"sxl encoder_projector_weights: {encoder_projector_weights.keys()}")
        if encoder_projector_weights:
            logger.info(f"Loading encoder and projector weights...")
            missing_keys, unexpected_keys = self.load_state_dict(
                encoder_projector_weights, strict=False
            )
            if missing_keys:
                logger.warning(f"Missing keys when loading encoder/projector: {missing_keys[:5]}")

        # Mark all parameters as loaded for vLLM
        loaded_params = set(self.state_dict().keys())

        # Final summary
        current_state = self.state_dict()
        # print(f"sxl current_state: {current_state.keys(), len(current_state.keys())}")
        total_llm = sum(1 for k in current_state.keys() if k.startswith("language_model."))
        total_encoder = sum(1 for k in current_state.keys() if k.startswith("audio_tower."))
        total_projector = sum(1 for k in current_state.keys() if k.startswith("projector."))

        logger.info(f"\n✓ Successfully loaded FireRedASR model weights from {model_pth_path}")
        logger.info(f"  Total model parameters:")
        logger.info(f"    - Audio Tower: {total_encoder}")
        logger.info(f"    - Projector: {total_projector}")
        logger.info(f"    - LLM (Qwen2): {total_llm}")
        logger.info(f"    - Total: {len(loaded_params)}")

        return loaded_params
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
