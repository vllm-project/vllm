# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FireRedASR LLM mode integration for vLLM.

This module implements the vLLM integration for FireRedASR's LLM-based ASR model,
which uses a speech encoder + projector + Qwen2 LLM architecture.
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal

import kaldi_native_fbank as knf
import kaldiio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, PretrainedConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
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
    AudioProcessorItems,
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
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.firered import (
    FireRedAudioConfig,
    FireRedAudioEncoderConfig,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = self.read_kaldi_cmvn(
            kaldi_cmvn_file
        )

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        if not os.path.exists(kaldi_cmvn_file):
            raise FileNotFoundError(f"{kaldi_cmvn_file} not found")
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
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0):
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

        fbank.accept_waveform(sample_rate, wav_np)
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat


class ASRFeatExtractor:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(
            num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0
        )
        self.sampling_rate = 16000
        self.hop_length = 160  # 10ms frame shift at 16kHz

    def __call__(self, wav_paths, return_attention_mask=True):
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

        output = {
            "input_features": feats_pad,
            "speech_lengths": lengths,
            "durs": durs,
        }

        if return_attention_mask:
            attention_mask = self.generate_attention_mask(lengths, feats_pad.size(1))
            output["attention_mask"] = attention_mask

        return output

    def pad_feat(self, xs, pad_value):
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = (
            torch.ones(n_batch, max_len, *xs[0].size()[1:])
            .to(xs[0].device)
            .to(xs[0].dtype)
            .fill_(pad_value)
        )
        for i in range(n_batch):
            pad[i, : xs[i].size(0)] = xs[i]
        return pad

    def generate_attention_mask(self, lengths, max_length):
        batch_size = lengths.size(0)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1
        return attention_mask


class RelPosEmbConformerBlock(nn.Module):
    def __init__(
        self, d_model, n_head, residual_dropout=0.1, dropout_rate=0.1, kernel_size=33
    ):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model, residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size, dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, slf_attn_mask=None, pad_mask=None):
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


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
        left_context = right_context = 3
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
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros([1, max_len * 2 - 1, d_model])
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T]
        return pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        pre_layer_norm = nn.LayerNorm(d_model)
        linear_expand = nn.Linear(d_model, d_model * 4)
        nonlinear = nn.SiLU()
        dropout_pre = nn.Dropout(dropout_rate)
        linear_project = nn.Linear(d_model * 4, d_model)
        dropout_post = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(
            pre_layer_norm,
            linear_expand,
            nonlinear,
            dropout_pre,
            linear_project,
            dropout_post,
        )

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
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 4, kernel_size=1, bias=False
        )
        self.glu = F.glu
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size,
            stride=1,
            padding=self.padding,
            groups=d_model * 2,
            bias=False,
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            d_model * 2, d_model, kernel_size=1, bias=False
        )
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
        out = self.batch_norm(out)
        out = self.swish(out)
        out = out.transpose(1, 2)

        out = self.dropout(self.pointwise_conv2(out))
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)

        out = out.transpose(1, 2)
        return out + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, residual_dropout=0.1):
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

        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)

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
        self.INF = float("inf")

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
    def __init__(self, n_head, d_model, residual_dropout=0.1):
        super().__init__(n_head, d_model, residual_dropout)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k**0.5)
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)).item() / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        length = x.size(1)
        return self.pe[:, :length].clone().detach()


def _fireredasr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    audio_feature_lengths = hf_inputs.get("audio_feature_lengths", torch.empty((0,)))

    return dict(
        input_audio_features=MultiModalFieldConfig.flat_from_sizes(
            "audio", audio_feature_lengths, dim=1
        ),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
    )


class FireRedAudioInputs(TensorSchema):
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

    feature_attention_mask: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "msl", dynamic_dims={"msl"}),
    ]


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
    padded = input_lengths + 6
    after_conv1 = (padded - 3) // 2 + 1
    encoder_frames = (after_conv1 - 3) // 2 + 1

    output_lengths = encoder_frames // 2

    return output_lengths


class FireRedAsrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> PretrainedConfig:
        hf_config = self.ctx.get_hf_config(FireRedAudioConfig)
        hf_config.cmvn_path = os.path.join(self.ctx.model_config.model, "cmvn.ark")
        return hf_config

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: int | None = None,
        **kwargs: object,
    ):
        pass

    def get_feature_extractor(self, **kwargs: object) -> ASRFeatExtractor:
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
        return num_audios * audio_token

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = 30 * sampling_rate
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


class FireRedAsrMultiModalProcessor(BaseMultiModalProcessor[FireRedAsrProcessingInfo]):
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

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "speech_lengths": MultiModalFieldConfig.batched("audio"),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
        }

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios

        tokenizer = self.info.get_tokenizer()
        if not mm_data.get("audio", []):
            prompt_ids = tokenizer.encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        audio_inputs = feature_extractor(audios)

        expanded_text = []
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
        audio_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()

        audio_token = "<speech>"
        text = [prompt]
        for sample in text:
            replace_str = []
            while audio_token in sample:
                audio_length = audio_lengths.pop(0)
                num_audio_tokens = _get_feat_extract_output_lengths(audio_length)

                expanded_audio_token = audio_token * num_audio_tokens
                replace_str.append(expanded_audio_token)
                sample = sample.replace(audio_token, "<placeholder>", 1)

            while "<placeholder>" in sample:
                sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
            expanded_text.append(sample)
        text = expanded_text
        inputs = tokenizer(text, padding=False, padding_side="left", truncation=False)
        if audios is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs}, tensor_type="pt")

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()

        audio_token = hf_config.default_speech_token
        vocab = tokenizer.get_vocab()
        audio_token_id = vocab[audio_token]

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")

        if feature_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(feature_attention_mask, torch.Tensor)
            audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_firered_audio(item_idx: int):
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                assert len(audio_embeds.shape) == 2, "audio_embeds must be a 2D tensor"
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(
                    f"The audio (len={audio_len}) is too short "
                    "to be represented inside the model"
                )

            audio_tokens = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_firered_audio,
            )
        ]


class FireRedAudioEncoder(nn.Module):
    def __init__(self, config: FireRedAudioEncoderConfig):
        super().__init__()
        self.encoder_dim = config.odim
        self.input_preprocessor = Conv2dSubsampling(config.idim, config.d_model)
        self.positional_encoding = RelPositionalEncoding(
            config.d_model, config.pe_maxlen
        )
        self.dropout = nn.Dropout(config.residual_dropout)

        self.layer_stack = nn.ModuleList(
            [
                RelPosEmbConformerBlock(
                    config.d_model,
                    config.encoder_attention_heads,
                    config.residual_dropout,
                    config.dropout_rate,
                    config.kernel_size,
                )
                for _ in range(config.encoder_layers)
            ]
        )

    def forward(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor, pad=True
    ):
        if pad:
            padded_input = F.pad(
                padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1),
                "constant",
                0.0,
            )
        src_mask = self.padding_position_is_0(padded_input, input_lengths)

        embed_output, input_lengths, src_mask = self.input_preprocessor(
            padded_input, src_mask
        )
        enc_output = self.dropout(embed_output)

        pos_emb = self.dropout(self.positional_encoding(embed_output))

        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(self, padded_input, input_lengths):
        N, T = padded_input.size()[:2]
        positions = torch.arange(T, device=padded_input.device)
        positions = positions.unsqueeze(0).expand(N, T)
        lengths = input_lengths.unsqueeze(1).expand(N, T)
        mask = (positions < lengths).unsqueeze(1).to(torch.uint8)
        return mask

    def _get_feat_extract_output_lengths(self, feat_frames: int) -> int:
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


class FireRedAudioMultiModalProjector(nn.Module):
    def __init__(
        self,
        audio_hidden_size: int,
        text_hidden_size: int,
        downsample_rate: int = 2,
    ):
        super().__init__()
        self.linear1 = nn.Linear(audio_hidden_size * downsample_rate, text_hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(text_hidden_size, text_hidden_size)
        self.ds = downsample_rate

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.reshape(batch_size, seq_len // self.ds, feat_dim * self.ds)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens


@MULTIMODAL_REGISTRY.register_processor(
    FireRedAsrMultiModalProcessor,
    info=FireRedAsrProcessingInfo,
    dummy_inputs=FireRedAsrDummyInputsBuilder,
)
class FireRedAudioForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "audio":
            return "<speech>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.audio_tower = FireRedAudioEncoder(config.audio_config)
        self.encoder_projector = FireRedAudioMultiModalProjector(
            config.audio_config.d_model,
            config.text_config.hidden_size,
            config.encoder_downsample_rate,
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=config.text_config.architectures,
        )

        self.audio_tower.float().to(self.device)
        self.encoder_projector.float().to(self.device)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> FireRedAudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        audio_feature_lengths = kwargs.pop("speech_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)

        if input_features is None and audio_embeds is None:
            return None

        if input_features is not None:
            return FireRedAudioInputs(
                type="audio_features",
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
                feature_attention_mask=feature_attention_mask,
            )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        speech_features = audio_input["input_features"]
        speech_lengths = audio_input["audio_feature_lengths"]

        encoder_outputs, output_lengths, encoder_mask = self.audio_tower(
            speech_features, speech_lengths
        )

        projected_features, projected_lengths = self.encoder_projector(
            encoder_outputs, output_lengths
        )

        batch_size = projected_features.size(0)
        return tuple(
            projected_features[i, : int(projected_lengths[i].item())]
            for i in range(batch_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights)
        return loaded_params

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
