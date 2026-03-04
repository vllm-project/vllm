# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    BatchFeature,
    Qwen2Config,
)

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.whisper_utils import (
    ISO639_1_SUPPORTED_LANGS,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.transformers_utils.processors.fireredasr2_processor import (
    FireRedASR2FeatureExtractor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
    _require_is_multimodal,
)
from .qwen2 import Qwen2ForCausalLM
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)

logger = init_logger(__name__)


class FireRedASR2AudioInputs(TensorSchema):
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
    speech_lengths: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b"),
    ]
    fake_token_lengths: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b"),
    ]


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = ReplicatedLinear(
            input_size=out_channels * subsample_idim,
            output_size=d_model,
            bias=True,
        )

        self.subsampling = 4
        left_context = right_context = 3  # both exclude currect frame
        self.context = left_context + 1 + right_context  # 7

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        N, C, T, D = x.size()
        x, _ = self.out(x.transpose(1, 2).contiguous().view(N, T, C * D))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)).item() / d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self.pe = torch.cat([pe_positive, pe_negative], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tmax = 2 * max_len - 1
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T].clone().detach()
        return pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.linear_expand = ReplicatedLinear(
            input_size=d_model,
            output_size=d_model * 4,
            bias=True,
        )
        self.nonlinear = Swish()
        self.linear_project = ReplicatedLinear(
            input_size=d_model * 4,
            output_size=d_model,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_layer_norm(x)
        x, _ = self.linear_expand(x)
        x = self.nonlinear(x)
        x, _ = self.linear_project(x)
        output = x + residual
        return output


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = ReplicatedLinear(
            input_size=d_model, output_size=n_head * self.d_k, bias=False
        )
        self.w_ks = ReplicatedLinear(
            input_size=d_model, output_size=n_head * self.d_k, bias=False
        )
        self.w_vs = ReplicatedLinear(
            input_size=d_model, output_size=n_head * self.d_v, bias=False
        )

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.fc = ReplicatedLinear(
            input_size=n_head * self.d_v, output_size=d_model, bias=False
        )

    def forward_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q)[0].view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k)[0].view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v)[0].view(sz_b, len_v, n_head, d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_output(
        self, output: torch.Tensor, residual: torch.Tensor, sz_b: int, len_q: int
    ) -> torch.Tensor:
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        fc_out, _ = self.fc(output)
        output = fc_out
        output = output + residual
        return output

    def forward_attention(
        self, attn: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -float("inf"))
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)

        d_attn = attn
        output = torch.matmul(d_attn, v)

        return output, attn


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head: int, d_model: int):
        super().__init__(n_head, d_model)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k**0.5)
        self.linear_pos = ReplicatedLinear(
            input_size=d_model, output_size=n_head * d_k, bias=False
        )
        self.pos_bias_u = nn.Parameter(torch.empty([n_head, d_k]))
        self.pos_bias_v = nn.Parameter(torch.empty([n_head, d_k]))

    def _rel_shift(self, x):
        N, H, T1, T2 = x.size()
        zero_pad = torch.zeros((N, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(N, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb)[0].view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn_scores = matrix_ac + matrix_bd
        attn_scores.mul_(self.scale)

        output, attn = self.forward_attention(attn_scores, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 33):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 4, kernel_size=1, bias=False
        )
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
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            d_model * 2, d_model, kernel_size=1, bias=False
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

        out = self.pointwise_conv2(out)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = out.transpose(1, 2)
        return out + residual


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model, n_head, kernel_size=33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model)
        self.conv = ConformerConvolution(d_model, kernel_size)
        self.ffn2 = ConformerFeedForward(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        slf_attn_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        n_layers_enc: int,
        n_head: int,
        d_model: int,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
    ):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)

        self.layer_stack = nn.ModuleList()
        for _ in range(n_layers_enc):
            block = RelPosEmbConformerBlock(d_model, n_head, kernel_size)
            self.layer_stack.append(block)

    def forward(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor, pad: bool = True
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
        enc_output = embed_output

        pos_emb = self.positional_encoding(embed_output)

        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        N, T = padded_input.size()[:2]
        mask = torch.ones((N, T)).to(padded_input.device)
        for i in range(N):
            mask[i, input_lengths[i] :] = 0
        mask = mask.unsqueeze(dim=1)
        return mask.to(torch.uint8)


class FireRedASR2Adapter(nn.Module):
    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int = 2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = ReplicatedLinear(
            input_size=encoder_dim * downsample_rate,
            output_size=llm_dim,
            bias=True,
        )
        self.relu = _ACTIVATION_REGISTRY["relu"]
        self.linear2 = ReplicatedLinear(
            input_size=llm_dim,
            output_size=llm_dim,
            bias=True,
        )

    def forward(self, x, x_lens):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.ds, feat_dim * self.ds)

        x, _ = self.linear1(x)
        x = self.relu(x)
        x, _ = self.linear2(x)

        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens


class FireRedASR2Encoder(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
    ):
        super().__init__()
        self.audio_encoder = ConformerEncoder(
            **vllm_config.model_config.hf_config.audio_encoder_conf
        )


class FireRedASR2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = FireRedASR2Encoder(
            vllm_config=vllm_config,
        )
        encoder_dim = self.encoder.audio_encoder.odim
        llm_dim = vllm_config.model_config.hf_config.hidden_size
        self.encoder_projector = FireRedASR2Adapter(
            encoder_dim,
            llm_dim,
            vllm_config.model_config.hf_config.encoder_downsample_rate,
        )

        self.decoder = Qwen2ForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "decoder")
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        return decoder_outputs

    def get_encoder_outputs(
        self,
        speech: torch.Tensor | list[torch.Tensor] | None,
        speech_lengths: torch.Tensor | list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        encoder_outs, enc_lengths, enc_mask = self.encoder.audio_encoder(
            speech, speech_lengths
        )
        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)
        return speech_features


class FireRedASR2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Qwen2Config:
        return self.ctx.get_hf_config(Qwen2Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_extractor(self, **kwargs: object) -> FireRedASR2FeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, FireRedASR2FeatureExtractor)
        return feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.get_target_channels(),
        )

    def get_target_channels(self) -> int:
        return 1


class FireRedASR2DummyInputsBuilder(BaseDummyInputsBuilder[FireRedASR2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|AUDIO|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio")

        ret = {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }
        return ret


class FireRedASR2MultiModalProcessor(
    BaseMultiModalProcessor[FireRedASR2ProcessingInfo]
):
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
            fake_token_lengths=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")

        audio_token_id = vocab[audio_token]

        out_mm_data = out_mm_kwargs.get_data()

        fake_token_lengths = out_mm_data.get("fake_token_lengths")

        if fake_token_lengths is None:
            audio_output_lengths = []
        else:
            assert isinstance(fake_token_lengths, torch.Tensor)

            audio_output_lengths = fake_token_lengths.tolist()

        def get_replacement_fireredasr2_audio(item_idx: int):
            num_features = audio_output_lengths[item_idx]

            audio_tokens = [audio_token_id] * int(num_features)

            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement_fireredasr2_audio,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    FireRedASR2MultiModalProcessor,
    info=FireRedASR2ProcessingInfo,
    dummy_inputs=FireRedASR2DummyInputsBuilder,
)
class FireRedASR2ForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "encoder_attn.kv_proj": ["encoder_attn.k_proj", "encoder_attn.v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "llm.": "model.decoder.",
            "encoder.": "model.encoder.audio_encoder.",
            "encoder_projector.": "model.encoder_projector.",
            "net.0": "pre_layer_norm",
            "net.1": "linear_expand",
            "net.4": "linear_project",
        }
    )

    supports_transcription_only = True
    supports_segment_timestamp = True
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            # TODO language should be optional and can be guessed.
            # For now we default to en. See
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py#L1520
            logger.warning(
                "Defaulting to language='en'. If you wish to transcribe "
                "audio in a different language, pass the `language` field "
                "in the TranscriptionRequest."
            )
            language = "en"
        return super().validate_language(language)

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
                "Language must be specified when creating the fireredasr2 prompt"
            )

        prompt_str = "<|im_start|>user\n<|AUDIO|>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        prompt = {
            "prompt": prompt_str,
            "multi_modal_data": {
                "audio": (audio, stt_config.sample_rate),
            },
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

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        processor = cached_processor_from_config(model_config)
        hop_length = processor.feature_extractor.hop_length
        assert hop_length is not None
        return math.ceil(audio_duration_s * stt_config.sample_rate / hop_length)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.model = FireRedASR2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        logit_scale = getattr(config, "logit_scale", 1.0)

        self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        decoder_outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        return decoder_outputs

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)

        speech = audio_input["input_features"]
        speech_lengths = audio_input["speech_lengths"].to(torch.int32)
        enc_output = self.model.get_encoder_outputs(
            speech=speech, speech_lengths=speech_lengths
        )

        return enc_output

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self.model.decoder.embed_input_ids(input_ids)

        ret = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )
        return ret

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> FireRedASR2AudioInputs:
        input_features = kwargs.pop("input_features", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        fake_token_lengths = kwargs.pop("fake_token_lengths", None)

        return FireRedASR2AudioInputs(
            input_features=input_features,
            speech_lengths=speech_lengths,
            fake_token_lengths=fake_token_lengths,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.model.decoder.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self, skip_prefixes=["model.encoder.audio_encoder.positional_encoding.pe"]
        )

        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
