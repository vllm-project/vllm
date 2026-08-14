# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FireRedLID – Language Identification model adapted for vLLM.

Architecture:  ConformerEncoder  +  TransformerDecoder (6-layer cross-attn)
Vocabulary:    120 LID tokens  (dict.txt)
Output:        Up to 2 tokens  (e.g. "en", "zh mandarin")

This implementation follows the Whisper-style encoder-decoder pattern:
  • Encoder processes audio features (Fbank + CMVN via FeatureExtractor)
  • Decoder performs single-step autoregressive forward
  • vLLM's generation loop handles beam search / sampling
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import ModelConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict, PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .conformer_encoder import ConformerEncoder
from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from .whisper_utils import ISO639_1_SUPPORTED_LANGS

logger = init_logger(__name__)


class FireRedLIDAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - t: Time frames  (variable across utterances)
        - nmb: Number of mel bins (80)
    """

    input_features: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b", "t", "nmb", dynamic_dims={"t"}),
    ]
    speech_lengths: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b"),
    ]
    fake_token_lengths: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b"),
    ]


FireRedLIDEncoder = ConformerEncoder


class FireRedLIDPositionalEmbedding(nn.Module):
    """Absolute sinusoidal positional embedding indexed by `positions`."""

    def __init__(self, d_model: int, max_len: int = 5000):
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
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe[position_ids]


class FireRedLIDAttention(nn.Module):
    """Base attention with shared QKV/FC projections for the LID decoder."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert n_head % tp_size == 0
        self.total_num_heads = n_head
        self.num_heads = n_head // tp_size
        self.num_kv_heads = max(1, n_head // tp_size)
        self.head_dim = d_model // n_head
        self.scaling = self.head_dim**-0.5

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.w_qs = ColumnParallelLinear(
            d_model,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w_qs",
        )
        self.w_ks = ColumnParallelLinear(
            d_model,
            d_model,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w_ks",
        )
        self.w_vs = ColumnParallelLinear(
            d_model,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w_vs",
        )
        self.fc = RowParallelLinear(
            d_model,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )
        self._init_attn(cache_config, quant_config, prefix)

    def _init_attn(self, cache_config, quant_config, prefix: str) -> None:
        raise NotImplementedError


class FireRedLIDSelfAttention(FireRedLIDAttention):
    def _init_attn(self, cache_config, quant_config, prefix: str) -> None:
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q, _ = self.w_qs(hidden_states)
        k, _ = self.w_ks(hidden_states)
        v, _ = self.w_vs(hidden_states)
        attn_output = self.attn(q, k, v)
        output, _ = self.fc(attn_output)
        return output


class FireRedLIDCrossAttention(FireRedLIDAttention):
    def _init_attn(self, cache_config, quant_config, prefix: str) -> None:
        self.attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        q, _ = self.w_qs(hidden_states)
        if encoder_hidden_states is not None:
            k, _ = self.w_ks(encoder_hidden_states)
            v, _ = self.w_vs(encoder_hidden_states)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)
        output, _ = self.fc(attn_output)
        return output


class FireRedLIDFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = ReplicatedLinear(d_model, d_ff, bias=True)
        self.act = nn.GELU()
        self.w_2 = ReplicatedLinear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w_1(x)
        x = self.act(x)
        x, _ = self.w_2(x)
        return x


class FireRedLIDDecoderLayer(nn.Module):
    """vLLM-native decoder layer while preserving FireRedLID parameter names."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = FireRedLIDSelfAttention(
            d_model,
            n_head,
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = FireRedLIDCrossAttention(
            d_model,
            n_head,
            vllm_config=vllm_config,
            prefix=f"{prefix}.cross_attn",
        )

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = FireRedLIDFFN(d_model, d_model * 4)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states = self.cross_attn(hidden_states, encoder_hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states


class FireRedLIDDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.pad_id = getattr(config, "pad_token_id", 2)
        self.n_layers = getattr(config, "n_layers_lid_dec", 6)
        self.d_model = getattr(config, "d_model", 1280)
        self.scale = self.d_model**0.5

        self.tgt_word_emb = nn.Embedding(
            getattr(config, "vocab_size", 120),
            self.d_model,
            padding_idx=self.pad_id,
        )
        self.positional_encoding = FireRedLIDPositionalEmbedding(
            self.d_model,
            max_len=getattr(config, "pe_maxlen", 5000),
        )

        self.layer_stack = nn.ModuleList(
            [
                FireRedLIDDecoderLayer(
                    self.d_model,
                    getattr(config, "n_head", 20),
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layer_stack.{idx}",
                )
                for idx in range(self.n_layers)
            ]
        )
        self.layer_norm_out = nn.LayerNorm(self.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = self.tgt_word_emb(input_ids) * self.scale
        hidden_states = hidden_states + self.positional_encoding(positions)

        for layer in self.layer_stack:
            hidden_states = layer(hidden_states, encoder_hidden_states)

        hidden_states = self.layer_norm_out(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tgt_word_emb(input_ids)


class FireRedLIDModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.encoder = FireRedLIDEncoder(
            idim=getattr(config, "idim", 80),
            n_layers_enc=getattr(config, "n_layers_enc", 16),
            n_head=getattr(config, "n_head", 20),
            d_model=getattr(config, "d_model", 1280),
            kernel_size=getattr(config, "kernel_size", 33),
            pe_maxlen=getattr(config, "pe_maxlen", 5000),
        )

        self.decoder = FireRedLIDDecoder(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "decoder"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        enc_states = (
            torch.cat(encoder_outputs, dim=0)
            if encoder_outputs and len(encoder_outputs) > 0
            else None
        )
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=enc_states,
        )
        return decoder_outputs

    def get_encoder_outputs(
        self,
        speech: torch.Tensor | list[torch.Tensor],
        speech_lengths: torch.Tensor | list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder and return padded outputs plus true sequence lengths."""
        enc_output, enc_lengths, _ = self.encoder(speech, speech_lengths)
        return enc_output, enc_lengths


class FireRedLIDProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_extractor(self, **kwargs):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor
        return feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
        )

    @property
    def skip_prompt_length_check(self) -> bool:
        return True

    def get_num_audio_tokens(self) -> int:
        # For encoder profiling – return a reasonable dummy length.
        # This doesn't affect actual inference since encoder processes
        # variable-length features.
        return 1


class FireRedLIDDummyInputsBuilder(BaseDummyInputsBuilder[FireRedLIDProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<sos>"

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
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class FireRedLIDMultiModalProcessor(
    EncDecMultiModalProcessor[FireRedLIDProcessingInfo]
):
    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        # Dummy encoder prompt for profiling (encoder only processes audio).
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
            fake_token_lengths=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_mm_data = out_mm_kwargs.get_data()
        fake_token_lengths = out_mm_data.get("fake_token_lengths")

        if fake_token_lengths is None:
            # Fallback to max encoder output length if not available
            audio_output_lengths = []
        else:
            assert isinstance(fake_token_lengths, torch.Tensor)
            audio_output_lengths = fake_token_lengths.tolist()

        def get_replacement(item_idx: int):
            if audio_output_lengths:
                num_tokens = int(audio_output_lengths[item_idx])
            else:
                num_tokens = self.info.get_num_audio_tokens()
            return [0] * num_tokens

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=get_replacement,
            )
        ]


# FireRedLID supports a wider set of languages than Whisper's shared list.
# Only ISO 639-1 codes are listed; FireRedLID's dialect tokens (mandarin,
# xinan, wu, …) are output tokens but not valid language *request* codes.
_FIREREDLID_SUPPORTED_LANGUAGES: Mapping[str, str] = {
    **ISO639_1_SUPPORTED_LANGS,
    "am": "Amharic",
    "as": "Assamese",
    "ba": "Bashkir",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "eu": "Basque",
    "fo": "Faroese",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "ht": "Haitian Creole",
    "jw": "Javanese",
    "ka": "Georgian",
    "km": "Khmer",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "mg": "Malagasy",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mt": "Maltese",
    "my": "Myanmar",
    "nn": "Nynorsk",
    "oc": "Occitan",
    "pa": "Panjabi",
    "ps": "Pashto",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "su": "Sundanese",
    "te": "Telugu",
    "tg": "Tajik",
    "tk": "Turkmen",
    "tt": "Tatar",
    "uz": "Uzbek",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
}


@MULTIMODAL_REGISTRY.register_processor(
    FireRedLIDMultiModalProcessor,
    info=FireRedLIDProcessingInfo,
    dummy_inputs=FireRedLIDDummyInputsBuilder,
)
class FireRedLIDForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    # -- SupportsTranscription protocol attributes --
    supports_transcription_only = True
    supported_languages = _FIREREDLID_SUPPORTED_LANGUAGES

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "encoder.": "model.encoder.",
            "lid_decoder.": "model.decoder.",
            # Encoder FFN: nn.Sequential indices → named children
            "net.0": "pre_layer_norm",
            "net.1": "linear_expand",
            "net.4": "linear_project",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        with self._mark_composite_model(
            vllm_config,
            language_targets=FireRedLIDDecoder,
            tower_targets={"audio": FireRedLIDEncoder},
        ):
            self.model = FireRedLIDModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        self.proj_out = ParallelLMHead(
            getattr(config, "vocab_size", 120),
            getattr(config, "d_model", 1280),
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "proj_out"),
        )
        self.proj_out = self.proj_out.tie_weights(self.model.decoder.tgt_word_emb)

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            getattr(config, "vocab_size", 120),
            scale=logit_scale,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_outputs is None:
            encoder_outputs = []
        decoder_outputs = self.model(
            input_ids=input_ids,
            positions=positions,
            encoder_outputs=encoder_outputs,
        )
        return decoder_outputs

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Run encoder on audio features and return per-item embeddings."""
        audio_input = self._parse_and_validate_audio_input(**kwargs)

        speech = audio_input["input_features"]
        speech_lengths = audio_input["speech_lengths"]
        if speech is None or speech_lengths is None:
            return []

        # When audio items have different time lengths, vLLM's
        # MultiModalBatchedField._reduce_data returns a plain
        # list[Tensor] instead of a stacked Tensor.  The encoder
        # expects a padded [B, Tmax, feat_dim] Tensor, so we
        # normalise both speech and speech_lengths here.
        if isinstance(speech, (list, tuple)):
            # Each element: [Ti, feat_dim]  (or [1, Ti, feat_dim])
            tensors = [
                s.squeeze(0) if s.dim() == 3 and s.size(0) == 1 else s for s in speech
            ]
            device = tensors[0].device
            dtype = tensors[0].dtype
            feat_dim = tensors[0].shape[-1]
            lengths = torch.tensor(
                [t.size(0) for t in tensors],
                device=device,
                dtype=torch.int32,
            )
            t_max = int(lengths.max().item())
            # Pre-allocate zero-padded batch tensor
            speech = torch.zeros(
                (len(tensors), t_max, feat_dim),
                device=device,
                dtype=dtype,
            )
            for i, t in enumerate(tensors):
                speech[i, : t.size(0)] = t
            speech_lengths = lengths
        else:
            # Already a batched Tensor [B, T, feat_dim]
            if speech.dim() == 2:
                speech = speech.unsqueeze(0)

        speech_lengths = torch.as_tensor(
            speech_lengths, dtype=torch.int32, device=speech.device
        )

        enc_output, enc_lengths = self.model.get_encoder_outputs(
            speech=speech,
            speech_lengths=speech_lengths,
        )

        # vLLM expects one 2D tensor per multimodal item. Slice each batch entry
        # by the true encoder length so cross-attention never sees padded frames.
        return tuple(
            enc_output[i, : max(0, int(enc_lengths[i].item()))]
            for i in range(enc_output.size(0))
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.decoder.embed_input_ids(input_ids)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> FireRedLIDAudioInputs:
        input_features = kwargs.pop("input_features", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        fake_token_lengths = kwargs.pop("fake_token_lengths", None)
        return FireRedLIDAudioInputs(
            input_features=input_features,
            speech_lengths=speech_lengths,
            fake_token_lengths=fake_token_lengths,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.proj_out, hidden_states)
        return logits

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        # FireRedLID is a language *identification* model – the caller does
        # not need to specify a language up-front.  Accept None silently.
        if language is None:
            return None
        return super().validate_language(language)

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Build the prompt for the FireRedLID encoder-decoder model.

        The decoder receives a single <sos> token; the encoder processes
        the raw audio waveform via the multimodal pipeline.
        """
        prompt: PromptType = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (audio, int(stt_config.sample_rate)),
                },
            },
            "decoder_prompt": {
                "prompt": "<sos>",
            },
        }
        return prompt

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: Literal["transcribe", "translate"],
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
            # LID output is at most 2 tokens – no chunking needed.
            min_energy_split_window_size=None,
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        # Strip any leading/trailing whitespace from the raw LID output.
        return text.strip()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                # Position encoding buffers are rebuilt at init
                "model.encoder.positional_encoding.pe",
                "model.decoder.positional_encoding.pe",
                # Tied output projection (shared with embedding)
                "model.decoder.tgt_word_prj.weight",
                "proj_out.",
            ],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
