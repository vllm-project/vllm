# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only NVIDIA Canary (FastConformer encoder + Transformer decoder).

Targets the native Transformers checkpoint layout (e.g. ``nvidia/canary-1b-v2``)
whose ``config.json`` has ``model_type == "canary"``.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar

import regex as re
import torch
from torch import nn
from transformers import BatchFeature, CanaryConfig, ParakeetFeatureExtractor
from transformers import ParakeetEncoder as HFParakeetEncoder

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import MultiModalDataDict, PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing.processor import HFMultiModalInputs
from vllm.renderers import TokenizeParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.collection_utils import is_list_of
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backend import AttentionType

from .cohere_asr import CohereASRAttention, CohereASRCrossAttention, CohereASRMLP
from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from .utils import WeightsMapper, make_layers, maybe_prefix

logger = init_logger(__name__)

# See https://huggingface.co/nvidia/canary-1b-v2
CANARY_SUPPORTED_LANGS = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "ru": "Russian",
    "uk": "Ukrainian",
}


# ----- Decoder -----
class CanaryPositionalEmbedding(nn.Module):
    """Fixed sinusoidal table laid out as ``[sin | cos]`` and scaled by
    ``1 / sqrt(hidden_size)``; matches ``CanaryPositionalEmbedding`` in
    Transformers."""

    def __init__(self, max_positions: int, hidden_size: int) -> None:
        super().__init__()
        half = hidden_size // 2
        log_timescale_increment = math.log(10000.0) * 2 / hidden_size
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(half).float()
        )
        scaled_time = torch.arange(max_positions).float()[:, None] * inv_timescales
        pos_emb = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        self.register_buffer(
            "pos_emb", pos_emb / math.sqrt(hidden_size), persistent=False
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pos_emb[positions]


class CanaryDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.decoder_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        hidden_size = config.hidden_size

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = CohereASRAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            bias=config.attention_bias,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.encoder_attn = CohereASRCrossAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            bias=config.attention_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.final_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = CohereASRMLP(
            embed_dim=hidden_size,
            ffn_dim=config.intermediate_size,
            act_fn=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states=hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        residual = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1})
class CanaryDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.decoder_config
        self.hidden_size = config.hidden_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.pos_emb = CanaryPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.embedding_layernorm = nn.LayerNorm(config.hidden_size)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CanaryDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = self.get_input_embeddings(input_ids, positions)
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        return self.norm(hidden_states)

    def get_input_embeddings(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        pos_emb = self.pos_emb(positions).to(inputs_embeds.dtype)
        return self.embedding_layernorm(inputs_embeds + pos_emb)


class CanaryModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: CanaryConfig = vllm_config.model_config.hf_config
        # Reuses the Transformers FastConformer encoder as-is, like `parakeet.py`.
        self.encoder = HFParakeetEncoder(config.encoder_config).to(
            vllm_config.model_config.dtype
        )
        self.decoder = CanaryDecoder(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "decoder")
        )
        assert config.encoder_config.hidden_size == config.decoder_config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        enc_states = torch.cat(encoder_outputs, dim=0) if encoder_outputs else None
        return self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=enc_states,
        )

    def get_encoder_outputs(
        self,
        input_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        # input_features: (batch, time, num_mel_bins)
        max_len = input_features.shape[1]
        attention_mask = (
            torch.arange(max_len, device=input_features.device)[None, :]
            < feature_lengths[:, None]
        )
        out = self.encoder(
            input_features=input_features, attention_mask=attention_mask
        ).last_hidden_state
        output_lengths = self.encoder._get_subsampling_output_length(feature_lengths)
        with gpu_sync_allowed():
            lengths = output_lengths.tolist()
        return [feat[:length] for feat, length in zip(out, lengths)]


# ----- Multi-modal processing -----
def _subsampled_length(num_frames: int, encoder_config) -> int:
    """Mirror of ``ParakeetEncoder._get_subsampling_output_length``."""
    kernel_size = encoder_config.subsampling_conv_kernel_size
    stride = encoder_config.subsampling_conv_stride
    add_pad = (kernel_size - 1) // 2 * 2 - kernel_size
    length = num_frames
    for _ in range(int(math.log2(encoder_config.subsampling_factor))):
        length = math.floor((length + add_pad) / stride + 1.0)
    return length


class CanaryProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> CanaryConfig:
        return self.ctx.get_hf_config(CanaryConfig)

    def get_default_tok_params(self) -> TokenizeParams:
        # The decoder prompt is built from control tokens; never add BOS/EOS.
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_feature_extractor(self, **kwargs: object) -> ParakeetFeatureExtractor:
        feature_extractor = self.get_hf_processor(**kwargs).feature_extractor
        assert isinstance(feature_extractor, ParakeetFeatureExtractor)
        return feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.get_feature_extractor().sampling_rate,
            target_channels=1,
        )

    @property
    def skip_prompt_length_check(self) -> bool:
        return True  # The encoder prompt is padded

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_num_audio_tokens(self, num_samples: int) -> int:
        fe = self.get_feature_extractor()
        num_frames = (num_samples + fe.n_fft // 2 * 2 - fe.n_fft) // fe.hop_length
        return _subsampled_length(num_frames, self.get_hf_config().encoder_config)


class CanaryDummyInputsBuilder(BaseDummyInputsBuilder[CanaryProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|startofcontext|>" * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        sampling_rate = self.info.get_feature_extractor().sampling_rate
        audio_len = CanaryForConditionalGeneration.max_audio_clip_s * sampling_rate
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class CanaryMultiModalProcessor(EncDecMultiModalProcessor[CanaryProcessingInfo]):
    skip_decoder_start_token: bool = True

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(
        self,
        prompt: list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        # The encoder only consumes audio features; this placeholder is
        # expanded to the number of encoder output frames.
        return [0]

    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_hf_mm_inputs(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> HFMultiModalInputs:
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)
        feature_extractor = self.info.get_feature_extractor(**hf_kwargs)
        return hf_inputs._replace(
            hf_kwargs=dict(
                hf_inputs.hf_kwargs, sampling_rate=feature_extractor.sampling_rate
            )
        )

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # `CanaryProcessor` renames `input_ids` to `decoder_input_ids`, so call
        # the feature extractor and tokenizer directly instead.
        feature_extractor = self.info.get_feature_extractor(**hf_kwargs)
        audios = hf_data.get("audio")
        if audios is None:
            result = BatchFeature()
        else:
            if not isinstance(audios, list):
                audios = [audios]
            # Extract one item at a time so features are never padded to the
            # longest clip in the batch.
            input_features = []
            feature_lengths = []
            for audio in audios:
                features = feature_extractor(
                    audio,
                    sampling_rate=feature_extractor.sampling_rate,
                    return_tensors="pt",
                )
                input_features.append(features["input_features"][0])
                feature_lengths.append(features["attention_mask"][0].sum())
            result = BatchFeature(
                dict(
                    input_features=input_features,
                    feature_lengths=torch.stack(feature_lengths),
                )
            )

        text = hf_data.get("text")
        if text is not None:
            assert isinstance(text, str)
            token_ids = self.info.get_tokenizer().encode(text, add_special_tokens=False)
            result["input_ids"] = torch.tensor([token_ids])
        return result

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            feature_lengths=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def get_replacement(item_idx: int) -> list[int]:
            audios = mm_items.get_items("audio", AudioProcessorItems)
            num_samples = audios.get_audio_length(item_idx)
            return [0] * self.info.get_num_audio_tokens(num_samples)

        return [
            PromptReplacement(modality="audio", target=[0], replacement=get_replacement)
        ]


@MULTIMODAL_REGISTRY.register_processor(
    CanaryMultiModalProcessor,
    info=CanaryProcessingInfo,
    dummy_inputs=CanaryDummyInputsBuilder,
)
class CanaryForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "kv_proj": ["k_proj", "v_proj"],
    }

    # Only the decoder is built from vLLM layers; encoder weights keep their
    # Transformers names.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            re.compile(r"(decoder\.layers\.\d+\.\w+_attn)\.o_proj\."): (
                r"\1.out_projection."
            ),
            re.compile(r"(decoder\.layers\.\d+\.mlp)\.fc1\."): r"\1.dense_in.",
            re.compile(r"(decoder\.layers\.\d+\.mlp)\.fc2\."): r"\1.dense_out.",
            re.compile(r"\.num_batches_tracked$"): None,
        },
    )

    supported_languages = CANARY_SUPPORTED_LANGS
    supports_transcription_only = True
    skip_warmup_audio_preprocessing = True
    # TODO: canary-1b-v2 handles long-form audio with dynamic chunking in NeMo;
    # confirm the right per-chunk limit for vLLM's chunker.
    max_audio_clip_s: ClassVar[int] = 40

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            logger.warning(
                "Defaulting to language='en'. If you wish to transcribe audio in "
                "a different language, pass the `language` field in the "
                "TranscriptionRequest."
            )
            language = "en"
        # The decoder prompt needs one of the trained language tokens, so
        # unlike the base class, do not fall back for other languages.
        if language not in cls.supported_languages:
            raise ValueError(
                f"Unsupported language: {language!r}. Canary supports "
                f"{list(cls.supported_languages)}."
            )
        return language

    @classmethod
    def get_decoder_prompt_tokens(
        cls, source_language: str, target_language: str, punctuation: bool = True
    ) -> list[str]:
        """Same token sequence as the checkpoint's ``chat_template.jinja``."""
        return [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            f"<|{source_language}|>",
            f"<|{target_language}|>",
            "<|pnc|>" if punctuation else "<|nopnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        source_language = stt_params.language
        if source_language is None:
            raise ValueError("Language must be specified for Canary")
        target_language = source_language
        if stt_params.task_type == "translate":
            target_language = stt_params.to_language or "en"
            if target_language not in cls.supported_languages:
                raise ValueError(
                    f"Unsupported target language {target_language!r}; expected "
                    f"one of {list(cls.supported_languages)}."
                )

        tokenizer = cached_tokenizer_from_config(stt_params.model_config)
        tokens = cls.get_decoder_prompt_tokens(source_language, target_language)
        prompt_token_ids = tokenizer.convert_tokens_to_ids(tokens)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if unk_token_id is not None and unk_token_id in prompt_token_ids:
            raise ValueError("Failed to resolve the Canary decoder control tokens.")

        audio = stt_params.audio
        return TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": (audio, stt_params.stt_config.sample_rate)},
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None
        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            max_audio_clip_s=cls.max_audio_clip_s,
            sample_rate=16_000,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        # Matches ParakeetFeatureExtractor defaults (n_fft=512, hop_length=160).
        num_frames = int(audio_duration_s * stt_config.sample_rate) // 160
        return _subsampled_length(num_frames, model_config.hf_config.encoder_config)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: CanaryConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        with self._mark_composite_model(
            vllm_config,
            language_targets=CanaryDecoder,
            tower_targets={"audio": HFParakeetEncoder},
        ):
            self.model = CanaryModel(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )

        decoder_config = config.decoder_config
        self.proj_out = ParallelLMHead(
            decoder_config.vocab_size,
            decoder_config.hidden_size,
            quant_config=quant_config,
            bias=True,
            prefix=maybe_prefix(prefix, "proj_out"),
        )
        self.logits_processor = LogitsProcessor(decoder_config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            encoder_outputs=encoder_outputs or [],
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.model.decoder

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = kwargs.pop("input_features", None)
        feature_lengths = kwargs.pop("feature_lengths", None)
        if input_features is None:
            return []
        if not isinstance(feature_lengths, torch.Tensor):
            raise ValueError("Audio feature lengths must be a tensor.")

        if isinstance(input_features, torch.Tensor):
            features = list(input_features)
        elif is_list_of(input_features, torch.Tensor):
            features = input_features
        else:
            raise ValueError(
                f"Incorrect type of audio features: {type(input_features)}"
            )
        padded = torch.nn.utils.rnn.pad_sequence(
            [f.to(self.dtype) for f in features], batch_first=True, padding_value=0.0
        )
        return self.model.get_encoder_outputs(padded, feature_lengths.reshape(-1))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.proj_out, hidden_states, self.proj_out.bias)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        params_dict.update(self.named_buffers())
        loaded_params: set[str] = set()

        for name, loaded_weight in self.hf_to_vllm_mapper.apply(weights):
            # The encoder is the Transformers module and keeps unfused q/k/v.
            is_decoder = name.startswith("model.decoder.")
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if not is_decoder or shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
