# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, cast

import numpy as np
import torch
from torch import nn
from transformers import (
    BatchFeature,
    Qwen2Config,
)

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict, PromptType
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
from vllm.transformers_utils.processors.fireredasr2 import (
    FireRedASR2FeatureExtractor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .conformer_encoder import ConformerEncoder
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

        with self._mark_composite_model(
            vllm_config,
            language_targets=Qwen2ForCausalLM,
            tower_targets={"audio": (FireRedASR2Encoder, FireRedASR2Adapter)},
        ):
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
