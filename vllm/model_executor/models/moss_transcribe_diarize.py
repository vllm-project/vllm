# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only MOSS-Transcribe-Diarize ASR model.

The checkpoint layout is:

* ``model.whisper_encoder.*``: Whisper-medium encoder weights.
* ``model.vq_adaptor.*``: 4x time-merge projector.
* ``model.language_model.*``: Qwen3-0.6B decoder weights.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import ModalityData, MultiModalDataDict, PromptType, TextPrompt
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    _require_is_multimodal,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.whisper import (
    WhisperEncoder,
    _create_fake_bias_for_k_proj,
)
from vllm.model_executor.models.whisper_utils import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
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
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.tensor_schema import TensorSchema, TensorShape

WHISPER_ENCODER_STRIDE = 2

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"

DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


class MossTranscribeDiarizeAudioInputs(TensorSchema):
    """
    Dimensions:
        - c: Audio chunks
        - m: Mel bins
        - f: Mel frames
        - n: Number of audio items
    """

    type: Literal["audio_features"] = "audio_features"

    input_features: Annotated[
        torch.Tensor | None,
        TensorShape("c", "m", "f"),
    ]
    audio_feature_lengths: Annotated[
        torch.Tensor | None,
        TensorShape("c"),
    ]
    audio_chunk_counts: Annotated[
        torch.Tensor | None,
        TensorShape("n"),
    ]


class MossTranscribeDiarizeEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - n: Number of audio items
        - t: Number of audio tokens
        - h: Hidden size
    """

    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("n", "t", "h", dynamic_dims={"t"}),
    ]


MossTranscribeDiarizeInputs: TypeAlias = (
    MossTranscribeDiarizeAudioInputs | MossTranscribeDiarizeEmbeddingInputs
)


def _compute_total_audio_tokens(
    num_samples: int,
    feature_extractor: Any,
    audio_merge_size: int,
) -> int:
    if num_samples <= 0:
        return 0

    n_samples = int(feature_extractor.n_samples)
    stride = (
        int(feature_extractor.hop_length)
        * WHISPER_ENCODER_STRIDE
        * int(audio_merge_size)
    )
    total = 0
    for start in range(0, num_samples, n_samples):
        chunk_samples = min(n_samples, num_samples - start)
        total += (chunk_samples - 1) // stride + 1
    return total


def _get_max_audio_samples(feature_extractor: Any) -> int:
    if hasattr(feature_extractor, "chunk_length"):
        return int(feature_extractor.chunk_length * feature_extractor.sampling_rate)
    return int(feature_extractor.n_samples)


def _as_audio_embedding_list(audio_embeds: object) -> list[torch.Tensor]:
    if isinstance(audio_embeds, torch.Tensor):
        if audio_embeds.ndim == 2:
            return [audio_embeds]
        if audio_embeds.ndim == 3:
            return list(audio_embeds.unbind(dim=0))
        raise ValueError(
            f"`audio_embeds` must be a 2D or 3D tensor, got {audio_embeds.ndim}D."
        )

    if isinstance(audio_embeds, (list, tuple)) and all(
        isinstance(audio_embed, torch.Tensor) for audio_embed in audio_embeds
    ):
        return list(audio_embeds)

    raise TypeError(
        "`audio_embeds` must be a torch.Tensor or a list of torch.Tensor "
        f"objects, got {type(audio_embeds)!r}."
    )


def _get_required_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or (isinstance(token_id, int) and token_id < 0):
        raise ValueError(f"Tokenizer is missing required token {token!r}.")
    return int(token_id)


def _get_audios_from_mm_data(mm_data: Mapping[str, object]) -> list[Any]:
    mm_data_dict = dict(mm_data)
    audio_data = mm_data_dict.pop("audios", mm_data_dict.pop("audio", []))
    if isinstance(audio_data, list):
        audios = audio_data
    elif isinstance(audio_data, tuple) and not (
        len(audio_data) == 2 and isinstance(audio_data[1], (int, float))
    ):
        audios = list(audio_data)
    elif audio_data is None:
        audios = []
    else:
        audios = [audio_data]

    audio_arrays: list[Any] = []
    for audio in audios:
        if isinstance(audio, (tuple, list)) and len(audio) == 2:
            audio = audio[0]
        audio_arrays.append(audio)
    return audio_arrays


def _add_vllm_audio_metadata(
    processed: BatchFeature,
    num_audios: int,
) -> BatchFeature:
    audio_feature_lengths = processed["audio_feature_lengths"]
    if not isinstance(audio_feature_lengths, torch.Tensor):
        audio_feature_lengths = torch.tensor(audio_feature_lengths, dtype=torch.long)
    audio_feature_lengths = audio_feature_lengths.to(dtype=torch.long)

    audio_chunk_mapping = processed.get("audio_chunk_mapping")
    if audio_chunk_mapping is None:
        if num_audios == 1:
            audio_chunk_mapping = torch.zeros_like(audio_feature_lengths)
        elif audio_feature_lengths.numel() == num_audios:
            audio_chunk_mapping = torch.arange(num_audios, dtype=torch.long)
        else:
            raise ValueError(
                "The MOSS processor did not return `audio_chunk_mapping`, and "
                "the chunk-to-audio mapping cannot be inferred."
            )
    elif not isinstance(audio_chunk_mapping, torch.Tensor):
        audio_chunk_mapping = torch.tensor(audio_chunk_mapping, dtype=torch.long)
    audio_chunk_mapping = audio_chunk_mapping.to(dtype=torch.long)
    if audio_chunk_mapping.numel() != audio_feature_lengths.numel():
        raise ValueError(
            "`audio_chunk_mapping` must contain one item per audio chunk: got "
            f"{audio_chunk_mapping.numel()} mappings for "
            f"{audio_feature_lengths.numel()} chunks."
        )
    if audio_chunk_mapping.numel() > 0 and (
        audio_chunk_mapping.min().item() < 0
        or audio_chunk_mapping.max().item() >= num_audios
    ):
        raise ValueError("`audio_chunk_mapping` contains an out-of-range audio index.")

    audio_chunk_counts = torch.bincount(
        audio_chunk_mapping.cpu(),
        minlength=num_audios,
    ).to(dtype=torch.long)
    audio_token_lengths = torch.zeros(
        num_audios,
        dtype=torch.long,
        device=audio_feature_lengths.device,
    )
    audio_token_lengths.scatter_add_(
        0,
        audio_chunk_mapping.to(device=audio_feature_lengths.device),
        audio_feature_lengths,
    )

    processed["audio_feature_lengths"] = audio_feature_lengths
    processed["audio_chunk_counts"] = audio_chunk_counts
    processed["audio_token_lengths"] = audio_token_lengths
    return processed


class MossTranscribeDiarizeWhisperEncoder(WhisperEncoder):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".fc1.": ".mlp.fc1.", ".fc2.": ".mlp.fc2."},
        orig_to_new_stacked={
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
            ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
            ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config.with_hf_config(
                vllm_config.model_config.hf_config.audio_config
            ),
            prefix=prefix,
        )
        self.audio_merge_size = int(vllm_config.model_config.hf_config.audio_merge_size)
        self.max_encoder_batch: int | None = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = _create_fake_bias_for_k_proj(weights, ".k_proj.weight")
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def forward(
        self,
        input_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if input_features.numel() == 0:
            return input_features.new_empty((1, 0, self.conv1.out_channels))
        device = self.conv1.weight.device
        dtype = self.conv1.weight.dtype
        input_features = input_features.to(device=device, dtype=dtype)
        audio_feature_lengths = audio_feature_lengths.to(device=device)
        batch_size = self.max_encoder_batch or input_features.shape[0]
        encoded_parts: list[torch.Tensor] = []

        for start in range(0, input_features.shape[0], batch_size):
            encoded_parts.append(
                super().forward([input_features[start : start + batch_size]])
            )

        hidden = torch.cat(encoded_parts, dim=0)
        chunks = [
            hidden[idx : idx + 1, : int(token_len.item()) * self.audio_merge_size]
            for idx, token_len in enumerate(audio_feature_lengths)
        ]
        return torch.cat(chunks, dim=1)


class MossTranscribeDiarizeVQAdaptor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=eps, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _mtd_field_config(
    hf_inputs: Mapping[str, torch.Tensor],
) -> Mapping[str, MultiModalFieldConfig]:
    fields: dict[str, MultiModalFieldConfig] = {}
    if "audio_embeds" in hf_inputs:
        fields["audio_embeds"] = MultiModalFieldConfig.batched("audio")
    if "audio_chunk_counts" in hf_inputs:
        audio_chunk_counts = hf_inputs["audio_chunk_counts"]
        fields.update(
            input_features=MultiModalFieldConfig.flat_from_sizes(
                "audio",
                audio_chunk_counts,
            ),
            audio_feature_lengths=MultiModalFieldConfig.flat_from_sizes(
                "audio",
                audio_chunk_counts,
            ),
            audio_chunk_counts=MultiModalFieldConfig.batched("audio"),
            audio_token_lengths=MultiModalFieldConfig.batched("audio"),
        )
    return fields


class MossTranscribeDiarizeMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_mtd_field_config,
            )

        return super()._parse_audio_data(data)


class MossTranscribeDiarizeProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_hf_processor(self, **kwargs: object) -> Any:
        return self.ctx.get_hf_processor(**kwargs)

    def get_feature_extractor(self, **kwargs: object) -> Any:
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MossTranscribeDiarizeMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        if mm_counts.get("audio", 0) <= 0:
            return {}

        feature_extractor = self.get_feature_extractor()
        max_audio_samples = _get_max_audio_samples(feature_extractor)
        max_audio_tokens = _compute_total_audio_tokens(
            max_audio_samples,
            feature_extractor,
            self.get_hf_processor().audio_merge_size,
        )
        return {"audio": min(seq_len, max_audio_tokens)}


class MossTranscribeDiarizeDummyInputsBuilder(
    BaseDummyInputsBuilder[MossTranscribeDiarizeProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return {}

        feature_extractor = self.info.get_feature_extractor()
        return {
            "audio": self._get_dummy_audios(
                length=_get_max_audio_samples(feature_extractor),
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> ProcessorInputs:
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data)
        num_audios = mm_counts.get("audio", 0)
        tokenizer = self.info.get_tokenizer()
        prompt = tokenizer.encode(
            AUDIO_PLACEHOLDER * num_audios,
            add_special_tokens=False,
        ) or tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )
        return ProcessorInputs(prompt=prompt, mm_data_items=dummy_mm_items)


class MossTranscribeDiarizeMultiModalProcessor(
    BaseMultiModalProcessor[MossTranscribeDiarizeProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        audios = _get_audios_from_mm_data(mm_data)
        if not audios:
            input_ids = tokenizer.encode(
                prompt,
                add_special_tokens=tok_kwargs.get("add_special_tokens", False),
            )
            return BatchFeature({"input_ids": [input_ids]}, tensor_type="pt")

        processed = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, audio=audios),
            dict(**mm_kwargs, **tok_kwargs),
        )
        return _add_vllm_audio_metadata(processed, len(audios))

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return mm_items.get_count("audio", strict=False) > 0

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _mtd_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_mm_data = out_mm_kwargs.get_data()
        audio_token_lengths_tensor = out_mm_data.get("audio_token_lengths")
        if audio_token_lengths_tensor is None:
            audio_embeds = out_mm_data.get("audio_embeds")
            if audio_embeds is None:
                audio_token_lengths: list[int] = []
            else:
                audio_token_lengths = [
                    int(audio_embed.shape[0])
                    for audio_embed in _as_audio_embedding_list(audio_embeds)
                ]
        else:
            if not isinstance(audio_token_lengths_tensor, torch.Tensor):
                raise TypeError(
                    "`audio_token_lengths` must be a torch.Tensor, got "
                    f"{type(audio_token_lengths_tensor)!r}."
                )
            audio_token_lengths = [
                int(length) for length in audio_token_lengths_tensor.tolist()
            ]
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        audio_start_id = _get_required_token_id(tokenizer, processor.audio_start_token)
        audio_token_id = int(processor.audio_token_id)
        audio_end_id = _get_required_token_id(tokenizer, processor.audio_end_token)

        def get_num_tokens(item_idx: int) -> int:
            if item_idx >= len(audio_token_lengths):
                raise ValueError(
                    "Cannot determine the number of audio tokens for audio item "
                    f"{item_idx}."
                )
            num_tokens = audio_token_lengths[item_idx]
            if num_tokens <= 0:
                raise ValueError("Audio input is too short to produce any tokens.")
            return num_tokens

        def get_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            num_tokens = get_num_tokens(item_idx)
            audio_tokens = processor._audio_span_ids(num_tokens)
            return PromptUpdateDetails.select_token_id(
                [audio_start_id] + audio_tokens + [audio_end_id],
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=AUDIO_PLACEHOLDER,
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MossTranscribeDiarizeMultiModalProcessor,
    info=MossTranscribeDiarizeProcessingInfo,
    dummy_inputs=MossTranscribeDiarizeDummyInputsBuilder,
)
class MossTranscribeDiarizeForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
):
    supports_transcription = True
    supports_transcription_only = True
    supports_segment_timestamp = False
    supported_languages = ISO639_1_SUPPORTED_LANGS
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "language_model.embed_tokens.": "language_model.model.embed_tokens.",
            "language_model.norm.": "language_model.model.norm.",
            "model.language_model.model.": "language_model.model.",
            "model.language_model.lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.whisper_encoder.": "whisper_encoder.",
            "model.vq_adaptor.": "vq_adaptor.",
            "model.": None,
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        return AUDIO_PLACEHOLDER if modality.startswith("audio") else None

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: str,
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        return SpeechToTextConfig(
            max_audio_clip_s=None,
            sample_rate=processor.feature_extractor.sampling_rate,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        processor = cached_processor_from_config(model_config)
        num_samples = math.ceil(audio_duration_s * stt_config.sample_rate)
        return _compute_total_audio_tokens(
            num_samples,
            processor.feature_extractor,
            processor.audio_merge_size,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        stt_config = stt_params.stt_config
        question = stt_params.request_prompt or DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT
        question = question.strip() or DEFAULT_MOSS_TRANSCRIBE_DIARIZE_PROMPT
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{AUDIO_PLACEHOLDER}\n"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return TextPrompt(
            prompt=prompt,
            multi_modal_data={"audio": (stt_params.audio, stt_config.sample_rate)},
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return text.strip()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype

        with self._mark_tower_model(vllm_config, "audio"):
            self.whisper_encoder = MossTranscribeDiarizeWhisperEncoder(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "whisper_encoder"),
            )
            self.vq_adaptor = MossTranscribeDiarizeVQAdaptor(
                input_dim=int(self.config.adaptor_input_dim),
                hidden_size=int(self.config.text_config.hidden_size),
                eps=float(self.config.text_config.rms_norm_eps),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen3ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _time_merge(self, features: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = features.shape
        merge_size = int(self.config.audio_merge_size)
        seq_len_trim = (seq_len // merge_size) * merge_size
        return features[:, :seq_len_trim, :].reshape(
            batch,
            seq_len_trim // merge_size,
            dim * merge_size,
        )

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> MossTranscribeDiarizeInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        audio_chunk_counts = kwargs.pop("audio_chunk_counts", None)
        if input_features is None and audio_embeds is None:
            return None
        if audio_embeds is not None:
            return MossTranscribeDiarizeEmbeddingInputs(
                type="audio_embeds",
                audio_embeds=_as_audio_embedding_list(audio_embeds),
            )
        return MossTranscribeDiarizeAudioInputs(
            type="audio_features",
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_counts=audio_chunk_counts,
        )

    def _process_audio_input(
        self, audio_input: MossTranscribeDiarizeInputs
    ) -> list[torch.Tensor]:
        if audio_input["type"] == "audio_embeds":
            return list(audio_input["audio_embeds"])

        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]
        audio_chunk_counts = audio_input["audio_chunk_counts"]
        if input_features is None or audio_feature_lengths is None:
            raise ValueError(
                "MOSS-Transcribe-Diarize audio inputs require both "
                "`input_features` and `audio_feature_lengths`."
            )
        if audio_feature_lengths.numel() != input_features.shape[0]:
            raise ValueError(
                "`audio_feature_lengths` must contain one length per "
                "`input_features` chunk: got "
                f"{audio_feature_lengths.numel()} lengths for "
                f"{input_features.shape[0]} chunks."
            )
        if audio_chunk_counts is None:
            audio_chunk_counts = audio_feature_lengths.new_tensor(
                [input_features.shape[0]],
                dtype=torch.long,
            )
        else:
            audio_chunk_counts = audio_chunk_counts.to(dtype=torch.long)
        if audio_chunk_counts.numel() == 0:
            raise ValueError("`audio_chunk_counts` must contain at least one item.")
        if torch.any(audio_chunk_counts <= 0):
            raise ValueError("`audio_chunk_counts` must contain positive counts.")
        num_audio_chunks = int(audio_chunk_counts.sum().item())
        if num_audio_chunks != input_features.shape[0]:
            raise ValueError(
                "`audio_chunk_counts` must sum to the number of input feature chunks: "
                f"got {num_audio_chunks} chunks for "
                f"{input_features.shape[0]} input chunks."
            )

        features = self.whisper_encoder(input_features, audio_feature_lengths)
        merged = self._time_merge(features.to(dtype=self.dtype))
        projected = self.vq_adaptor(merged).squeeze(0)

        audio_chunk_offsets = torch.cumsum(audio_chunk_counts, dim=0)
        audio_chunk_offsets = torch.cat(
            [audio_chunk_offsets.new_zeros(1), audio_chunk_offsets]
        )
        tokens_per_item = [
            int(audio_feature_lengths[start:end].sum().item())
            for start, end in zip(
                audio_chunk_offsets[:-1].tolist(),
                audio_chunk_offsets[1:].tolist(),
            )
        ]
        if any(num_tokens <= 0 for num_tokens in tokens_per_item):
            raise ValueError("Audio input is too short to produce any tokens.")
        return list(projected.split(tokens_per_item, dim=0))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio_input(audio_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
