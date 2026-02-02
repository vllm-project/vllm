# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property, partial
from math import ceil
from typing import Literal, cast

import numpy as np
import regex as re
import torch
import torch.nn as nn
from mistral_common.audio import mel_filter_bank
from mistral_common.protocol.instruct.chunk import AudioChunk, RawAudio, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioEncoder,
)
from transformers import BatchFeature, TensorType, WhisperConfig
from transformers.tokenization_utils_base import TextInput

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.whisper import (
    WhisperEncoder,
    _create_fake_bias_for_k_proj,
)
from vllm.model_executor.models.whisper_causal import WhisperCausalEncoder
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
    NestedTensors,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder, ProcessorInputs
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.mistral import MistralTokenizer

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsTranscription
from .utils import init_vllm_registered_model, maybe_prefix

logger = init_logger(__name__)

ISO639_1_SUPPORTED_LANGS = {
    "ar": "Arabic",
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
}


class VoxtralProcessorAdapter:
    """
    Provide a HF-compatible interface for
    :class:`mistral_common.tokens.tokenizers.multimodal.AudioEncoder`.
    """

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @cached_property
    def _audio_processor(self) -> AudioEncoder:
        audio_encoder = self.tokenizer.instruct.audio_encoder
        assert isinstance(audio_encoder, AudioEncoder)
        return audio_encoder

    @cached_property
    def audio_token_id(self) -> int:
        return self._audio_processor.special_ids.audio

    @cached_property
    def begin_audio_token_id(self) -> int:
        return self._audio_processor.special_ids.begin_audio

    @cached_property
    def sampling_rate(self) -> int:
        return self._audio_processor.audio_config.sampling_rate

    @cached_property
    def frame_rate(self) -> float:
        return self._audio_processor.audio_config.frame_rate

    def get_num_audio_tokens(
        self,
        audio_length: int,
    ) -> int:
        return ceil(audio_length / (self.sampling_rate // self.frame_rate))

    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        audios: np.ndarray | list[np.ndarray] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> Mapping[str, NestedTensors]:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if audios is None:
            audios = []
        if not isinstance(audios, list):
            audios = [audios]

        if not audios:
            input_ids = self.tokenizer(text).input_ids
            return {"input_ids": torch.tensor(input_ids)}

        # Allow dummy text, which is used for profiling as well as token inputs
        if any(len(t) > 0 for t in text):
            raise ValueError(
                "You've passed text inputs instead of token inputs. "
                "Make sure to process your input via `mistral_common`'s "
                "tokenizer or pass a chat completion request. "
                "For more info, see: "
                "https://github.com/vllm-project/vllm/issues/8411."
            )

        audios_tokens = list[torch.Tensor]()
        audios_processed = list[torch.Tensor]()
        for audio in audios:
            assert isinstance(audio, np.ndarray)
            assert audio.ndim == 1

            if not self._audio_processor.audio_config.is_streaming:
                audio = self._audio_processor.pad(
                    audio, self.sampling_rate, is_online_streaming=False
                )

            audio_tokens = [self.begin_audio_token_id] + [
                self.audio_token_id
            ] * self.get_num_audio_tokens(len(audio))

            audios_tokens.append(torch.tensor(audio_tokens))
            audios_processed.append(torch.tensor(audio))

        return BatchFeature(
            {
                "input_ids": torch.cat(audios_tokens)[None].expand(len(text), -1),
                "audio_arrays": audios_processed,
            }
        )


class VoxtralProcessingInfo(BaseProcessingInfo):
    def get_tokenizer(self) -> MistralTokenizer:
        tokenizer = cached_tokenizer_from_config(self.ctx.model_config)
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("This model requires `--tokenizer-mode mistral`")

        return tokenizer

    def get_hf_processor(self) -> VoxtralProcessorAdapter:
        return VoxtralProcessorAdapter(self.get_tokenizer())

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=self.get_hf_processor().sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 5}  # Performance tends to degrade after 5

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"audio": self.get_max_audio_tokens()}

    def get_max_audio_tokens(self) -> int:
        return self.ctx.model_config.max_model_len

    def get_max_audio_array_len(self) -> int:
        processor = self.get_hf_processor()
        return self.get_max_audio_tokens() * int(
            processor.sampling_rate // processor.frame_rate
        )


class VoxtralDummyInputsBuilder(BaseDummyInputsBuilder[VoxtralProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)

        target_length = self.info.get_max_audio_array_len()

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=target_length, num_audios=num_audios, overrides=audio_overrides
            )
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        tokenizer = self.info.get_tokenizer()

        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        dummy_audios = dummy_mm_data.get("audio", [])

        audio_chunks: list[AudioChunk] = []
        format = "wav"
        for audio in dummy_audios:
            audio_item = Audio(
                audio_array=audio,
                sampling_rate=self.info.get_hf_processor().sampling_rate,
                format=format,
            )
            chunk = AudioChunk(input_audio=RawAudio.from_audio(audio_item))
            audio_chunks.append(chunk)

        request = ChatCompletionRequest(
            messages=[
                UserMessage(content=[TextChunk(text=dummy_text), *audio_chunks]),
            ]
        )
        res = tokenizer.mistral.encode_chat_completion(request)
        dummy_tokens = res.tokens

        dummy_mm_inputs = self.info.parse_mm_data(
            # whixtral tokenizer adds padding to the audio
            # so we need to update the audio arrays
            {**dummy_mm_data, "audio": [a.audio_array for a in res.audios]},
        )

        return ProcessorInputs(prompt=dummy_tokens, mm_items=dummy_mm_inputs)


class VoxtralMultiModalProcessor(BaseMultiModalProcessor[VoxtralProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(audio_arrays=MultiModalFieldConfig.batched("audio"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        audio_id = processor.audio_token_id

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)

            nb_audio_tokens = processor.get_num_audio_tokens(audio_len)

            return [audio_id] * nb_audio_tokens

        return [
            PromptReplacement(
                modality="audio",
                target="",  # Never match the prompt (see below note)
                replacement=get_replacement,
            ),
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: str | list[int],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        prompt_ids, mm_info, _ = super()._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        # NOTE: The tokens are already inserted by the chat template
        return prompt_ids, mm_info, True


@MULTIMODAL_REGISTRY.register_processor(
    VoxtralMultiModalProcessor,
    info=VoxtralProcessingInfo,
    dummy_inputs=VoxtralDummyInputsBuilder,
)
class VoxtralForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsTranscription
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # update quant config to so that ignored module and target module names
        # match the vLLM model names
        if hasattr(vllm_config, "quant_config"):
            vllm_config.quant_config = self.maybe_update_quant_config(
                vllm_config.quant_config
            )

        config = vllm_config.model_config.hf_config
        self.config = config
        self.downsample_factor = self.config.audio_config.downsample_factor

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        with self._mark_tower_model(vllm_config, "audio"):
            self.whisper_encoder = VoxtralEncoderModel(
                vllm_config.with_hf_config(config.audio_config),
                prefix=maybe_prefix(prefix, "whisper_encoder"),
            )
            self.audio_language_adapter = AudioLanguageAdapter(
                hidden_size=config.audio_config.d_model * self.downsample_factor,
                dim=config.text_config.hidden_size,
            )

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get module prefix for multimodal models to filter LoRA modules."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="audio_language_adapter",
            tower_model=["whisper_encoder"],
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

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def embed_multimodal(
        self, **kwargs
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None:
        audio_inputs = self._parse_and_validate_audio_arrays(**kwargs)
        if audio_inputs is None:
            return None

        audio_embeddings = self.whisper_encoder(audio_inputs)

        for i, audio_embedding in enumerate(audio_embeddings):
            seq_len, dim = audio_embedding.shape
            # Pad such that seq_len is divisible by downsample_factor
            target_seq_len = self.downsample_factor * math.ceil(
                seq_len / self.downsample_factor
            )
            audio_embedding = torch.nn.functional.pad(
                audio_embedding,
                (0, 0, 0, target_seq_len - seq_len),
            )
            audio_embeddings[i] = audio_embedding.reshape(
                target_seq_len // self.downsample_factor, dim * self.downsample_factor
            )

        # Concat, project and resplit
        audio_embeddings_packed = torch.cat(audio_embeddings, dim=0)
        audio_embeddings_packed = self.audio_language_adapter(audio_embeddings_packed)
        audio_embeddings = torch.split(
            audio_embeddings_packed, [a.shape[0] for a in audio_embeddings], dim=0
        )

        return audio_embeddings

    def _parse_and_validate_audio_arrays(
        self, **kwargs: object
    ) -> list[torch.Tensor] | None:
        audio_arrays = kwargs.pop("audio_arrays", None)
        if audio_arrays is None:
            return None

        if not isinstance(audio_arrays, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of audio_arrays. Got type: {type(audio_arrays)}"
            )

        if isinstance(audio_arrays, torch.Tensor):
            audio_arrays = list(audio_arrays.unbind(0))
        return audio_arrays

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_config = tokenizer.instruct.audio_encoder.audio_config
        max_audio_clip_s = audio_config.chunk_length_s
        sample_rate = audio_config.sampling_rate
        return SpeechToTextConfig(
            max_audio_clip_s=max_audio_clip_s,
            sample_rate=sample_rate,
            # mistral_common and whisper encoder take care of chunking
            min_energy_split_window_size=None,
        )

    @classmethod
    # for speech-to-text transcription
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio = Audio(audio, int(stt_config.sample_rate), format="wav")  # lossless
        req = TranscriptionRequest(
            model=model_config.model,
            audio=RawAudio.from_audio(audio),
            language=language,
        )

        tokenized = tokenizer.instruct.encode_transcription(req)

        return TokensPrompt(
            prompt_token_ids=tokenized.tokens,
            multi_modal_data={
                "audio": (tokenized.audios[0].audio_array, stt_config.sample_rate)
            },
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        """
        Map from audio duration to number of audio tokens produced by the ASR
        model, without running a forward pass.
        This is used for estimating the amount of processing for this audio.
        """
        tokenizer = cached_tokenizer_from_config(model_config)
        adapter = VoxtralProcessorAdapter(tokenizer)
        return adapter.get_num_audio_tokens(
            int(audio_duration_s * stt_config.sample_rate)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        remapping_rules = [
            (r"mm_streams_embeddings.embedding_module\.(.*)", r"\1"),
            (r"mm_whisper_embeddings\.(.*)", r"\1"),
            (r"audio_language_projection\.(.*)", r"audio_language_adapter.\1"),
            (
                r"audio_language_adapter\.0\.weight",
                r"audio_language_adapter.w_in.weight",
            ),
            (
                r"audio_language_adapter\.2\.weight",
                r"audio_language_adapter.w_out.weight",
            ),
        ]

        audio_params = dict(
            nn.ModuleDict(
                {
                    "audio_language_adapter": self.audio_language_adapter,
                }
            ).named_parameters()
        )
        weights = _create_fake_bias_for_k_proj(weights, ".wk.weight")

        loaded_weights = set()

        def llm_weights_generator():
            nonlocal loaded_weights
            for name, w in weights:
                is_encoder = False
                for k in [
                    "mm_whisper_embeddings",
                    "mm_streams_embeddings.embedding_module",
                ]:
                    is_encoder |= (
                        name.startswith(k)
                        and not name.startswith(f"{k}.tok_embeddings")
                        and not name.startswith(f"{k}.audio_language_projection")
                    )

                for pattern, repl in remapping_rules:
                    if re.fullmatch(pattern, name):
                        name = re.sub(pattern, repl, name)

                if is_encoder:
                    name = self.whisper_encoder.load_weight((name, w))
                    loaded_weights.add(f"whisper_encoder.{name}")
                    continue

                if name in audio_params:
                    param = audio_params[name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                    loaded_weights.add(name)
                else:
                    yield (name, w)

        for name in self.language_model.load_weights(llm_weights_generator()):
            loaded_weights.add(f"language_model.{name}")

        # potentially manually add position embeddings
        sin_key = "whisper_encoder.whisper_encoder.embed_positions.weight"
        if sin_key not in loaded_weights:
            # make sure we don't hit an error here
            loaded_weights.add(sin_key)

        return loaded_weights

    def maybe_update_quant_config(
        self, quant_config: QuantizationConfig
    ) -> QuantizationConfig:
        """
        Update quant config to so that ignored module and target module names
        match the vLLM model names.
        Right now this is specific for compressed-tensors format and
        load_format mistral.
        """
        remapping_rules = [
            (r"output", r"language_model.lm_head"),
            (
                r"layers\.(\d+)\.attention\.wo",
                r"language_model.model.layers.\1.self_attn.out_proj",
            ),
            (
                r"layers\.(\d+)\.attention\.w(.*)",
                r"language_model.model.layers.\1.self_attn.\2_proj",
            ),
            (
                r"layers\.(\d+)\.feed_forward\.w1",
                r"language_model.model.layers.\1.mlp.gate_proj",
            ),
            (
                r"layers\.(\d+)\.feed_forward\.w2",
                r"language_model.model.layers.\1.mlp.down_proj",
            ),
            (
                r"layers\.(\d+)\.feed_forward\.w3",
                r"language_model.model.layers.\1.mlp.up_proj",
            ),
            (
                r"mm_whisper_embeddings\.whisper_encoder\.transformer\.layers\.(\d+)\.attention.w(.*)",
                r"whisper_encoder.whisper_encoder.layers.\1.layers.self_attn.\2_proj",
            ),
            (
                r"mm_whisper_embeddings\.whisper_encoder\.transformer\.layers\.(\d+)\.attention.wo",
                r"whisper_encoder.whisper_encoder.layers.\1.layers.self_attn.out_proj",
            ),
            (
                r"mm_whisper_embeddings\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward.w(\d+)",
                r"whisper_encoder.whisper_encoder.layers.\1.layers.mlp.fc\2",
            ),
            (
                r"mm_whisper_embeddings\.whisper_encoder\.conv_layers\.0",
                r"whisper_encoder.whisper_encoder.conv1",
            ),
            (
                r"mm_whisper_embeddings\.whisper_encoder\.conv_layers\.1",
                r"whisper_encoder.whisper_encoder.conv2",
            ),
            (
                r"mm_whisper_embeddings\.audio_language_projection\.0",
                r"audio_language_adapter.w_in",
            ),
            (
                r"mm_whisper_embeddings\.audio_language_projection\.2",
                r"audio_language_adapter.w_out",
            ),
        ]

        # Update ignore list
        if hasattr(quant_config, "ignore"):
            mistral_ignore = []
            for name in quant_config.ignore:
                mistral_name = name
                for pattern, repl in remapping_rules:
                    if re.fullmatch(pattern, name):
                        mistral_name = re.sub(pattern, repl, name)
                mistral_ignore.append(mistral_name)
            quant_config.ignore = mistral_ignore

        # Update target list
        if hasattr(quant_config, "config_groups"):
            config_groups = quant_config.config_groups
            for group_name in config_groups:
                if "targets" in config_groups[group_name]:
                    targets = []
                    for name in config_groups[group_name]["targets"]:
                        mistral_name = name
                        for pattern, repl in remapping_rules:
                            if re.fullmatch(pattern, name):
                                mistral_name = re.sub(pattern, repl, name)
                        targets.append(mistral_name)
                config_groups[group_name]["targets"] = targets
            quant_config.config_groups = config_groups

        return quant_config


class AudioLanguageAdapter(nn.Module):
    def __init__(self, hidden_size: int, dim: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(hidden_size, dim, bias=False)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


class VoxtralEncoderModel(nn.Module):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    mistral_remapping = [
        (r"mm_streams_embeddings.embedding_module\.(.*)", r"\1"),
        (
            r"whisper_encoder\.conv_layers\.0\.(weight|bias)",
            r"whisper_encoder.conv1.\1",
        ),
        (
            r"whisper_encoder\.conv_layers\.1\.(weight|bias)",
            r"whisper_encoder.conv2.\1",
        ),
        (
            r"whisper_encoder\.conv_layers\.0\.conv\.(weight|bias)",
            r"whisper_encoder.conv1.\1",
        ),  # noqa: E501
        (
            r"whisper_encoder\.conv_layers\.1\.conv\.(weight|bias)",
            r"whisper_encoder.conv2.\1",
        ),  # noqa: E501
        (
            r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.w([qkv])\.(weight|bias)",  # noqa: E501
            r"whisper_encoder.layers.\1.self_attn.\2_proj.\3",
        ),
        (
            r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo\.(weight|bias)",  # noqa: E501
            r"whisper_encoder.layers.\1.self_attn.out_proj.\2",
        ),
        (
            r"whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm\.(weight|bias)",  # noqa: E501
            r"whisper_encoder.layers.\1.self_attn_layer_norm.\2",
        ),
        (
            r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1\.(weight|bias)",  # noqa: E501
            r"whisper_encoder.layers.\1.mlp.fc1.\2",
        ),
        (
            r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2\.(weight|bias)",  # noqa: E501
            r"whisper_encoder.layers.\1.mlp.fc2.\2",
        ),
        (
            r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3\.(weight|bias)",
            r"whisper_encoder.layers.\1.mlp.fc3.\2",
        ),  # noqa: E501
        (
            r"whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm\.(weight|bias)",
            r"whisper_encoder.layers.\1.final_layer_norm.\2",
        ),
        (
            r"whisper_encoder\.transformer\.norm\.(weight|bias)",
            r"whisper_encoder.layer_norm.\1",
        ),
    ]

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = cast(WhisperConfig, vllm_config.model_config.hf_config)
        self.dtype: torch.dtype = vllm_config.model_config.dtype
        self.is_causal = getattr(self.config, "is_causal", False)
        if self.is_causal:
            WhisperEncoderCls = WhisperCausalEncoder
        else:
            WhisperEncoderCls = partial(WhisperEncoder, init_in_fp32=True)

        self.whisper_encoder = WhisperEncoderCls(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "whisper_encoder"),
        )
        mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.config.window_size // 2,
            num_mel_bins=self.config.num_mel_bins,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=self.config.sampling_rate,
        )
        self.mel_filters = torch.tensor(mel_filters, dtype=torch.float32)

    def compute_whisper_melspec(
        self,
        audio_waveforms: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = audio_waveforms.dtype
        window = torch.hann_window(self.config.window_size).to(audio_waveforms.device)
        stft = torch.stft(
            audio_waveforms,
            self.config.window_size,
            self.config.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if global_log_mel_max := self.config.global_log_mel_max:
            if not isinstance(global_log_mel_max, float):
                raise TypeError(f"{global_log_mel_max=} needs to be of type float.")
            log_spec_max = torch.tensor(
                global_log_mel_max,
                device=log_spec.device,
                dtype=log_spec.dtype,
            )
        else:
            log_spec_max = log_spec.max()

        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.to(input_dtype)

    @property
    def downsample_factor(self) -> int:
        return (
            self.whisper_encoder.conv1.stride[0] * self.whisper_encoder.conv2.stride[0]
        )

    @property
    def chunk_size(self) -> int:
        return self.config.max_source_positions * self.downsample_factor

    def prepare_inputs_for_conv(
        self,
        audio_waveforms: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[int]]:
        assert isinstance(audio_waveforms, list)
        # list[num_mel_bins, seq_len]
        input_features = [
            self.compute_whisper_melspec(audio).to(self.dtype)
            for audio in audio_waveforms
        ]

        chunked_features: list[torch.Tensor] = []
        chunks_per_example: list[int] = []
        for feature in input_features:
            chunks = feature.split(self.chunk_size, dim=-1)
            chunked_features += chunks
            chunks_per_example.append(len(chunks))

        # [total_num_chunks, num_mel_bins, chunk_size]
        return torch.stack(chunked_features), chunks_per_example

    def forward(
        self, input_features: torch.Tensor | list[torch.Tensor]
    ) -> list[torch.Tensor]:
        if not isinstance(input_features, list):
            input_features = [input_features]

        # Split long inputs into chunks
        input_embeds, chunks_per_example = self.prepare_inputs_for_conv(input_features)

        # [total_num_chunks, ceil(chunk_size / downsample_factor), hidden_size]
        out = self.whisper_encoder([input_embeds])

        # Re-concatenate the chunks
        chunk_idx = 0
        results = []
        for n_chunks in chunks_per_example:
            result = out[chunk_idx : chunk_idx + n_chunks].flatten(0, 1)
            results.append(result)
            chunk_idx += n_chunks

        return results

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_mapping = []

        if self.is_causal:
            # For `WhisperCausalEncoder` we need
            # some more renaming
            stacked_params_mapping.extend(
                [
                    (".mlp.gate_up_proj", ".mlp.fc1", 0),
                    (".mlp.gate_up_proj", ".mlp.fc3", 1),
                ]
            )
            params_mapping.extend(
                [
                    (".mlp.down_proj", ".mlp.fc2"),
                ]
            )
        params_dict = dict(self.named_parameters())

        name, loaded_weight = weight
        for pattern, repl in self.mistral_remapping:
            if re.fullmatch(pattern, name):
                name = re.sub(pattern, repl, name)

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            for param_name, weight_name in params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        return name
