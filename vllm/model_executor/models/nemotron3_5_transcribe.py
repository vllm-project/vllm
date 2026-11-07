# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference-only NVIDIA Nemotron-3.5-Transcribe.

A NeMo speech-language model. The audio tower (log-mel, Parallel Expert
Encoder, projection; see ``nemotron3_5_transcribe_audio.py``) turns each
80 ms of audio into one embedding, and those embeddings replace the
``<|audio|>`` placeholders in the prompt of a NemotronH decoder (hybrid
Mamba, MoE and attention layers).

The checkpoint uses NeMo names: the audio tower under ``perception.*``, the
decoder under ``llm.*``, the multi-token-prediction head under ``llm.mtp.*``
and an optional CTC timestamp head under ``ctc_timestamp.*``. It is loaded
only with ``enable_ctc_timestamps=True`` and aligns completed transcripts.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import MultiModalDataDict, PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.audio_alignment import AudioAlignmentStore
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.nemotron3_5_transcribe import (
    Nemotron3_5TranscribeConfig,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsAudioAlignment,
    SupportsEagle3,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
    SupportsMultiModalEmbeddings,
    SupportsPP,
    SupportsTranscription,
)
from .module_mapping import MultiModelKeys
from .nemotron3_5_transcribe_audio import Nemotron3_5TranscribeAudioTower
from .nemotron_h import NemotronHForCausalLM
from .nemotron_h_mtp import NemotronHMTP
from .utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    init_vllm_registered_model,
    maybe_prefix,
)

if TYPE_CHECKING:
    from .nemotron3_5_transcribe_ctc import NemotronCTCHead

logger = init_logger(__name__)

AUDIO_PLACEHOLDER = "<|audio|>"
# Default longest audio used to profile memory, unless --limit-mm-per-prompt
# sets an audio length.
_DEFAULT_MAX_AUDIO_DURATION_S = 3600.0
# Prompt tokens kept free beside the longest dummy audio.
_DUMMY_TEXT_TOKEN_RESERVE = 256

# The transcription instruction of the model card's evaluation prompts. The
# model was trained with the audio after the instruction.
_TRANSCRIPTION_PROMPT = (
    "Audio transcription. Language: as spoken. "
    "Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. Output: transcript only."
)
# Training audio was at most one hour, so transcription sends pieces of at most an
# hour; vLLM splits longer audio at quiet points.
_MAX_TRANSCRIPTION_CLIP_S = 60 * 60
# Context kept free for the transcript of each second of audio: a fast speaker,
# with speaker tags.
_TRANSCRIPT_TOKENS_PER_S = 4.0

# The 40 ASR languages of the model card, plus the existing Norwegian entry.
ISO639_1_SUPPORTED_LANGS = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mr": "Marathi",
    "mt": "Maltese",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def _num_audio_embeddings(audio_config: Mapping[str, Mapping], num_samples: int) -> int:
    """Embeddings the audio tower produces for ``num_samples`` samples."""
    preprocessor = audio_config["preprocessor"]
    factor = audio_config["asr_encoder"]["subsampling_factor"]
    n_fft = preprocessor["n_fft"]
    num_frames = (num_samples + n_fft // 2 * 2 - n_fft) // preprocessor["hop_length"]
    return (num_frames + factor - 1) // factor


def _transcription_clip_s(
    audio_config: Mapping[str, Mapping], max_model_len: int
) -> int:
    """Longest piece ``/v1/audio/transcriptions`` sends: at most an hour, and short
    enough that its embeddings and transcript fit in the context."""
    sample_rate = audio_config["preprocessor"]["sample_rate"]
    hour = 3600
    tokens_per_s = (
        _num_audio_embeddings(audio_config, hour * sample_rate) / hour
        + _TRANSCRIPT_TOKENS_PER_S
    )
    fits_s = int((max_model_len - _DUMMY_TEXT_TOKEN_RESERVE) / tokens_per_s)
    return max(1, min(_MAX_TRANSCRIPTION_CLIP_S, fits_s))


def _max_audio_samples(
    audio_config: Mapping[str, Mapping], seq_len: int, override_samples: int | None
) -> int:
    """Longest audio whose embeddings fit in ``seq_len`` beside a short prompt."""
    preprocessor = audio_config["preprocessor"]
    min_samples = preprocessor["hop_length"]
    limit = int(
        override_samples or _DEFAULT_MAX_AUDIO_DURATION_S * preprocessor["sample_rate"]
    )

    def num_tokens(num_samples: int) -> int:
        return _num_audio_embeddings(audio_config, max(num_samples, min_samples))

    max_tokens = max(1, seq_len - _DUMMY_TEXT_TOKEN_RESERVE)
    if num_tokens(limit) <= max_tokens:
        return limit
    low, high = min_samples, limit
    while low < high:
        mid = (low + high + 1) // 2
        if num_tokens(mid) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return low


class Nemotron3_5TranscribeAudioInputs(TensorSchema):
    """Dimensions:
    - b: Number of audio clips
    - t: Number of samples (16 kHz)
    """

    type: Literal["audio_signal"] = "audio_signal"
    audio_signal: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("b", "t", dynamic_dims={"t"}),
    ]
    audio_signal_length: Annotated[torch.Tensor, TensorShape("b")]


class Nemotron3_5TranscribeProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Nemotron3_5TranscribeConfig:
        return self.ctx.get_hf_config(Nemotron3_5TranscribeConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.get_sampling_rate(), target_channels=1
        )

    def get_sampling_rate(self) -> int:
        return self.get_hf_config().audio_config["preprocessor"]["sample_rate"]

    def get_min_num_samples(self) -> int:
        """Shorter clips are zero-padded so that they yield one embedding."""
        return self.get_hf_config().audio_config["preprocessor"]["hop_length"]

    def get_num_audio_tokens(self, num_samples: int) -> int:
        num_samples = max(num_samples, self.get_min_num_samples())
        return _num_audio_embeddings(self.get_hf_config().audio_config, num_samples)

    def get_audio_token_id(self) -> int:
        return self.get_tokenizer().convert_tokens_to_ids(AUDIO_PLACEHOLDER)

    def get_max_audio_samples(
        self, seq_len: int, override_samples: int | None = None
    ) -> int:
        """Longest audio whose embeddings fit in ``seq_len`` beside a short prompt."""
        return _max_audio_samples(
            self.get_hf_config().audio_config, seq_len, override_samples
        )

    def get_audio_length_override(self) -> int | None:
        """The audio length set with ``--limit-mm-per-prompt``, in samples."""
        options = self.ctx.get_mm_config().limit_per_prompt.get("audio")
        return getattr(options, "length", None)

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        # The same longest clip as the dummy inputs, so the encoder cache that
        # vLLM sizes from this number admits every clip the limits allow.
        samples = self.get_max_audio_samples(seq_len, self.get_audio_length_override())
        return {"audio": self.get_num_audio_tokens(samples)}


class Nemotron3_5TranscribeDummyInputsBuilder(
    BaseDummyInputsBuilder[Nemotron3_5TranscribeProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        audio_options = mm_options.get("audio") if mm_options else None
        override = getattr(audio_options, "length", None)
        if override is None:
            override = self.info.get_audio_length_override()
        return {
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_samples(seq_len, override),
                num_audios=mm_counts.get("audio", 0),
            )
        }


class Nemotron3_5TranscribeMultiModalProcessor(
    BaseMultiModalProcessor[Nemotron3_5TranscribeProcessingInfo]
):
    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # There is no Hugging Face processor: the model computes features
        # from the waveform, so pass each clip through as float32 samples.
        min_samples = self.info.get_min_num_samples()
        signals, lengths = [], []
        for audio in cast(Iterable[Any], hf_data.get("audio", [])):
            signal = torch.as_tensor(np.asarray(audio, dtype=np.float32)).flatten()
            if signal.shape[0] < min_samples:
                signal = torch.nn.functional.pad(
                    signal, (0, min_samples - signal.shape[0])
                )
            signals.append(signal)
            lengths.append(signal.shape[0])
        return BatchFeature(
            {
                "audio_signal": signals,
                "audio_signal_length": torch.tensor(lengths, dtype=torch.long),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_signal=MultiModalFieldConfig.batched("audio"),
            audio_signal_length=MultiModalFieldConfig.batched(
                "audio", keep_on_cpu=True
            ),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        audio_token_id = self.info.get_audio_token_id()
        lengths = out_mm_kwargs.get_data().get("audio_signal_length")
        num_tokens = (
            []
            if lengths is None
            else [
                self.info.get_num_audio_tokens(int(n))
                for n in torch.as_tensor(lengths).tolist()
            ]
        )

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            return PromptUpdateDetails.select_token_id(
                [audio_token_id] * num_tokens[item_idx], embed_token_id=audio_token_id
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement,
            )
        ]


def _unpack_experts(
    name: str, tensor: torch.Tensor
) -> Iterator[tuple[str, torch.Tensor]]:
    """NeMo packs the routed experts of a MoE layer into one rank-3 tensor per
    projection; vLLM loads one transposed matrix per expert."""
    for suffix, projection in (
        (".experts.down_projs", "down_proj"),
        (".experts.gate_and_up_projs", "up_proj"),
    ):
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            for expert in range(tensor.shape[0]):
                yield (
                    f"{prefix}.experts.{expert}.{projection}.weight",
                    tensor[expert].t(),
                )
            return
    yield name, tensor


def _pad_vocab_rows(tensor: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if tensor.shape[0] >= vocab_size:
        return tensor
    return torch.cat(
        [tensor, tensor.new_zeros(vocab_size - tensor.shape[0], tensor.shape[1])], dim=0
    )


def _decoder_weights(
    weights: Iterable[tuple[str, torch.Tensor]], vocab_size: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """Shape the mapped decoder tensors for ``NemotronHForCausalLM``.

    NeMo packs the routed experts of each MoE layer into two tensors; the FP8 and
    NVFP4 checkpoints have one tensor per expert, with its quantization scales.
    The embedding and LM head are zero-padded to the decoder's padded vocabulary.
    """
    for name, tensor in weights:
        if name in (
            "model.embed_tokens.weight",
            "model.embeddings.weight",
            "lm_head.weight",
        ):
            tensor = _pad_vocab_rows(tensor, vocab_size)
        yield from _unpack_experts(name, tensor)


def _mtp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Select the MTP head's NeMo tensors and rename them for ``NemotronHMTP``.

    The head is ``llm.mtp.*``; the drafter shares the target's embedding and LM
    head.
    """
    for name, tensor in weights:
        if name.startswith("llm.mtp.") and not name.endswith("._extra_state"):
            yield from _unpack_experts(name[len("llm.") :], tensor)


@MULTIMODAL_REGISTRY.register_processor(
    Nemotron3_5TranscribeMultiModalProcessor,
    info=Nemotron3_5TranscribeProcessingInfo,
    dummy_inputs=Nemotron3_5TranscribeDummyInputsBuilder,
)
class Nemotron3_5TranscribeForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    IsHybrid,
    SupportsMambaPrefixCaching,
    SupportsEagle3,
    SupportsTranscription,
    SupportsAudioAlignment,
):
    packed_modules_mapping = NemotronHForCausalLM.packed_modules_mapping
    # The BF16 checkpoint names the decoder in NeMo's layout (``llm.*``; Mamba
    # ``A_log``/``D``/``dt_bias`` in a float32 holder; final norm ``norm``). The FP8
    # and NVFP4 checkpoints, and their quantization configs, use Hugging Face names.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".mixer._fp32_params.": ".mixer."},
        orig_to_new_prefix={
            "llm.model.norm.": "language_model.model.norm_f.",
            "llm.model.": "language_model.model.",
            "llm.lm_head.": "language_model.lm_head.",
            "backbone.": "language_model.model.",
            "lm_head": "language_model.lm_head",
            "perception.": "audio_tower.",
        },
    )
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_PLACEHOLDER
        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        audio_config = model_config.hf_config.audio_config
        return SpeechToTextConfig(
            sample_rate=audio_config["preprocessor"]["sample_rate"],
            max_audio_clip_s=_transcription_clip_s(
                audio_config, model_config.max_model_len
            ),
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        """The model card's transcription prompt, whatever the request's language.

        The prompt says "Language: as spoken" and the model detects the
        language. No published prompt names it, so the request's language is
        only validated.
        """
        if stt_params.task_type != "transcribe":
            raise ValueError(
                "Nemotron-3.5-Transcribe does not support task "
                f"{stt_params.task_type!r} yet."
            )
        if stt_params.request_prompt or stt_params.hotwords:
            logger.warning_once(
                "Nemotron-3.5-Transcribe ignores the prompt and hotwords of "
                "transcription requests."
            )
        tokenizer = cached_tokenizer_from_config(stt_params.model_config)
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{_TRANSCRIPTION_PROMPT} {AUDIO_PLACEHOLDER}",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return TokensPrompt(
            prompt_token_ids=tokenizer.encode(prompt, add_special_tokens=False),
            multi_modal_data={
                "audio": (stt_params.audio, stt_params.stt_config.sample_rate)
            },
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int:
        audio_config = model_config.hf_config.audio_config
        num_samples = max(
            int(audio_duration_s * stt_config.sample_rate),
            audio_config["preprocessor"]["hop_length"],
        )
        return _num_audio_embeddings(audio_config, num_samples)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Nemotron3_5TranscribeConfig = vllm_config.model_config.hf_config
        self.config = config
        offload = vllm_config.offload_config
        if config.enable_ctc_timestamps and (
            (
                offload.offload_backend in ("auto", "uva")
                and offload.uva.cpu_offload_gb > 0
            )
            or (
                offload.offload_backend in ("auto", "prefetch")
                and offload.prefetch.offload_group_size > 0
            )
        ):
            raise ValueError(
                "CTC timestamps currently do not support CPU weight offloading"
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Nemotron3_5TranscribeAudioTower(
                config.audio_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )

        budget = config.ctc_timestamp_max_cpu_bytes
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("ctc_timestamp_max_cpu_bytes must be a positive integer")
        self.audio_alignment_store = AudioAlignmentStore(budget)
        self.ctc_timestamp_head: NemotronCTCHead | None = None
        self._ctc_special_tokens: tuple[str, ...] = ()
        if config.enable_ctc_timestamps:
            if (
                vllm_config.parallel_config.tensor_parallel_size != 1
                or vllm_config.parallel_config.pipeline_parallel_size != 1
            ):
                raise ValueError("CTC timestamps currently require TP=1 and PP=1")
            if vllm_config.quant_config is not None:
                raise ValueError(
                    "CTC timestamps currently require an unquantized model"
                )
            from .nemotron3_5_transcribe_ctc import NemotronCTCHead as CTCHead

            tokenizer_config = config.ctc_timestamps.get("tokenizer_config", {})
            if tokenizer_config.get("legacy", False) or not tokenizer_config.get(
                "ignore_extra_whitespaces", True
            ):
                raise ValueError(
                    "Only the released non-legacy CTC tokenizer is supported"
                )
            with self._mark_tower_model(vllm_config, "audio"):
                self.ctc_timestamp_head = CTCHead(
                    config.ctc_timestamps["decoder_config"],
                    prefix=maybe_prefix(prefix, "ctc_timestamp_head"),
                )
            if not vllm_config.model_config.skip_tokenizer_init:
                self._ctc_special_tokens = tuple(
                    cached_tokenizer_from_config(
                        vllm_config.model_config
                    ).all_special_tokens
                )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_audio_input(
        self, **kwargs: object
    ) -> Nemotron3_5TranscribeAudioInputs | None:
        audio_signal = kwargs.pop("audio_signal", None)
        if audio_signal is None:
            return None
        return Nemotron3_5TranscribeAudioInputs(
            audio_signal=audio_signal,
            audio_signal_length=kwargs.pop("audio_signal_length"),
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_audio_input(**kwargs)
        if audio_input is None:
            return []
        signals = audio_input.audio_signal
        if isinstance(signals, torch.Tensor):
            signals = list(signals)
        return tuple(
            self.audio_tower.encode(signals, audio_input.audio_signal_length.flatten())
        )

    def embed_multimodal_with_alignment(
        self, mm_hashes: Sequence[str], **kwargs: object
    ) -> MultiModalEmbeddings:
        if not self.config.enable_ctc_timestamps:
            return self.embed_multimodal(**kwargs)
        audio_input = self._parse_audio_input(**kwargs)
        if audio_input is None:
            return []
        signals = list(audio_input.audio_signal)
        if len(signals) != len(mm_hashes):
            raise ValueError("Audio hashes must match the encoder batch")
        for mm_hash in mm_hashes:
            self.audio_alignment_store.mark_cached(mm_hash)
        flags = [self.audio_alignment_store.needs_capture(h) for h in mm_hashes]
        if not any(flags):
            return tuple(
                self.audio_tower.encode(
                    signals, audio_input.audio_signal_length.flatten()
                )
            )
        lengths_cpu = audio_input.audio_signal_length.flatten()
        embeddings, captures = self.audio_tower.encode(
            signals, lengths_cpu, capture_flags=flags
        )
        sample_rate = self.config.audio_config["preprocessor"]["sample_rate"]
        for index, (mm_hash, capture) in enumerate(zip(mm_hashes, captures)):
            if capture is not None:
                self.audio_alignment_store.put(
                    mm_hash, capture, float(lengths_cpu[index]) / sample_rate
                )
        return tuple(embeddings)

    def register_audio_alignment(self, handle: str, mm_hashes: Sequence[str]) -> None:
        if not self.config.enable_ctc_timestamps:
            raise ValueError("Enable CTC timestamps with enable_ctc_timestamps=True")
        self.audio_alignment_store.register(handle, mm_hashes)

    def on_audio_alignment_cache_evicted(self, mm_hash: str) -> None:
        self.audio_alignment_store.evict(mm_hash)

    def reset_audio_alignment(self) -> None:
        self.audio_alignment_store.reset()

    def release_audio_alignment(self, handles: Sequence[str]) -> None:
        self.audio_alignment_store.release(handles)

    def get_audio_alignment_stats(self) -> dict[str, Any]:
        return {
            **self.audio_alignment_store.stats(),
            "enabled": self.config.enable_ctc_timestamps,
        }

    @torch.inference_mode()
    def align_audio(self, handle: str, text: str) -> dict[str, Any]:
        if self.ctc_timestamp_head is None:
            return {"words": [], "error": "ctc_timestamps_not_enabled"}
        from .nemotron3_5_transcribe_ctc import (
            alignment_speaker_priors,
            estimate_alignment_workspace,
            public_words,
        )

        try:
            capture = self.audio_alignment_store.get(handle)
            if capture is None:
                return {
                    "words": [],
                    "error": self.audio_alignment_store.error(handle)
                    or "no_audio_alignment_capture",
                }
            tensors, duration = capture["tensors"], capture["duration_s"]
            if not 0 < duration <= 3600:
                raise ValueError(
                    "CTC timestamps require audio lasting at most one hour"
                )
            online = self.config.audio_config["online"]
            head = self.ctc_timestamp_head
            aligner = head.make_aligner(online, self.config.ctc_timestamp_language)
            text = aligner.clean_sot_transcript(text, self._ctc_special_tokens)
            if not text:
                return {"words": [], "error": None}
            ctc_lengths = head.output_lengths(tensors["asr_encoded_lengths"])
            frames = int(ctc_lengths[0])
            budget = estimate_alignment_workspace(aligner, text, frames)
            with self.audio_alignment_store.reserve_workspace(budget):

                def read_columns(
                    columns: list[list[int]], lengths: list[int]
                ) -> list[torch.Tensor]:
                    return head.read_columns(
                        tensors,
                        columns,
                        lengths,
                        core_length=online["window_frames"],
                        left_context=online["left_context_frames"],
                        right_context=online["right_context_frames"],
                    )

                speaker_probs, diarization_labels = alignment_speaker_priors(
                    text,
                    tensors["sortformer_sigmoids"],
                    tensors["diarization_labels"],
                )
                prepared = aligner._prepare(
                    (1, frames, head.num_classes_with_blank),
                    read_columns,
                    speaker_probs,
                    [text],
                    ctc_lengths=ctc_lengths,
                    sortformer_lengths=tensors["sortformer_lengths"],
                    audio_durations=[duration],
                    diarization_labels=diarization_labels,
                    diarization_lengths=tensors["diarization_lengths"],
                    diarization_frame_seconds=0.01,
                )
                return public_words(aligner.align_prepared(prepared)[0], duration)
        except (ValueError, ImportError) as error:
            return {"words": [], "error": str(error)}

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
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.language_model.compute_logits(hidden_states)
        if logits is not None:
            # Rows past the trained vocabulary (the audio placeholder and the
            # embedding headroom) are never generated.
            logits[..., self.config.output_vocab_size :] = -torch.inf
        return logits

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="audio_tower.proj",
            tower_model=["audio_tower.preprocessor", "audio_tower.encoder"],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        load_audio = isinstance(self.audio_tower, Nemotron3_5TranscribeAudioTower)
        audio_weights: list[tuple[str, torch.Tensor]] = []

        def decoder_source() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in self.hf_to_vllm_mapper.apply(weights):
                if name.endswith("._extra_state"):
                    continue
                if name.startswith("audio_tower."):
                    if load_audio:
                        # Copied so the tensor outlives a reusable read buffer.
                        audio_weights.append(
                            (name[len("audio_tower.") :], tensor.detach().clone())
                        )
                elif name.startswith("llm.mtp."):
                    continue  # loaded by the multi-token-prediction drafter
                elif name.startswith("ctc_timestamp."):
                    if self.ctc_timestamp_head is not None:
                        self.ctc_timestamp_head.load_weight(
                            name.removeprefix("ctc_timestamp."), tensor
                        )
                elif name.startswith("language_model."):
                    yield name[len("language_model.") :], tensor
                else:
                    raise ValueError(f"Unexpected checkpoint weight {name!r}.")

        decoder_weights = _decoder_weights(
            decoder_source(), self.config.text_config.vocab_size
        )
        loaded = {
            f"language_model.{name}"
            for name in self.language_model.load_weights(decoder_weights)
        }
        if self.ctc_timestamp_head is not None:
            self.ctc_timestamp_head.validate_loaded()
            loaded |= {
                f"ctc_timestamp_head.{name}" for name in self.ctc_timestamp_head._loaded
            }
        if load_audio:
            loaded |= {
                f"audio_tower.{name}"
                for name in self.audio_tower.load_weights(audio_weights)
            }
        return loaded

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        return NemotronHForCausalLM.get_mamba_state_shape_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return NemotronHForCausalLM.get_mamba_state_copy_func()


class Nemotron3_5TranscribeMTP(NemotronHMTP, SupportsMultiModalEmbeddings):
    """Multi-token-prediction draft head of Nemotron-3.5-Transcribe.

    One attention-plus-MoE block, trained to be reused for every draft token.
    It was trained on the decoder's input stream, so it receives the audio
    embeddings at the ``<|audio|>`` positions, not the placeholder embedding.
    """

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required with multimodal_embeddings.")
        return _merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            logits[..., self.config.speechlm_output_vocab_size :] = -torch.inf
        return logits

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        # Shard selection passes the raw llm.mtp.* names; load_weights passes
        # the mtp.* names that _mtp_weights produces.
        return super().is_unused_checkpoint_weight(name.removeprefix("llm."))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(_mtp_weights(weights))
