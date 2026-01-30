# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only VibeVoice-ASR model compatible with HuggingFace weights.

The `microsoft/VibeVoice-ASR` checkpoint uses `model_type="vibevoice"` without
`auto_map`, so vLLM must provide:
  - a config mapping (see `vllm/transformers_utils/configs/vibevoice.py`)
  - a model implementation that supports audio placeholders

VibeVoice-ASR represents audio by inserting embeddings into placeholder token
positions (RoPE stays standard 1D). The embeddings are produced by:
  - acoustic tokenizer + connector
  - semantic tokenizer + connector
and then summed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
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
from vllm.sequence import IntermediateTensors

_AUDIO_PLACEHOLDER = "<|object_ref_start|><|box_start|><|object_ref_end|>"
_SPEECH_START_TOKEN = "<|object_ref_start|>"
_SPEECH_END_TOKEN = "<|object_ref_end|>"
_SPEECH_PAD_TOKEN = "<|box_start|>"


class _SpeechConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


def _require_vibevoice():
    try:
        from vibevoice.modular.modular_vibevoice_tokenizer import (
            VibeVoiceAcousticTokenizerModel,
            VibeVoiceSemanticTokenizerModel,
            VibeVoiceTokenizerEncoderOutput,
            VibeVoiceTokenizerStreamingCache,
        )
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "VibeVoice-ASR support requires the external `vibevoice` package."
        ) from exc

    return (
        VibeVoiceTokenizerEncoderOutput,
        VibeVoiceTokenizerStreamingCache,
        VibeVoiceAcousticTokenizerModel,
        VibeVoiceSemanticTokenizerModel,
    )


if TYPE_CHECKING:  # pragma: no cover
    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceAcousticTokenizerModel as _VibeVoiceAcousticTokenizerModelT,
    )
    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceSemanticTokenizerModel as _VibeVoiceSemanticTokenizerModelT,
    )
    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceTokenizerEncoderOutput as _VibeVoiceTokenizerEncoderOutputT,
    )
    from vibevoice.modular.modular_vibevoice_tokenizer import (
        VibeVoiceTokenizerStreamingCache as _VibeVoiceTokenizerStreamingCacheT,
    )
else:  # pragma: no cover
    _VibeVoiceTokenizerEncoderOutputT = Any
    _VibeVoiceTokenizerStreamingCacheT = Any
    _VibeVoiceAcousticTokenizerModelT = Any
    _VibeVoiceSemanticTokenizerModelT = Any


@lru_cache(maxsize=1)
def _vibevoice_classes() -> tuple[
    type[_VibeVoiceTokenizerEncoderOutputT],
    type[_VibeVoiceTokenizerStreamingCacheT],
    type[_VibeVoiceAcousticTokenizerModelT],
    type[_VibeVoiceSemanticTokenizerModelT],
]:
    return _require_vibevoice()


class VibeVoiceASRProcessingInfo(BaseProcessingInfo):
    target_sr: int = 24_000
    speech_tok_compress_ratio: int = 3200
    # Keep startup profiling bounded; still covers ~100s audio @ 24kHz.
    max_audio_tokens_cap: int = 768

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_target_channels(self) -> int:
        return 1

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        del mm_counts
        return {"audio": min(int(seq_len), int(self.max_audio_tokens_cap))}


class VibeVoiceASRDummyInputsBuilder(
    BaseDummyInputsBuilder[VibeVoiceASRProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return _AUDIO_PLACEHOLDER * int(mm_counts.get("audio", 0))

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        max_audio_tokens = self.info.get_mm_max_tokens_per_item(seq_len, mm_counts)[
            "audio"
        ]
        # tokens ~= ceil(num_samples / compress_ratio)
        audio_len = int(max_audio_tokens) * int(self.info.speech_tok_compress_ratio)
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=int(mm_counts.get("audio", 0)),
                overrides=audio_overrides,
            )
        }


def _vibevoice_asr_field_config(hf_inputs: Mapping[str, object]):
    del hf_inputs
    return {
        "speech_tensors": MultiModalFieldConfig.batched("audio"),
    }


class VibeVoiceASRMultiModalProcessor(
    BaseMultiModalProcessor[VibeVoiceASRProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=float(self.info.target_sr),
            target_channels=self.info.get_target_channels(),
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # We do prompt updates ourselves via `_get_prompt_updates`.
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del mm_kwargs
        del tok_kwargs

        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        audios = mm_data.get("audios", [])
        speech_tensors: list[torch.Tensor] = []
        for audio in audios:
            if audio is None:
                continue
            t = audio if isinstance(audio, torch.Tensor) else torch.tensor(audio)
            # Keep waveform in fp32; tower will cast as needed.
            speech_tensors.append(t.to(dtype=torch.float32))

        return BatchFeature(
            {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                "speech_tensors": speech_tensors,
            },
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        return _vibevoice_asr_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del hf_processor_mm_kwargs
        del out_mm_kwargs

        audios = mm_items.get_items("audio", AudioProcessorItems)
        tokenizer = self.info.get_tokenizer()

        start_id = tokenizer.convert_tokens_to_ids(_SPEECH_START_TOKEN)
        end_id = tokenizer.convert_tokens_to_ids(_SPEECH_END_TOKEN)
        pad_id = tokenizer.convert_tokens_to_ids(_SPEECH_PAD_TOKEN)
        if start_id is None or end_id is None or pad_id is None:
            raise ValueError(
                "Tokenizer is missing required special tokens for VibeVoice-ASR: "
                f"{_SPEECH_START_TOKEN}, {_SPEECH_PAD_TOKEN}, {_SPEECH_END_TOKEN}."
            )

        compress_ratio = int(self.info.speech_tok_compress_ratio)

        def replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            audio_len = int(audios.get_audio_length(item_idx))
            num_tokens = int(math.ceil(audio_len / compress_ratio))
            num_tokens = max(1, num_tokens)
            token_ids = [int(start_id)] + [int(pad_id)] * num_tokens + [int(end_id)]
            return PromptUpdateDetails.select_token_id(
                token_ids, embed_token_id=int(pad_id)
            )

        return [
            PromptReplacement(
                modality="audio",
                target=_AUDIO_PLACEHOLDER,
                replacement=replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceASRMultiModalProcessor,
    info=VibeVoiceASRProcessingInfo,
    dummy_inputs=VibeVoiceASRDummyInputsBuilder,
)
class VibeVoiceForASRTraining(nn.Module, SupportsMultiModal, SupportsPP):
    """vLLM entry for HF `architectures = ["VibeVoiceForASRTraining"]`."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        (
            vibevoice_encoder_output_cls,
            vibevoice_streaming_cache_cls,
            vibevoice_acoustic_tokenizer_cls,
            vibevoice_semantic_tokenizer_cls,
        ) = _vibevoice_classes()
        self._vibevoice_encoder_output_cls = vibevoice_encoder_output_cls
        self._vibevoice_streaming_cache_cls = vibevoice_streaming_cache_cls

        with self._mark_tower_model(vllm_config, "audio"):
            self.acoustic_tokenizer = vibevoice_acoustic_tokenizer_cls(
                config.acoustic_tokenizer_config
            )
            self.semantic_tokenizer = vibevoice_semantic_tokenizer_cls(
                config.semantic_tokenizer_config
            )

            hidden_size = int(config.decoder_config.hidden_size)
            self.acoustic_connector = _SpeechConnector(
                int(config.acoustic_vae_dim), hidden_size
            )
            self.semantic_connector = _SpeechConnector(
                int(config.semantic_vae_dim), hidden_size
            )

        with self._mark_language_model(vllm_config):
            lm_prefix = "language_model" if not prefix else f"{prefix}.language_model"
            self.language_model = Qwen2Model(
                vllm_config=vllm_config,
                prefix=lm_prefix,
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        decoder_config = getattr(self.config, "decoder_config", None)
        vocab_size = None
        hidden_size = None
        if isinstance(decoder_config, Mapping):
            vocab_size = decoder_config.get("vocab_size")
            hidden_size = decoder_config.get("hidden_size")
        elif decoder_config is not None:
            vocab_size = getattr(decoder_config, "vocab_size", None)
            hidden_size = getattr(decoder_config, "hidden_size", None)

        if vocab_size is None:
            vocab_size = int(self.language_model.config.vocab_size)
        if hidden_size is None:
            hidden_size = int(self.language_model.config.hidden_size)

        self.lm_head = ParallelLMHead(
            int(vocab_size),
            int(hidden_size),
            quant_config=vllm_config.quant_config,
            prefix="lm_head",
        )
        self.logits_processor = LogitsProcessor(int(vocab_size))

        # Ensure tower runs in eval mode.
        self.acoustic_tokenizer.eval()
        self.semantic_tokenizer.eval()
        self.acoustic_connector.eval()
        self.semantic_connector.eval()

    def get_language_model(self) -> nn.Module:
        return self.language_model

    @torch.no_grad()
    def _encode_speech_single(self, speech: torch.Tensor) -> torch.Tensor:
        # speech: [samples]
        speech = speech.to(dtype=self.dtype)
        if speech.ndim != 1:
            speech = speech.flatten()

        total_samples = int(speech.shape[0])
        sample_rate = int(getattr(self, "target_sr", 24_000))
        segment_samples = int(60.0 * sample_rate)
        use_streaming = total_samples > segment_samples

        # Prepare batch dim: [1, 1, samples]
        speech_batched = speech.unsqueeze(0)

        if not use_streaming:
            acoustic_out = self.acoustic_tokenizer.encode(speech_batched.unsqueeze(1))
            audio_tokens = acoustic_out.sample(
                dist_type=self.acoustic_tokenizer.std_dist_type
            )[0]
            acoustic_features = self.acoustic_connector(audio_tokens)

            semantic_tokens = self.semantic_tokenizer.encode(
                speech_batched.unsqueeze(1)
            ).mean
            semantic_features = self.semantic_connector(semantic_tokens)
        else:
            acoustic_cache = self._vibevoice_streaming_cache_cls()
            semantic_cache = self._vibevoice_streaming_cache_cls()
            sample_indices = torch.arange(1, device=speech.device)

            acoustic_mean_segments = []
            semantic_mean_segments = []

            num_segments = int(math.ceil(total_samples / segment_samples))
            for seg_idx in range(num_segments):
                start = seg_idx * segment_samples
                end = min((seg_idx + 1) * segment_samples, total_samples)
                if end <= start:
                    continue
                chunk = speech_batched[:, start:end].contiguous()
                is_final = seg_idx == (num_segments - 1)

                acoustic_encoder_output = self.acoustic_tokenizer.encode(
                    chunk.unsqueeze(1),
                    cache=acoustic_cache,
                    sample_indices=sample_indices,
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                acoustic_mean_segments.append(acoustic_encoder_output.mean)

                semantic_encoder_output = self.semantic_tokenizer.encode(
                    chunk.unsqueeze(1),
                    cache=semantic_cache,
                    sample_indices=sample_indices,
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                semantic_mean_segments.append(semantic_encoder_output.mean)

            acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()
            acoustic_encoder_output = self._vibevoice_encoder_output_cls(
                mean=acoustic_mean_full,
                std=self.acoustic_tokenizer.fix_std,
            )
            audio_tokens = acoustic_encoder_output.sample(
                dist_type=self.acoustic_tokenizer.std_dist_type
            )[0]
            acoustic_features = self.acoustic_connector(audio_tokens)

            semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
            semantic_features = self.semantic_connector(semantic_tokens)

        combined = acoustic_features + semantic_features
        return combined.squeeze(0)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        speech_tensors = kwargs.get("speech_tensors")
        if speech_tensors is None:
            return []

        if isinstance(speech_tensors, torch.Tensor):
            # Unexpected but handle: treat as batched tensor.
            tensors = [t for t in speech_tensors]
        elif isinstance(speech_tensors, (list, tuple)):
            tensors = list(speech_tensors)
        else:
            raise TypeError(f"Unexpected speech_tensors type: {type(speech_tensors)}")

        compress_ratio = int(getattr(self.config, "speech_tok_compress_ratio", 3200))
        if compress_ratio <= 0:
            compress_ratio = 3200

        out: list[torch.Tensor] = []
        for speech in tensors:
            if speech is None:
                continue
            speech = cast(torch.Tensor, speech)
            features = self._encode_speech_single(speech)
            # Match the number of `<|box_start|>` placeholders.
            n_tokens = int(math.ceil(int(speech.shape[0]) / compress_ratio))
            n_tokens = max(1, n_tokens)
            features = features[:n_tokens]
            out.append(features.to(dtype=self.dtype))
        return tuple(out)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map HF checkpoint prefixes:
        # - model.* -> local modules (tower)
        # - model.language_model.* -> language_model.model.*
        # - lm_head.* -> lm_head.*
        mapped = []
        for name, tensor in weights:
            if name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "language_model.", 1)
            elif name.startswith("model."):
                name = name.replace("model.", "", 1)
            elif name.startswith("lm_head."):
                # keep
                pass
            mapped.append((name, tensor))

        loader = AutoWeightsLoader(self)
        return loader.load_weights(mapped)
