# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, TypedDict, Union

import os
import numpy as np
import torch
import torch.nn as nn
import transformers

from transformers import BatchFeature
from vllm.config import VllmConfig
from ...transformers_utils.configs import KimiAudioConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from .moonaudio import MoonshotKimiaModel
from ...transformers_utils.processors import KimiAudioProcessor, WhisperEncoder
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, maybe_prefix

from packaging import version

assert version.parse(transformers.__version__) >= version.parse("4.34.1")

if version.parse(transformers.__version__) >= version.parse("4.35.0"):
    from transformers.utils import is_flash_attn_2_available as is_flash_attn_available
else:
    from transformers.utils import is_flash_attn_available

if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
else:
    raise RuntimeError("flash attention must be installed")

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from .utils import maybe_prefix
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)


class KimiAudioMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class KimiAudioInputs(TypedDict):
    audio_input_ids: Optional[torch.Tensor]
    """Shape: `(num_audios, seq_len)`"""

    # text_input_ids: Optional[torch.Tensor]
    # """Shape: `(num_audios, seq_len)`"""

    is_continuous_mask: list[torch.Tensor]
    """Shape: `(num_audios, seq_len)`"""

    whisper_input_feature: list[torch.Tensor]
    """Shape: `(num_audios, seq_len, feature_dim)`"""


class KimiAudioProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> KimiAudioConfig:
        return self.ctx.get_hf_config(KimiAudioConfig)

    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        return self.ctx.get_hf_processor(KimiAudioProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_max_audio_len(self) -> int:
        processor: KimiAudioProcessor = self.get_hf_processor()
        audio_tokenizer = processor.audio_tokenizer
        sampling_rate = getattr(audio_tokenizer, "sampling_rate", 16000)
        chunk_length = getattr(audio_tokenizer, "chunk_length", 30)
        return int(sampling_rate * chunk_length)


class KimiAudioDummyInputsBuilder(BaseDummyInputsBuilder[KimiAudioProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        target_length = self.info.get_max_audio_len()
        return {
            "audio": self._get_dummy_audios(length=target_length, num_audios=num_audios)
        }


class KimiAudioMultiModalProcessor(BaseMultiModalProcessor[KimiAudioProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_nums = hf_inputs.get("audio", {}).get("nums", [])
        if audio_nums > self.info.get_supported_mm_limits():
            raise ValueError(
                f"Audio count {audio_nums} exceeds the supported limit "
                f"{self.info.get_supported_mm_limits()}"
            )
        return dict(
            whisper_input_feature=MultiModalFieldConfig.batched("audio"),
            is_continuous_mask=MultiModalFieldConfig.batched("audio"),
            text_input_ids=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        media_begin_token = tokenizer.decode([vocab.kimia_media_begin])
        media_end_token = tokenizer.decode([vocab.kimia_media_end])

        def get_replacement_kimi_audio(item_idx: int):
            whisper_features = out_mm_kwargs.get("whisper_input_feature", [])
            if item_idx < len(whisper_features):
                feature_len = whisper_features[item_idx].shape[1]
            else:
                feature_len = 30

            replacement_ids = (
                [vocab.kimia_media_begin]
                + [vocab.kimia_media_pad] * feature_len
                + [vocab.kimia_media_end]
            )

            return PromptUpdateDetails.select_token_id(
                replacement_ids,
                embed_token_id=vocab.kimia_media_pad,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=f"{media_begin_token}.*?{media_end_token}",
                replacement=get_replacement_kimi_audio,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class KimiAudioForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<|im_media_begin|><|im_media_end|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiAudioConfig = model_config.hf_config
        self.config = config
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end
        self.kimia_text_eos = config.kimia_text_eos_token_id
        self.kimia_text_blank = config.kimia_text_blank_token_id
        self.kimia_audio_eos = config.kimia_audio_eos_token_id

        mel_batch_size = getattr(config, "mel_batch_size", 20)
        encoder_path = os.path.join(model_config.model, "whisper-large-v3")
        self.audio_tower = WhisperEncoder(
            encoder_path,
            mel_batch_size=mel_batch_size,
        )
        self.multi_modal_projector = KimiAudioMultiModalProjector(self.config)
        self.language_model = MoonshotKimiaModel(
            vllm_config=config,
            prefix=maybe_prefix(prefix, "multi_modal_model"),
        )

        # text only
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, self.config.vocab_size, logit_scale
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of {name}. \
                             Got type: {type(mm_input)}"
            )

        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Optional[KimiAudioInputs]:
        audio_input_ids = kwargs.pop("audio_input_ids", None)
        # text_input_ids = kwargs.pop('text_input_ids', None)
        is_continuous_mask = kwargs.pop("is_continuous_mask", None)
        whisper_input_feature = kwargs.pop("whisper_input_feature", None)

        if whisper_input_feature is None:
            return None

        if is_continuous_mask is not None:
            is_continuous_mask = self._validate_and_reshape_mm_tensor(
                is_continuous_mask, "is_continuous_mask"
            )
        else:
            return None

        return KimiAudioInputs(
            audio_input_ids=audio_input_ids,
            # text_input_ids= text_input_ids,
            is_continuous_mask=is_continuous_mask,
            whisper_input_feature=whisper_input_feature,
        )

    def _process_audio_input(self, audio_input: KimiAudioInputs) -> torch.Tensor:
        audio_input_ids = audio_input["audio_input_ids"]
        # text_input_ids = audio_input["text_input_ids"]
        whisper_input_feature = audio_input["whisper_input_feature"]
        is_continuous_mask = audio_input["is_continuous_mask"]
        is_continuous_mask = torch.tensor([is_continuous_mask], dtype=torch.bool)
        whisper_input_feature = self.audio_tower.tokenize_waveform(
            whisper_input_feature
        )
        whisper_input_feature = whisper_input_feature.reshape(
            whisper_input_feature.shape[0],
            int(whisper_input_feature.shape[1] // 4),
            whisper_input_feature.shape[2] * 4,
        )

        # shape: batch, seq_len, hidden_size
        device = self.language_model.embed_tokens.weight.device
        audio_input_ids = audio_input_ids.to(device)
        # text_input_ids = text_input_ids.to(device)
        audio_emb = self.language_model.get_input_embeddings(audio_input_ids)
        if getattr(self.config, "use_whisper_feature"):
            assert isinstance(whisper_input_feature, list)

            media_start_idx = (audio_input_ids == self.kimia_media_begin).nonzero()
            media_end_idx = (audio_input_ids == self.kimia_media_end).nonzero()
            # shape: batch, seq_len, hidden_size
            whisper_input_dim = whisper_input_feature[0].shape[-1]
            whisper_dtype = whisper_input_feature[0].dtype
            projector_device = self.multi_modal_projector.layers[0].weight.device
            expanded_whisper = torch.zeros(
                audio_emb.shape[1],
                whisper_input_dim,
                dtype=whisper_dtype,
                device=projector_device,
            )
            for (seg_idx, start_idx), (_, end_idx) in zip(
                media_start_idx, media_end_idx
            ):
                feat_len = end_idx - (start_idx + 1)
                whisper_input_feature_i = whisper_input_feature[seg_idx].squeeze(0)
                assert feat_len == is_continuous_mask[seg_idx].sum()
                expanded_whisper[start_idx + 1 : end_idx, :] = whisper_input_feature_i[
                    :feat_len, :
                ]

            expanded_whisper = expanded_whisper.unsqueeze(0)
            whisper_emb = self.multi_modal_projector(expanded_whisper)
            whisper_emb = whisper_emb.to(device)
            is_continuous_mask = is_continuous_mask.to(device)
            whisper_emb = whisper_emb * is_continuous_mask[:, :, None]

            encoder_input_addwith_discrete_token = (
                audio_emb + whisper_emb
            ) * torch.sqrt(
                torch.tensor(2.0, dtype=whisper_emb.dtype, device=whisper_emb.device)
            )
            audio_emb = (
                audio_emb * (~is_continuous_mask[:, :, None])
                + encoder_input_addwith_discrete_token * is_continuous_mask[:, :, None]
            )
        return audio_emb

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None

        processed_features = self._process_audio_input(audio_input)
        return processed_features

    def _merge_multimodal_embeddings(
        self, inputs_embeds: torch.Tensor, audio_emb: MultiModalEmbeddings
    ) -> torch.Tensor:
        inputs_embeds += audio_emb
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            # customized merge
            inputs_embeds = self._merge_multimodal_embeddings(
                inputs_embeds,
                multimodal_embeddings,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[tuple[torch.Tensor], IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            input_ids = None

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
        sampling_metadata: SamplingMetadata,
        **kwargs: object,
    ) -> Optional[torch.Tensor]:
        # TODO(HelloWorldU): Since currently only text logits
        # are supported, we can add multimodal logits in the future.
        text_logits = self.logits_processor(
            self.lm_head, hidden_states, sampling_metadata, **kwargs
        )

        # return text_logits to vLLM; vLLM sampler will sample from text_logits
        return text_logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
