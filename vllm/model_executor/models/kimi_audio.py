# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import TypedDict, cast, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature

from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.moonaudio import MoonshotKimiaModel
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.model_executor.models.whisper import WhisperForConditionalGeneration
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
    NestedTensors,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder, ProcessorInputs
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.glm4 import Glm4Tokenizer
from vllm.tokenizers.tiktoken import TikTokenTokenizer
from vllm.transformers_utils.configs import KimiAudioConfig
from vllm.transformers_utils.processors import KimiAudioProcessor
from vllm.transformers_utils.processors.kimi_audio import WhisperEncoder

logger = init_logger(__name__)

ISO639_1_SUPPORTED_LANGS = {
    "zh": "Chinese",
    "en": "English",
}


def _get_extract_output_lengths(
    audio_input_ids: torch.Tensor, audio_placeholder_id: int
) -> list[int]:
    # Kimi Audio's audio input ids is single batch.
    # We identify continuous segments of audio tokens and 
    # then use the return to replace the placeholder in text.
    batch_size = audio_input_ids.shape[0]
    audio_output_lengths = []
    for batch_idx in range(batch_size):
        seq = audio_input_ids[batch_idx].flatten().tolist()
        count = 0
        in_audio_segment = False
        for tok in seq:
            if tok == audio_placeholder_id:
                if not in_audio_segment:
                    in_audio_segment = True
                    count = 1
                else:
                    count += 1
            else:
                if in_audio_segment:
                    audio_output_lengths.append(count)
                    count = 0
                    in_audio_segment = False

    return audio_output_lengths


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
    audio_input_ids: torch.Tensor | None
    """Shape: `(num_audios, seq_len)`"""

    is_continuous_mask: list[torch.Tensor]
    """Shape: `(num_audios, seq_len)`"""

    audio_waveforms: list[torch.Tensor]
    """List of audio waveforms as numpy arrays for 
    GLM4 tokenization and Whisper feature extraction"""


class KimiAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> KimiAudioConfig:
        return self.ctx.get_hf_config(KimiAudioConfig)

    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        model_config = self.ctx.model_config
        mm_multimodal_config = getattr(model_config, "multimodal_config", None)
        mm_processor_kwargs = getattr(
            mm_multimodal_config, "mm_processor_kwargs", {}
        )
        if isinstance(mm_processor_kwargs, dict):
            audio_tokenizer = mm_processor_kwargs.get("audio_tokenizer")
            if audio_tokenizer and isinstance(audio_tokenizer, str):
                kwargs["audio_tokenizer"] = audio_tokenizer
            else:
                raise ValueError(f"audio_tokenizer must be provided and must be"
                f"a string path.")

        return self.ctx.get_hf_processor(KimiAudioProcessor, **kwargs)

    def get_tokenizer(self) -> TikTokenTokenizer:
        return self.ctx.tokenizer

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # NOTE: override this to bypass the zero situation in profile_run().
        # This is because of the specific logic in Kimi-Audio and it got
        # empty placeholder.
        return {"audio": self.get_max_audio_tokens()}

    def get_max_audio_tokens(self) -> int:
        return self.ctx.model_config.max_model_len
    
    def get_max_audio_len(self) -> int:
        # 16000 samples * 30s
        return 480000

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)


class KimiAudioDummyInputsBuilder(BaseDummyInputsBuilder[KimiAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        target_length = self.info.get_max_audio_len()

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
        """Override to bypass the vLLM's profill stage.
        This is because of Kimi-Audio's special logic 
        when constructing empty prompt."""
        hf_processor = self.info.get_hf_processor()
        num_audios = mm_counts.get("audio", 0)

        if num_audios > 0:
            dummy_audio = [np.zeros((16000,), dtype=np.float32) for _ in range(
                num_audios)]
            dummy_mm_data = {"audio": dummy_audio}
        else:
            dummy_audio = []
            dummy_mm_data = {}
            
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data, validate=False)

        kimia_messages = []
        kimia_messages.append({"role": "user", "message_type": "text", 
                               "content": "Dummy profiling text."})
        
        for a in dummy_audio:
            kimia_messages.append({"role": "user", 
                                   "message_type": "audio", "content": a})

        # Also bypass __call__()
        prompt_data = hf_processor.get_prompt(
            kimia_messages, 
            output_type="text", 
        )
        
        text_input_ids = prompt_data["prompt_token_ids"]
        return ProcessorInputs(
            prompt=text_input_ids,
            mm_items=dummy_mm_items,
        )
    

class KimiAudioMultiModalProcessor(BaseMultiModalProcessor[KimiAudioProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_input_ids=MultiModalFieldConfig.batched("audio"),
            is_continuous_mask=MultiModalFieldConfig.batched("audio"),
            audio_waveforms=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        audio_placeholder_id = getattr(processor, "audio_placeholder_id", 152074)
        kimia_text_blank = processor.extra_tokens.kimia_text_blank

        out_mm_data = out_mm_kwargs.get_data()
        audio_input_ids = out_mm_data.get("audio_input_ids")
        audio_output_lens = _get_extract_output_lengths(audio_input_ids, audio_placeholder_id)

        def get_replacement_kimi_audio(item_idx: int):
            audio_len = 0 
            
            if audio_output_lens and item_idx < len(audio_output_lens):
                audio_len = audio_output_lens[item_idx]
            else:
                audio_len = 0

            full_seq = [kimia_text_blank] * audio_len

            return PromptUpdateDetails.select_token_id(
                seq=full_seq,
                embed_token_id=kimia_text_blank,
            )
        
        return [
            PromptReplacement(
                modality="audio",
                target=[audio_placeholder_id],
                replacement=get_replacement_kimi_audio,
            )
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: str | list[int],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        # KimiAudio's dual-stream requires text and audio to be processed
        # together in a single __call__ invocation.  The default cached path
        # splits them (text-only + mm-only), which breaks the dual-stream.
        # Always use the non-cached path that passes both to __call__.
        prompt_ids, mm_info, _ = self._apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        return prompt_ids, mm_info, False


@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class MoonshotKimiaForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsTranscription
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|AUDIO|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiAudioConfig = model_config.hf_config
        multimodal_config = model_config.multimodal_config
        
        self.config = config
        self.multimodal_config = multimodal_config
        mm_processor_kwargs = getattr(self.multimodal_config, "mm_processor_kwargs", {})
        audio_tokenizer_path = mm_processor_kwargs.get("audio_tokenizer", None)
        self._audio_tokenizer = Glm4Tokenizer(audio_tokenizer_path)
        self.kimia_text_blank = 151666
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end

        mel_batch_size = getattr(config, "mel_batch_size", 20)
        encoder_path = os.path.join(model_config.model, "whisper-large-v3")

        # NOTE: The audio tower's weight are not in the main checkpoint
        # so we need to manually add them to the loaded order later.
        with self._mark_tower_model(vllm_config, modalities=["audio"]):
            self.audio_tower = WhisperEncoder(
                encoder_path,
                mel_batch_size=mel_batch_size,
            )

        # The projector's weights ARE in the main checkpoint (as
        # model.vq_adaptor.*), so it must NOT be inside _mark_tower_model.
        # Weight renaming is handled in load_weights().
        self.multi_modal_projector = KimiAudioMultiModalProjector(self.config)

        with self._mark_language_model(vllm_config):
            self.model = MoonshotKimiaModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "multi_modal_model"),
            )

            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            )
            # NOTE: In current model, we only support text logits.
            self.mimo_output = ParallelLMHead(
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
                self.model.make_empty_intermediate_tensors
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
            mm_input = [
                torch.tensor(x) if not isinstance(x, torch.Tensor) else x
                for x in mm_input
            ]
            return torch.concat(mm_input)

    def _replace_audio_placeholders(
        self,
        audio_input_ids: torch.Tensor,
        audio_waveforms: list[np.ndarray],
    ) -> torch.Tensor:
        """
        Replace audio placeholder tokens with real discrete audio tokens.

        Args:
            audio_input_ids: Tensor contains [media_begin, placeholder, ..., media_end]
            audio_waveforms: List of audio waveform (numpy arrays) for GLM4 tokenization

        Returns:
            Tensor with placeholders replaced by real audio token IDs
        """
        if self._audio_tokenizer is None:
            raise ValueError("Audio tokenizer not initialized. Please provide "
                             "'audio_tokenizer' in mm_processor_kwargs.")

        # Get audio token offset from config
        audio_token_offset = getattr(self.config, "kimia_token_offset", 152064)
        
        if audio_input_ids.dim() == 3 and audio_input_ids.shape[0] == 1:
             audio_input_ids = audio_input_ids.squeeze(0)

        # Process concatenated sequence (batch_size=1 in concatenation mode)
        # Kimi-Audio concatenates multiple audios into [1, total_len] with
        # multiple [media_begin...media_end] segments
        batch_size = audio_input_ids.shape[0]
        result_sequences = []

        for batch_idx in range(batch_size):
            seq = audio_input_ids[batch_idx].tolist()

            # Find ALL audio segments (multiple media_begin/media_end pairs)
            audio_segments = []  # [(start_idx, end_idx), ...]
            start_idx = None

            for idx, tok in enumerate(seq):
                if tok == self.kimia_media_begin:
                    start_idx = idx
                elif tok == self.kimia_media_end and start_idx is not None:
                    audio_segments.append((start_idx, idx))
                    start_idx = None
            
            new_seq = seq.copy()
            for segment_idx in reversed(range(len(audio_segments))):
                if segment_idx < len(audio_waveforms):
                    start, end = audio_segments[segment_idx]
                    audio_waveform = audio_waveforms[segment_idx]

                    if isinstance(audio_waveform, list):
                        audio_waveform = np.array(audio_waveform)

                    audio_tokens = self._audio_tokenizer.tokenize(speech=audio_waveform)
                    audio_tokens = audio_tokens.squeeze(0).cpu().tolist()
                    audio_tokens = [tok + audio_token_offset for tok in audio_tokens]
                    
                    # Replace: keep media_begin, replace placeholders, keep media_end
                    new_seq = new_seq[: start + 1] + audio_tokens + new_seq[end:]

            result_sequences.append(new_seq)
        
        result = torch.tensor(
            result_sequences, dtype=torch.long, device=audio_input_ids.device
        )
        
        return result

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> KimiAudioInputs | None:
        audio_input_ids = kwargs.pop("audio_input_ids", None)
        is_continuous_mask = kwargs.pop("is_continuous_mask", None)

        audio_waveforms = kwargs.pop("audio_waveforms", None)
        if audio_waveforms is None:
            audio_waveforms = kwargs.pop("audio", None)

        if audio_waveforms is None:
            return None
        
        if is_continuous_mask is not None:
            if isinstance(is_continuous_mask, torch.Tensor):
                if is_continuous_mask.dim() == 3 and is_continuous_mask.shape[0] == 1:
                    is_continuous_mask = is_continuous_mask.squeeze(0)

            is_continuous_mask = self._validate_and_reshape_mm_tensor(
                is_continuous_mask, "is_continuous_mask"
            )
        else:
            return None

        if audio_input_ids is not None:
            audio_input_ids = self._replace_audio_placeholders(
                audio_input_ids,
                audio_waveforms,
            )

        return KimiAudioInputs(
            audio_input_ids=audio_input_ids,
            is_continuous_mask=is_continuous_mask,
            audio_waveforms=audio_waveforms,
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def _extract_discrete_tokens_per_segment(
        self,
        audio_input_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Extract discrete audio token IDs for each media_begin/media_end
        segment from the (already replaced) audio_input_ids.

        Returns a list of 1-D tensors, one per audio segment, containing
        the discrete token IDs between media_begin+1 and media_end.
        """
        if audio_input_ids.dim() == 2:
            seq = audio_input_ids[0]
        else:
            seq = audio_input_ids

        segments: list[torch.Tensor] = []
        seq_list = seq.tolist()
        start_idx = None
        for idx, tok in enumerate(seq_list):
            if tok == self.kimia_media_begin:
                start_idx = idx
            elif tok == self.kimia_media_end and start_idx is not None:
                # Tokens between media_begin+1 and media_end (exclusive)
                segment_tokens = seq[start_idx + 1 : idx]
                segments.append(segment_tokens)
                start_idx = None

        return segments

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        """
        Produce the FULL-LENGTH audio stream embedding, faithful to the
        official KimiAudio dual-stream formula.

        Official formula (applied at ALL positions):
            audio_emb = embed_tokens(audio_input_ids)           # full seq
            audio_emb[continuous] = (audio_emb[continuous]
                                     + whisper_proj) * sqrt(2)  # fuse
            inputs_embeds = audio_emb + text_emb                # add streams

        We return ``audio_emb`` (same length as the text stream) so that
        ``_merge_audio_embeddings`` can simply add it to ``text_emb``.
        """
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None

        audio_waveforms = audio_input["audio_waveforms"]
        audio_input_ids = audio_input["audio_input_ids"]
        is_continuous_mask = audio_input["is_continuous_mask"]

        # --- 1. Embed the FULL audio stream ---
        embed_device = self.model.embed_tokens.weight.device
        if audio_input_ids is None:
            return None

        # Flatten to 1-D if needed
        if audio_input_ids.dim() == 2:
            audio_ids_flat = audio_input_ids[0]
        else:
            audio_ids_flat = audio_input_ids
        audio_ids_flat = audio_ids_flat.to(embed_device)

        # Full audio stream embedding: embed(audio_input_ids)
        audio_emb = self.model.embed_tokens(audio_ids_flat)  # (seq_len, hidden)

        # --- 2. Whisper fusion at continuous positions ---
        if is_continuous_mask.dim() == 2:
            cont_mask = is_continuous_mask[0].bool()
        else:
            cont_mask = is_continuous_mask.bool()
        cont_mask = cont_mask.to(embed_device)

        # Compute whisper features for each audio waveform
        tower_device = next(self.audio_tower.parameters()).device
        projector_device = self.multi_modal_projector.layers[0].weight.device
        projector_dtype = self.multi_modal_projector.layers[0].weight.dtype

        whisper_projs: list[torch.Tensor] = []
        for audio_data in audio_waveforms:
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            elif isinstance(audio_data, list):
                audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            else:
                audio_tensor = (
                    audio_data.unsqueeze(0) if audio_data.dim() == 1 else audio_data
                )
            audio_tensor = audio_tensor.to(device=tower_device, dtype=torch.float32)
            feature = self.audio_tower.tokenize_waveform(audio_tensor)
            feature = feature.reshape(
                feature.shape[0],
                int(feature.shape[1] // 4),
                feature.shape[2] * 4,
            ).squeeze(0)
            feature = feature.to(device=projector_device, dtype=projector_dtype)
            proj = self.multi_modal_projector(feature)
            whisper_projs.append(proj)

        if whisper_projs:
            whisper_all = torch.cat(whisper_projs, dim=0)  # (total_cont, hidden)
            cont_count = cont_mask.sum().item()

            # Apply fusion: audio_emb[cont] = (audio_emb[cont] + whisper) * sqrt(2)
            scale = torch.sqrt(torch.tensor(
                2.0, dtype=audio_emb.dtype, device=audio_emb.device
            ))
            whisper_all = whisper_all.to(dtype=audio_emb.dtype, device=audio_emb.device)
            min_len = min(cont_count, whisper_all.shape[0])
            cont_indices = torch.where(cont_mask)[0][:min_len]
            audio_emb[cont_indices] = (
                audio_emb[cont_indices] + whisper_all[:min_len]
            ) * scale

        # Cache the FULL-LENGTH audio embedding for embed_input_ids.
        # The framework's gather_mm_embeddings slices to PlaceholderRange.length
        # (130 tokens), losing 13 structural tokens.  We bypass that by using
        # the cached full embedding directly in embed_input_ids.
        self._pending_full_audio_emb = audio_emb.clone()

        # Return for framework (will be sliced, but we use the cache instead).
        return [audio_emb]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:

        # Embed text stream
        inputs_embeds = self.model.embed_input_ids(input_ids)

        # Try to use the cached full-length audio embedding (set by
        # embed_multimodal in the same execute_model step).
        full_audio = getattr(self, "_pending_full_audio_emb", None)
        if full_audio is not None:
            self._pending_full_audio_emb = None  # consume
            if full_audio.shape[0] == inputs_embeds.shape[0]:
                # Official formula: inputs_embeds = audio_emb + text_emb
                inputs_embeds = inputs_embeds + full_audio.to(inputs_embeds.dtype)
                
                return inputs_embeds

        if input_ids.shape[-1] == 1 and (is_multimodal is None or not is_multimodal.any()):
            dummy_audio_token_id = self.kimia_text_blank 
            dummy_audio_ids = torch.full_like(input_ids, dummy_audio_token_id)
            
            dummy_audio_emb = self.model.embed_tokens(dummy_audio_ids)
            inputs_embeds = inputs_embeds + dummy_audio_emb.to(inputs_embeds.dtype)
            
            return inputs_embeds
        
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor] | IntermediateTensors:
        hidden_states = self.model(
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
        # NOTE: Since currently only text logits
        # are supported, we can add multimodal logits in the future.
        text_logits = self.logits_processor(self.lm_head, hidden_states)
        return text_logits

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
        language: str,
        task_type: str,
        request_prompt: str,
    ) -> PromptType:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio = Audio(audio, int(stt_config.sample_rate), format="wav")  # lossless
        req = TranscriptionRequest(
            model=model_config.model,
            audio=RawAudio.from_audio(audio),
            language=language,
        )

        tokenized = tokenizer.instruct.encode_transcription(req)
        audio = (tokenized.audios[0].audio_array, stt_config.sample_rate)
        prompts_dict = {"multi_modal_data": {"audio": audio}}
        prompts_dict["prompt_token_ids"] = tokenized.tokens
        return cast(PromptType, prompts_dict)

    @classmethod
    def validate_language(cls, language: str) -> bool:
        # same as whisper
        return WhisperForConditionalGeneration.validate_language(language)

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
        adapter = KimiAudioProcessor(tokenizer)
        return adapter.get_num_audio_tokens(
            int(audio_duration_s * stt_config.sample_rate)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint stores the whisper-to-LLM projector as
        # "model.vq_adaptor.*" (inside MoonshotKimiaModel), but our vLLM
        # architecture places it as "multi_modal_projector.*" (top-level).
        # Rename on-the-fly so AutoWeightsLoader can match them.
        def _remap(weights_iter):
            for name, tensor in weights_iter:
                if name.startswith("model.vq_adaptor."):
                    name = name.replace("model.vq_adaptor.",
                                        "multi_modal_projector.", 1)
                yield name, tensor

        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(_remap(weights))

        # audio_tower is initialized in __init__ via WhisperModel.from_pretrained
        # and its weights are NOT in the main checkpoint.
        for name, _ in self.audio_tower.named_parameters():
            loaded_weights.add(f"audio_tower.{name}")

        return loaded_weights
