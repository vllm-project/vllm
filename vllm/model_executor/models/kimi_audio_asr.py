# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kimi-Audio (MoonshotKimiaForCausalLM) with vLLM-native Transcriptions API support.

Goal:
- Enable vLLM OpenAI-compatible endpoint: POST /v1/audio/transcriptions
- Use vLLM engine for attention + decoding
- Use kimia_infer prompt manager to build multimodal tensors
  (whisper features + token streams)
- Do *whisper feature -> hidden adaptor* and *embedding-time mixing* inside the
  vLLM model forward, using the model's own parameters.

Notes
-----
This is a native integration:
- Audio preprocessing relies on kimia_infer prompt manager at request time
- No HuggingFace ProcessorMixin dependency
- Full integration with vLLM's multimodal pipeline
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
from collections.abc import Iterable, Mapping
from contextlib import suppress
from typing import Any, ClassVar, Literal

import numpy as np
import regex as re
import torch
from scipy.io import wavfile
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
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
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing.processor import BaseMultiModalProcessor

logger = init_logger(__name__)

__all__ = ["KimiAudioForConditionalGeneration"]


# ---- helpers / caching ----

_KIMIA_PROMPT_MANAGER_LOCK = threading.Lock()
_KIMIA_PROMPT_MANAGER = None
_KIMIA_PROMPT_MANAGER_KEY: tuple[str, int, int] | None = None


def _write_wav_tmp(audio: np.ndarray, sample_rate: int) -> str:
    """Write float32 waveform to a temporary wav file."""
    x = np.clip(audio, -1.0, 1.0).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    wavfile.write(tmp_name, sample_rate, x)
    return tmp_name


def _get_kimia_prompt_manager(
    *,
    model_path: str,
    kimia_token_offset: int,
    kimia_text_audiodelaytokens: int,
):
    """Create or reuse a cached KimiAPromptManager.

    KimiAPromptManager loads Whisper + audio tokenizer + text tokenizer and is
    expensive. Re-initializing it per request is slow and can cause GPU memory
    churn. We cache one instance per process.
    """

    global _KIMIA_PROMPT_MANAGER
    global _KIMIA_PROMPT_MANAGER_KEY

    key = (model_path, int(kimia_token_offset), int(kimia_text_audiodelaytokens))

    with _KIMIA_PROMPT_MANAGER_LOCK:
        if _KIMIA_PROMPT_MANAGER is not None and key == _KIMIA_PROMPT_MANAGER_KEY:
            return _KIMIA_PROMPT_MANAGER

        from kimia_infer.api.prompt_manager import KimiAPromptManager

        _KIMIA_PROMPT_MANAGER = KimiAPromptManager(
            model_path=model_path,
            kimia_token_offset=key[1],
            kimia_text_audiodelaytokens=key[2],
        )
        _KIMIA_PROMPT_MANAGER_KEY = key
        return _KIMIA_PROMPT_MANAGER


# ---- Multimodal processor plumbing ----


def _kimia_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    # All tensors are batched (= one audio item per request). Shape expectations:
    # - whisper_input_features: [B, S, F]
    # - is_continuous_mask:     [B, S]
    # - text_input_ids:         [B, S]
    # - audio_input_ids:        [B, S] (original Kimi-Audio ids incl. audio-vocab)

    return dict(
        whisper_input_features=MultiModalFieldConfig.batched("audio"),
        is_continuous_mask=MultiModalFieldConfig.batched("audio"),
        text_input_ids=MultiModalFieldConfig.batched("audio"),
        audio_input_ids=MultiModalFieldConfig.batched("audio"),
    )


def _flatten_seq_inputs(value: object) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() <= 1:
            return value
        return value.reshape(-1)
    if isinstance(value, (list, tuple)):
        elems = [elem for elem in value if isinstance(elem, torch.Tensor)]
        if not elems:
            return None
        if len(elems) == 1:
            return elems[0]
        return torch.cat([elem.reshape(-1) for elem in elems], dim=0)
    return None


def _flatten_feature_inputs(value: object) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() <= 2:
            return value
        return value.reshape(-1, value.shape[-1])
    if isinstance(value, (list, tuple)):
        elems = [elem for elem in value if isinstance(elem, torch.Tensor)]
        if not elems:
            return None
        if len(elems) == 1:
            return elems[0]
        return torch.cat(elems, dim=0)
    return None


class KimiAudioASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        # We only need HF config values (token ids) and let vLLM handle weights.

        return self.ctx.model_config.hf_config

    def get_data_parser(self) -> MultiModalDataParser:
        # Kimi-Audio uses an `audio` modality payload that is *not* a waveform
        # AudioItem, but a dict-of-tensors produced by the processor.
        #
        # Override the parser so vLLM doesn't route this through the core audio
        # parser (which expects waveform AudioItems).
        return KimiAudioASRMultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # One audio clip per request for now.
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # Avoid slow dummy multimodal processing at startup. Kimi-Audio maps
        # audio into the *same* token sequence (prompt_token_ids) and provides
        # additional tensors aligned to that sequence, so we cap per-audio token
        # budget to the model max.
        return {"audio": min(int(seq_len), int(self.ctx.model_config.max_model_len))}


class KimiAudioASRDummyInputsBuilder(
    BaseDummyInputsBuilder[KimiAudioASRProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # Dummy text is tokenized by our processor override to a single
        # placeholder token id.
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        # IMPORTANT: Always return dict-of-tensors, never a file path.
        # Returning a string would be parsed by vLLM's core audio parser
        # (waveform AudioItem) and can crash during V1 dummy runs.
        bsz = 1
        s = 1

        config = self.info.get_hf_config()
        whisper_feat_dim = int(
            getattr(
                config,
                "kimia_adaptor_input_dim",
                KimiAudioForConditionalGeneration.DEFAULT_KIMIA_ADAPTOR_INPUT_DIM,
            )
        )

        return {
            "audio": {
                "whisper_input_features": torch.zeros(
                    (bsz, s, whisper_feat_dim), dtype=torch.float16
                ),
                "is_continuous_mask": torch.zeros((bsz, s), dtype=torch.bool),
                "text_input_ids": torch.zeros((bsz, s), dtype=torch.long),
                "audio_input_ids": torch.zeros((bsz, s), dtype=torch.long),
            }
        }


class KimiAudioASRMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | Any,
    ) -> ModalityDataItems[Any, Any] | None:
        # Kimi-Audio expects dict-of-tensors under the `audio` modality.
        # Do not accept string paths here; vLLM's core audio parser expects
        # waveform AudioItems and will crash on strings during V1 dummy runs.

        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={
                    "whisper_input_features",
                    "is_continuous_mask",
                    "text_input_ids",
                    "audio_input_ids",
                },
                fields_factory=_kimia_field_config,
            )

        return None


class KimiAudioASRMultiModalProcessor(
    BaseMultiModalProcessor[KimiAudioASRProcessingInfo]
):
    """Minimal processor for Kimi-Audio ASR.

    Key point: Kimi-Audio does not ship a HuggingFace `ProcessorMixin`.
    Its tokenizer is a custom TikTokenTokenizer, so we must bypass
    vLLM's default HF-processor path.

    We therefore:
    - tokenize `prompt` ourselves
    - pass through our precomputed dict-of-tensors (whisper_input_features,
      is_continuous_mask, text_input_ids)
    """

    PLACEHOLDER_TOKEN_ID = 151666

    # NOTE: Do not override `_get_data_parser` / `build_data_parser`.
    # vLLM routes data parsing through `ProcessingInfo.get_data_parser()`.

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # We bypass HF processors entirely, and rely on vLLM's prompt-update
        # mechanism (PromptReplacement) to expand a single placeholder token
        # into the full placeholder sequence.
        return False

    def _call_hf_processor(
        self,
        prompt: str | list[int],
        mm_data: dict[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Tokenize prompt without using HF ProcessorMixin.
        # For Kimi-Audio we intentionally collapse the prompt into a single
        # placeholder token. The PromptReplacement produced by
        # `_get_prompt_updates` will expand it to the correct sequence length.
        #
        # This also makes dummy/profiling paths deterministic.
        prompt_ids = [self.PLACEHOLDER_TOKEN_ID]

        # Pass-through multimodal dict tensors. The keys here are expected to be
        # a flattened dict produced by BaseMultiModalProcessor.
        out: dict[str, object] = {"input_ids": [prompt_ids]}
        out.update(mm_data)
        return BatchFeature(out, tensor_type="pt")

    def _apply_hf_processor_text_only(
        self,
        prompt_text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
        # For dummy/profiling paths vLLM may pass a string prompt.
        # We want a single placeholder token id so our PromptReplacement
        # can reliably match and expand it.
        return [self.PLACEHOLDER_TOKEN_ID]

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, torch.Tensor],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _kimia_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        # vLLM requires one PromptUpdate per audio item to establish placeholder
        # ranges. For Kimi-Audio we use a token-id placeholder sequence of the
        # *same length as the prompt*, so the placeholder range covers the full
        # prompt. The model's forward() uses `audio_input_ids` + masks to apply
        # audio features at the right positions.
        audio_items = mm_items.get_items("audio", DictEmbeddingItems)

        placeholder_id = self.PLACEHOLDER_TOKEN_ID

        def _placeholder_seq(item_idx: int) -> list[int]:
            d = audio_items.get(item_idx)

            audio_ids = d["audio_input_ids"]
            if isinstance(audio_ids, torch.Tensor):
                if audio_ids.dim() == 2:
                    s = int(audio_ids.shape[1])
                elif audio_ids.dim() == 1:
                    s = int(audio_ids.shape[0])
                else:
                    s = 1
            else:
                s = 1
            return [placeholder_id] * max(s, 1)

        # Expand the single placeholder token to cover the full audio sequence
        # length, so that vLLM's placeholder-range bookkeeping matches the
        # shapes of our tensors (audio_input_ids / masks / features).

        seq = _placeholder_seq(0)

        return [
            PromptReplacement(
                modality="audio",
                target=[placeholder_id],
                replacement=seq,
            )
        ]


# ---- VQ Adaptor ----


class VQAdaptor(torch.nn.Module):
    """Kimi-Audio VQ Adaptor for whisper feature -> hidden dim.

    Matches the architecture in Kimi-Audio's modeling_kimia.py
    """

    def __init__(self, input_dim: int, hidden_size: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
            torch.nn.LayerNorm(hidden_size, eps=rms_norm_eps, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


# ---- Model ----


@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioASRMultiModalProcessor,
    info=KimiAudioASRProcessingInfo,
    dummy_inputs=KimiAudioASRDummyInputsBuilder,
)
class KimiAudioForConditionalGeneration(
    Qwen2ForCausalLM, SupportsTranscription, SupportsMultiModal
):
    """Kimi-Audio model for conditional generation + transcription."""

    # Default config values (from HF generation_config.json)
    DEFAULT_KIMIA_TOKEN_OFFSET: ClassVar[int] = 152064
    DEFAULT_KIMIA_TEXT_AUDIODELAYTOKENS: ClassVar[int] = 0
    DEFAULT_KIMIA_ADAPTOR_INPUT_DIM: ClassVar[int] = 5120
    PLACEHOLDER_TOKEN_ID: ClassVar[int] = 151666

    skip_warmup_audio_preprocessing: ClassVar[bool] = True

    # NOTE: Following Whisper's pattern - do NOT set supports_multimodal_raw_input_only.
    # The multimodal processor handles audio encoding and returns pre-computed
    # embeddings. This avoids batch packing issues that cause concurrent request
    # corruption.

    def __init__(self, *, vllm_config, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

        # NOTE: Do NOT register external audio tower submodules.
        # External components may carry parameters not present in the HF
        # checkpoint; registering them would cause V1 multiprocessing strict
        # weight loading to fail ("Following weights were not initialized from
        # checkpoint").

        # Manually add vq_adaptor if not present (vLLM may not load it)

        config = vllm_config.model_config.hf_config

        if (
            hasattr(config, "use_whisper_feature")
            and config.use_whisper_feature
            and not hasattr(self.model, "vq_adaptor")
        ):
            # Manually add vq_adaptor if not present (vLLM may not load it)
            input_dim = getattr(
                config,
                "kimia_adaptor_input_dim",
                KimiAudioForConditionalGeneration.DEFAULT_KIMIA_ADAPTOR_INPUT_DIM,
            )
            hidden_size = config.hidden_size
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

            self.model.vq_adaptor = VQAdaptor(input_dim, hidden_size, rms_norm_eps)

            logger.warning(
                "[Kimi-Audio] Manually initialized vq_adaptor (%d -> %d)",
                input_dim,
                hidden_size,
            )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Apply token embeddings, merging multimodal embeddings if provided.

        For Kimi-Audio, embed_multimodal() already produces fully mixed embeddings
        (token + audio + text), so we return them directly when available.
        """
        # If we have pre-computed multimodal embeddings from embed_multimodal(),
        # return them directly (they already contain the mixed audio features)
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            # embeddings is a tuple of tensors, one per audio item
            # For single audio item, just return the first (and only) embedding
            if isinstance(multimodal_embeddings, (list, tuple)):
                return multimodal_embeddings[0]
            return multimodal_embeddings

        # Fallback: return base token embeddings
        return self.model.embed_tokens(input_ids)

    # Transcriptions API support

    supported_languages: ClassVar[Mapping[str, str]] = ISO639_1_SUPPORTED_LANGS

    supports_transcription: ClassVar[Literal[True]] = True

    def embed_multimodal(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        """Process audio features and return mixed embeddings.

        Following Whisper's pattern, this method is called by vLLM's multimodal
        pipeline to get pre-computed embeddings. The embeddings are then passed
        to embed_input_ids() which merges them with text embeddings.
        """
        # Extract multimodal tensors from kwargs
        whisper_input_features = kwargs.get("whisper_input_features")
        is_continuous_mask = kwargs.get("is_continuous_mask")
        text_input_ids = kwargs.get("text_input_ids")
        audio_input_ids = kwargs.get("audio_input_ids")

        if not isinstance(whisper_input_features, torch.Tensor):
            return ()

        # Flatten inputs (handle batched inputs)
        flat_whisper = _flatten_feature_inputs(whisper_input_features)
        flat_mask = _flatten_seq_inputs(is_continuous_mask)
        flat_text_ids = _flatten_seq_inputs(text_input_ids)
        flat_audio_ids = _flatten_seq_inputs(audio_input_ids)

        # Determine the true input IDs to use (audio stream preferred)
        if isinstance(flat_audio_ids, torch.Tensor):
            true_input_ids = flat_audio_ids
        elif isinstance(flat_text_ids, torch.Tensor):
            true_input_ids = flat_text_ids
        else:
            # Fallback: create dummy input IDs
            true_input_ids = torch.zeros(
                flat_whisper.shape[0],
                dtype=torch.long,
                device=flat_whisper.device,
            )

        # Get base token embeddings
        emb = self.model.embed_tokens(true_input_ids)
        device = emb.device
        dtype = emb.dtype

        # Process mask
        mask = None
        if isinstance(flat_mask, torch.Tensor):
            mask = flat_mask.to(device)
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            if mask.dim() != 1:
                mask = mask.reshape(-1)

        # Process whisper features through VQ-adaptor and mix
        if isinstance(flat_whisper, torch.Tensor):
            whisper_feats = flat_whisper.to(device=device, dtype=dtype)

            # Handle shape mismatch
            if whisper_feats.shape[0] != emb.shape[0]:
                if mask is not None and mask.shape[0] == emb.shape[0]:
                    expanded = emb.new_zeros((emb.shape[0], whisper_feats.shape[-1]))
                    try:
                        expanded[mask] = whisper_feats
                    except RuntimeError:
                        logger.warning(
                            "[Kimi-Audio] whisper/mask length mismatch: "
                            "features=%d mask_len=%d; skipping conditioning.",
                            whisper_feats.shape[0],
                            mask.shape[0],
                        )
                        whisper_feats = None
                    else:
                        whisper_feats = expanded
                else:
                    logger.warning(
                        "[Kimi-Audio] whisper_input_features length mismatch: "
                        "expected %d tokens but got %d features; skipping.",
                        emb.shape[0],
                        whisper_feats.shape[0],
                    )
                    whisper_feats = None

            if (
                isinstance(whisper_feats, torch.Tensor)
                and whisper_feats.shape[0] == emb.shape[0]
            ):
                if whisper_feats.shape[-1] == emb.shape[-1]:
                    whisper_emb = whisper_feats
                else:
                    # VQ-adaptor expects [S, B, F]
                    whisper_sbF = (
                        whisper_feats.unsqueeze(1)
                        if whisper_feats.dim() == 2
                        else whisper_feats
                    )
                    whisper_emb = self.model.vq_adaptor(whisper_sbF).squeeze(1)

                if mask is not None:
                    mask_f = mask[:, None]
                    whisper_emb = whisper_emb * mask_f

                    # Kimi-Audio mixing formula: (emb + whisper) * sqrt(2)
                    sqrt2 = math.sqrt(2.0)
                    encoder_add = (emb + whisper_emb) * sqrt2
                    emb = emb * (~mask_f) + encoder_add * mask_f

        # Add text embeddings if present
        if isinstance(flat_text_ids, torch.Tensor):
            text_ids = flat_text_ids.to(device)
            text_emb = self.model.embed_tokens(text_ids)
            has_text = (text_ids != 0).any()
            emb = emb + text_emb * has_text.to(dtype=emb.dtype)

        # Return as tuple (one item per audio input)
        # Split by batch if needed - for now assume single audio item
        return (emb,)

    # Text-only logits masking (avoid audio token generation)

    def _mask_audio_logits_(self, logits: torch.Tensor) -> torch.Tensor:
        cutoff = getattr(self.config, "kimia_token_offset", None)

        if cutoff is None:
            cutoff = getattr(self.config, "kimia_text_output_vocab", None)

        if cutoff is None:
            return logits

        cutoff = int(cutoff)

        if cutoff <= 0 or cutoff >= logits.shape[-1]:
            return logits

        logits[..., cutoff:] = -1e30

        return logits

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: Literal["transcribe", "translate"]
    ) -> SpeechToTextConfig:
        # Kimi-Audio prompt manager uses whisper-large-v3 style features; 16kHz.

        # We allow longer clips at server layer via chunking if enabled.

        # Use a finite limit to satisfy server-side duration checks.

        # Long-audio chunking can be implemented later.

        return SpeechToTextConfig(
            sample_rate=16_000,
            max_audio_clip_s=30,
            default_sampling_params={
                "temperature": 0.0,
                "top_k": 5,
                "top_p": 1.0,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
            skip_reading_prefix_cache=True,
        )

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
        """Construct prompt_token_ids + extra tensors for forward mixing.

        Using native processing approach.

        """

        # Create a temporary WAV file for the audio data.
        wav_path = _write_wav_tmp(audio, int(stt_config.sample_rate))

        try:
            # Build the Kimi-Audio prompt exactly like the reference
            # implementation (KimiAPromptManager). This ensures the returned
            # multimodal tensors (audio/text token streams + whisper features)
            # match training-time expectations.
            try:
                import kimia_infer.api.prompt_manager  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Kimi-Audio ASR requires `kimia_infer` to be installed. "
                    "Please install the dependency before serving this model."
                ) from exc

            hf_cfg = model_config.hf_config
            kimia_token_offset = int(
                getattr(
                    hf_cfg,
                    "kimia_token_offset",
                    KimiAudioForConditionalGeneration.DEFAULT_KIMIA_TOKEN_OFFSET,
                )
            )
            kimia_text_audiodelaytokens = int(
                getattr(
                    hf_cfg,
                    "kimia_text_audiodelaytokens",
                    KimiAudioForConditionalGeneration.DEFAULT_KIMIA_TEXT_AUDIODELAYTOKENS,
                )
            )

            prompt_manager = _get_kimia_prompt_manager(
                model_path=str(model_config.model),
                kimia_token_offset=kimia_token_offset,
                kimia_text_audiodelaytokens=kimia_text_audiodelaytokens,
            )

            messages = []
            if request_prompt.strip():
                messages.append(
                    {
                        "role": "user",
                        "message_type": "text",
                        "content": request_prompt,
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": wav_path,
                }
            )

            # Build multimodal tensors without grad; vLLM may hash tensors.
            with torch.inference_mode():
                content = prompt_manager.get_prompt(messages, output_type="text")
                (
                    audio_ids,
                    text_ids,
                    is_continuous_mask,
                    _audio_loss_mask,
                    _text_loss_mask,
                ) = content.to_tensor()

                if not content.continuous_feature:
                    raise RuntimeError("No whisper features produced by prompt manager")

                whisper_feats = content.continuous_feature[0]
                if isinstance(whisper_feats, torch.Tensor) and whisper_feats.dim() == 2:
                    whisper_feats = whisper_feats.unsqueeze(0)

                if (
                    isinstance(whisper_feats, torch.Tensor)
                    and isinstance(is_continuous_mask, torch.Tensor)
                    and whisper_feats.dim() == 3
                    and is_continuous_mask.dim() == 2
                    and whisper_feats.shape[0] == is_continuous_mask.shape[0]
                    and whisper_feats.shape[1] != is_continuous_mask.shape[1]
                ):
                    # Some Kimi-Audio preprocessing paths return whisper features only
                    # for masked (continuous) positions. Expand to full token length so
                    # the model forward path can avoid data-dependent scattering.
                    if whisper_feats.shape[0] != 1:
                        logger.warning(
                            "[Kimi-Audio] Unexpected batch size for "
                            "whisper features: %d",
                            whisper_feats.shape[0],
                        )
                    else:
                        mask = is_continuous_mask[0].to(torch.bool)
                        idx = mask.nonzero(as_tuple=False).squeeze(-1)
                        if idx.numel() == whisper_feats.shape[1]:
                            full = whisper_feats.new_zeros(
                                (1, is_continuous_mask.shape[1], whisper_feats.shape[2])
                            )
                            full[0, idx] = whisper_feats[0]
                            whisper_feats = full
                        else:
                            logger.warning(
                                "[Kimi-Audio] Mask/feature length mismatch: "
                                "mask_true=%d features=%d",
                                idx.numel(),
                                whisper_feats.shape[1],
                            )

                whisper_input_features = whisper_feats

            # IMPORTANT: Return a single placeholder token in the prompt.
            # The multimodal processor expands it to match multimodal length.

            mm_audio = {
                "whisper_input_features": whisper_input_features,
                "is_continuous_mask": is_continuous_mask,
                "text_input_ids": text_ids,
                "audio_input_ids": audio_ids,
            }

            # IMPORTANT: vLLM's multimodal pipeline expects *placeholder
            # tokens* in the prompt to mark where multimodal items are
            # inserted. Kimi-Audio's true input_ids include non-text ids that
            # a text tokenizer cannot validate/decode, so we keep the prompt
            # ids minimal and represent the whole audio sequence with a single
            # placeholder.
            #
            # The processor's PromptReplacement will expand this single
            # placeholder into a placeholder sequence of the same length as
            # audio_input_ids, ensuring vLLM's placeholder-range bookkeeping
            # matches our tensors.
            # Return a TokensPrompt with the placeholder token directly.
            # This avoids text-based tokenization issues and ensures the
            # placeholder is correctly recognized by the multimodal processor.
            prompt: PromptType = TokensPrompt(
                prompt_token_ids=[KimiAudioASRMultiModalProcessor.PLACEHOLDER_TOKEN_ID],
                multi_modal_data={"audio": mm_audio},
            )

            return prompt

        finally:
            with suppress(OSError):
                os.unlink(wav_path)

    @classmethod
    def post_process_output(cls, text: str) -> str:
        """Post-process transcription output.

        Kimi-Audio sometimes repeats the same sentence when the text EOS token
        is not emitted. If we detect a duplicated sentence, return only the
        first copy. Also normalize common Chinese spacing artifacts.
        """
        if not text:
            return text

        cleaned = text

        if "。" in cleaned:
            parts = [p.strip() for p in cleaned.split("。") if p.strip()]
            if len(parts) >= 2:
                norm0 = "".join(parts[0].split())
                norm1 = "".join(parts[1].split())
                if norm0 == norm1:
                    cleaned = f"{parts[0]}。"

        # Remove extra spaces between CJK characters and punctuation.
        cleaned = re.sub(r"\s*([，。！？；：])\s*", r"\1", cleaned)
        cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
        return cleaned

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | Any:
        # Forward pass delegates to Qwen2ForCausalLM.
        # Audio encoding and mixing is now handled by the multimodal processor
        # (following Whisper's pattern), which returns pre-computed embeddings.
        out = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            self._mask_audio_logits_(out.logits)

        return out

    # Weights loading: reuse Qwen2's loader with audio-specific skipping.
    hf_to_vllm_mapper = WeightsMapper()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Skip audio-specific modules not instantiated in this text-only path.

        # Weight names can appear at the root level or under the Qwen2 `model.` prefix.

        skip_prefixes = {
            "mimo_layers.",
            "mimo_output.",
            "audio_encoder.",
            "speech_encoder.",
            "model.mimo_layers.",
            "model.mimo_output.",
            "model.audio_encoder.",
            "model.speech_encoder.",
        }

        # Also skip nested model prefixes if any.

        # Use a generator to avoid putting all weights in memory at once
        return super().load_weights(
            (name, tensor)
            for name, tensor in weights
            if not any(name.startswith(p) for p in skip_prefixes)
        )
