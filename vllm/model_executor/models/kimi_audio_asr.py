# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kimi-Audio (MoonshotKimiaForCausalLM) with vLLM-native Transcriptions API support.

Goal:
- Enable vLLM OpenAI-compatible endpoint: POST /v1/audio/transcriptions
- Use vLLM engine for attention + decoding
- Do *audio feature + token construction* in Python (via Kimi-Audio prompt manager)
- Do *whisper feature -> hidden adaptor* and *embedding-time mixing* inside the
  vLLM model forward, using the model's own parameters.

Notes
-----
This is an incremental integration:
- We currently depend on the upstream Kimi-Audio repository for preprocessing
  (`kimia_infer.api.prompt_manager.KimiAPromptManager`).
- We do NOT vendor the Kimi-Audio repo into vLLM.

Environment
-----------
Set `KIMI_AUDIO_REPO=/path/to/Kimi-Audio` if it's not in sys.path.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from functools import lru_cache
from typing import Any, ClassVar, Literal

import numpy as np
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

# Debug toggle: set KIMI_AUDIO_ASR_DEBUG=1 to emit one-time diagnostics.
_DEBUG_ASR = os.getenv("KIMI_AUDIO_ASR_DEBUG", "0") not in ("0", "false", "False", "")
_DEBUG_ONCE = True

__all__ = ["KimiAudioForCausalLM"]


# ---- helpers / caching ----


def _ensure_kimi_repo_on_path() -> None:
    repo = os.environ.get("KIMI_AUDIO_REPO", "/root/learning/vllm/Kimi-Audio")
    if os.path.isdir(repo) and repo not in sys.path:
        sys.path.insert(0, repo)


@lru_cache(maxsize=1)
def _get_kimia_prompt_manager(model_path: str):
    """Load the Kimi-Audio prompt manager once per process."""
    _ensure_kimi_repo_on_path()
    # Load kimia-specific offsets from model config.json
    import json

    from kimia_infer.api.prompt_manager import KimiAPromptManager  # type: ignore

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)

    pm = KimiAPromptManager(
        model_path=model_path,
        kimia_token_offset=int(cfg["kimia_token_offset"]),
        kimia_text_audiodelaytokens=int(cfg["kimia_mimo_audiodelaytokens"]),
    )
    return pm


def _write_wav_tmp(audio: np.ndarray, sample_rate: int) -> str:
    """Write float32 waveform to a temporary wav file. Returns the file path."""
    # Convert float32 [-1,1] to int16 PCM.
    x = np.clip(audio, -1.0, 1.0)
    pcm16 = (x * 32767.0).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    wavfile.write(tmp_name, sample_rate, pcm16)
    return tmp_name


# ---- Multimodal processor plumbing ----


def _kimia_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    # All tensors are batched (= one audio item per request).
    # Shape expectations:
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


class KimiAudioASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        # We only need HF config values (token ids) and let vLLM handle weights.
        return self.ctx.model_config.hf_config

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # One audio clip per request for now.
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # Avoid slow dummy multimodal processing at startup.
        # Kimi-Audio maps audio into the *same* token sequence (prompt_token_ids)
        # and provides additional tensors aligned to that sequence, so we cap
        # per-audio token budget to the model max.
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
    ) -> MultiModalDataDict:
        # Return a dummy audio file path for vLLM profiling.
        # The actual preprocessing happens in the processor.
        import os

        # Use a test audio file if available, otherwise return empty
        test_audio = "/root/learning/vllm/Kimi-Audio/test_audios/asr_example.wav"
        if os.path.exists(test_audio):
            return {"audio": test_audio}
        # Fallback: return a dummy tensor that will be handled by custom parser
        bsz = 1
        s = min(seq_len, 128)
        info = self.info
        config = info.get_hf_config()
        whisper_feat_dim = getattr(config, "kimia_adaptor_input_dim", 5120)
        return {
            "audio": {
                "whisper_input_features": torch.zeros(
                    (bsz, s, whisper_feat_dim), dtype=torch.bfloat16
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
        # Handle both dict-of-tensors (from real requests) and
        # string paths (from dummy data during profiling).
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

        # Handle string paths (audio files) - convert to tensors
        if isinstance(data, str):
            try:
                _ensure_kimi_repo_on_path()

                # Get model path from config
                model_path = os.environ.get(
                    "KIMI_AUDIO_MODEL",
                    "/data1/moonshotai/Kimi-Audio-7B-Instruct",
                )
                pm = _get_kimia_prompt_manager(model_path)

                # Process audio file
                messages = [
                    {
                        "role": "user",
                        "message_type": "text",
                        "content": "Transcribe:",
                    },
                    {"role": "user", "message_type": "audio", "content": data},
                ]
                result = pm.get_generation_prompt(messages)

                # Extract tensors from result
                mm_data = result.get("multi_modal_data", {}).get("audio", {})
                if mm_data:
                    return DictEmbeddingItems(
                        mm_data,
                        modality="audio",
                        required_fields={
                            "whisper_input_features",
                            "is_continuous_mask",
                            "text_input_ids",
                            "audio_input_ids",
                        },
                        fields_factory=_kimia_field_config,
                    )
            except Exception as e:
                logger.warning(
                    "[Kimi-Audio] Failed to process audio file %s: %s", data, e
                )
                # Fall through to default handler

        return super()._parse_audio_data(data)


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

    def _get_data_parser(self) -> MultiModalDataParser:
        # SpeechToTextConfig is provided at request-time; for parser we only need
        # a nominal target sample rate.
        return KimiAudioASRMultiModalDataParser(target_sr=16_000)

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
        prompt_ids = [151666]

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
        return [151666]

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
            placeholder_id = 151666
            return [placeholder_id] * max(s, 1)

        # Expand the single placeholder token to cover the full audio sequence
        # length, so that vLLM's placeholder-range bookkeeping matches the
        # shapes of our tensors (audio_input_ids / masks / features).
        placeholder_id = 151666
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
class KimiAudioForCausalLM(Qwen2ForCausalLM, SupportsTranscription, SupportsMultiModal):
    """Kimi-Audio model for conditional generation + transcription."""

    # vLLM V1: treat this as a "raw input only" multimodal model so that
    # multimodal kwargs (whisper_input_features / masks / ids) are forwarded
    # directly into the model forward/embed methods.
    supports_multimodal_raw_input_only = True

    def __init__(self, *, vllm_config, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

        # Manually add vq_adaptor if not present (vLLM may not load it)
        config = vllm_config.model_config.hf_config
        if (
            hasattr(config, "use_whisper_feature")
            and config.use_whisper_feature
            and not hasattr(self.model, "vq_adaptor")
        ):
            input_dim = getattr(config, "kimia_adaptor_input_dim", 5120)
            hidden_size = config.hidden_size
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

            self.model.vq_adaptor = VQAdaptor(input_dim, hidden_size, rms_norm_eps)
            logger.warning(
                "[Kimi-Audio] Manually initialized vq_adaptor (%d -> %d)",
                input_dim,
                hidden_size,
            )

            # Note: weights will be loaded by vLLM's weight loader if present
            # in checkpoint. The vq_adaptor keys are in the safetensors as
            # 'model.vq_adaptor.layers.*'

    # vLLM V1 passes `multimodal_embeddings=` into embed_input_ids when
    # --enable-mm-embeds is set. We override embed_input_ids to:
    # 1) accept that kwarg
    # 2) build the *true* embeddings for Kimi-Audio ASR using audio/text ids
    #    and whisper features.
    def embed_input_ids(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:  # type: ignore[override]
        # Pop V1-only kwargs we don't use directly.
        kwargs.pop("multimodal_embeddings", None)

        whisper_input_features = kwargs.pop("whisper_input_features", None)
        is_continuous_mask = kwargs.pop("is_continuous_mask", None)
        text_input_ids = kwargs.pop("text_input_ids", None)
        audio_input_ids = kwargs.pop("audio_input_ids", None)

        # Squeeze batch dimension if present (vLLM returns batched tensors
        # but embed_input_ids is called per-sequence)
        if (
            isinstance(whisper_input_features, torch.Tensor)
            and whisper_input_features.dim() == 3
        ):
            whisper_input_features = whisper_input_features.squeeze(0)
        if (
            isinstance(is_continuous_mask, torch.Tensor)
            and is_continuous_mask.dim() == 2
        ):
            is_continuous_mask = is_continuous_mask.squeeze(0)
        if isinstance(text_input_ids, torch.Tensor) and text_input_ids.dim() == 2:
            text_input_ids = text_input_ids.squeeze(0)
        if isinstance(audio_input_ids, torch.Tensor) and audio_input_ids.dim() == 2:
            audio_input_ids = audio_input_ids.squeeze(0)

        global _DEBUG_ONCE
        if _DEBUG_ASR and _DEBUG_ONCE:
            _DEBUG_ONCE = False
            try:
                wi = whisper_input_features
                mi = is_continuous_mask
                ti = text_input_ids
                ai = audio_input_ids
                logger.warning(
                    "[Kimi-Audio ASR DEBUG] embed_input_ids received: "
                    "input_ids=%s whisper_input_features=%s is_continuous_mask=%s "
                    "text_input_ids=%s audio_input_ids=%s",
                    tuple(input_ids.shape),
                    None if wi is None else tuple(getattr(wi, "shape", ())),
                    None if mi is None else tuple(getattr(mi, "shape", ())),
                    None if ti is None else tuple(getattr(ti, "shape", ())),
                    None if ai is None else tuple(getattr(ai, "shape", ())),
                )
                if isinstance(mi, torch.Tensor):
                    logger.warning(
                        "[Kimi-Audio ASR DEBUG] is_continuous_mask true_count=%s",
                        int(mi.to(torch.bool).sum().item()),
                    )
                if isinstance(wi, torch.Tensor):
                    logger.warning(
                        "[Kimi-Audio ASR DEBUG] whisper_input_features abs_mean=%s",
                        float(wi.abs().mean().item()),
                    )
            except Exception:
                logger.exception("[Kimi-Audio ASR DEBUG] failed to log mm kwargs")

        true_input_ids = (
            audio_input_ids if isinstance(audio_input_ids, torch.Tensor) else input_ids
        )

        # Base token embeddings
        emb = self.model.embed_tokens(true_input_ids)

        # Add whisper features on masked positions.
        if isinstance(whisper_input_features, torch.Tensor):
            whisper_feats = whisper_input_features.to(emb.device)
            if whisper_feats.dim() == 2:
                whisper_feats = whisper_feats.unsqueeze(0)

            whisper_emb = self.model.vq_adaptor(
                whisper_feats.transpose(0, 1)
            ).transpose(0, 1)

            if isinstance(is_continuous_mask, torch.Tensor):
                mask = is_continuous_mask.to(emb.device)
                if mask.dtype != torch.bool:
                    mask = mask.to(torch.bool)

                whisper_emb = whisper_emb * mask[:, :, None]
                sqrt2 = torch.sqrt(
                    torch.tensor(2.0, dtype=whisper_emb.dtype, device=emb.device)
                )
                encoder_add = (emb + whisper_emb) * sqrt2
                emb = emb * (~mask[:, :, None]) + encoder_add * mask[:, :, None]

        # Add aligned text embeddings (instruction etc.)
        if isinstance(text_input_ids, torch.Tensor) and torch.any(text_input_ids != 0):
            emb = emb + self.model.embed_tokens(text_input_ids.to(emb.device))

        return emb

    # Transcriptions API support
    supported_languages: ClassVar[Mapping[str, str]] = ISO639_1_SUPPORTED_LANGS
    supports_transcription: ClassVar[Literal[True]] = True

    def embed_multimodal(self, **kwargs: object):
        # vLLM expects one embedding tensor per multimodal item.
        # We don't actually *use* mm embeddings for Kimi-Audio ASR (we construct
        # inputs_embeds inside forward()), but we must return correctly-shaped
        # placeholders to satisfy vLLM's startup/profile checks.
        feats = kwargs.get("whisper_input_features")
        if not isinstance(feats, torch.Tensor):
            return []

        # feats: [B, S, F] or [S, F]
        if feats.dim() == 3:
            s = int(feats.shape[1])
        elif feats.dim() == 2:
            s = int(feats.shape[0])
        else:
            s = 1

        hidden = int(
            getattr(self.config, "hidden_size", self.model.embed_tokens.embedding_dim)
        )
        dtype = self.model.embed_tokens.weight.dtype
        device = feats.device

        # Return one item (since we limit audio=1).
        return (torch.zeros((max(s, 1), hidden), device=device, dtype=dtype),)

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
        return SpeechToTextConfig(sample_rate=16_000, max_audio_clip_s=30)

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
        """Construct prompt_token_ids + extra tensors for forward mixing."""
        # Kimi-Audio preprocessing currently relies on upstream prompt manager,
        # which expects an audio file path.
        model_path = model_config.model
        pm = _get_kimia_prompt_manager(model_path)

        wav_path = _write_wav_tmp(audio, int(stt_config.sample_rate))
        try:
            # NOTE: We keep the instruction minimal and language-agnostic.
            # request_prompt can be appended if provided.
            instruction = "请将音频内容转换为文字。"
            if request_prompt:
                instruction = instruction + "\n" + request_prompt

            chats = [
                {"role": "user", "message_type": "text", "content": instruction},
                {"role": "user", "message_type": "audio", "content": wav_path},
            ]
            history = pm.get_prompt(chats, output_type="text")

            audio_input_ids, text_input_ids, is_continuous_mask, _, _ = (
                history.to_tensor()
            )

            # Expand whisper features to sequence length on CPU.
            cfg = model_config.hf_config
            kimia_media_begin = int(cfg.kimia_media_begin)
            kimia_media_end = int(cfg.kimia_media_end)

            input_ids = audio_input_ids[0]
            seq_len = int(input_ids.shape[0])

            media_start_idx = (input_ids == kimia_media_begin).nonzero(as_tuple=False)
            media_end_idx = (input_ids == kimia_media_end).nonzero(as_tuple=False)

            feats = history.continuous_feature
            if not isinstance(feats, list):
                feats = [feats]

            feat_dim = int(feats[0].shape[-1]) if len(feats) > 0 else 1
            expanded = torch.zeros(
                (seq_len, feat_dim),
                dtype=feats[0].dtype if len(feats) > 0 else torch.float32,
            )

            for seg_idx, (s, e) in enumerate(zip(media_start_idx, media_end_idx)):
                start = int(s.item())
                end = int(e.item())
                feat_len = max(end - (start + 1), 0)
                if seg_idx >= len(feats):
                    break
                w = feats[seg_idx]
                if isinstance(w, torch.Tensor):
                    if w.dim() == 3:
                        w = w.squeeze(0)
                    expanded[start + 1 : end, :] = w[:feat_len, :].cpu()

            # Kimi-Audio's prompt manager stores whisper features separately in
            # `continuous_feature`. Its `is_continuous_mask` in the repo code is
            # often all-False for speech tokens, so we build our own mask that
            # marks the [media_begin+1 : media_end) span(s) where whisper features
            # should be added.
            cont_mask = torch.zeros((1, seq_len), dtype=torch.bool)
            for s, e in zip(media_start_idx, media_end_idx):
                start = int(s.item())
                end = int(e.item())
                if end > start + 1:
                    cont_mask[0, start + 1 : end] = True

            mm_audio = {
                # [B, S, F]
                "whisper_input_features": expanded.unsqueeze(0),
                # [B, S]
                "is_continuous_mask": cont_mask,
                # [B, S]
                "text_input_ids": text_input_ids.to(torch.long).cpu(),
                # [B, S] original Kimi audio ids (may include audio-vocab ids)
                "audio_input_ids": input_ids.to(torch.long).unsqueeze(0).cpu(),
            }

            # IMPORTANT:
            # vLLM's multimodal pipeline expects *placeholder tokens* in the
            # prompt to mark where multimodal items are inserted. Kimi-Audio's
            # true `audio_input_ids` include non-text ids that a text tokenizer
            # cannot validate/decode, so we keep the prompt ids minimal and
            # represent the whole audio sequence with a single placeholder.
            #
            # The processor's PromptReplacement will expand this single
            # placeholder into a placeholder sequence of the same length as
            # `audio_input_ids`, ensuring vLLM's placeholder-range bookkeeping
            # matches our tensors.
            placeholder_id = 151666

            prompt: TokensPrompt = {
                "prompt_token_ids": [placeholder_id],
                "multi_modal_data": {"audio": mm_audio},
            }
            return prompt
        finally:
            with suppress(OSError):
                os.unlink(wav_path)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        # Pull out our extra multimodal tensors
        # (added by KimiAudioASRMultiModalProcessor).
        whisper_input_features = kwargs.pop("whisper_input_features", None)
        is_continuous_mask = kwargs.pop("is_continuous_mask", None)
        text_input_ids = kwargs.pop("text_input_ids", None)
        audio_input_ids = kwargs.pop("audio_input_ids", None)

        # vLLM forward provides input_ids (bookkeeping ids). For Kimi-Audio we
        # may also receive `audio_input_ids` containing the true ids.
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]

        true_input_ids = audio_input_ids
        if isinstance(true_input_ids, torch.Tensor):
            # [B,S]
            if true_input_ids.dim() == 3:
                true_input_ids = true_input_ids.squeeze(0)
        else:
            true_input_ids = input_ids

        # IMPORTANT (V1): vLLM may provide `inputs_embeds` that do not include
        # our audio conditioning (because it was computed from placeholder token
        # ids). If we have the raw multimodal tensors, always rebuild
        # `inputs_embeds` here so whisper features are applied.
        if true_input_ids is not None and whisper_input_features is not None:
            # Compute inputs_embeds using this model's weights.
            audio_emb = self.model.embed_tokens(true_input_ids)

            # whisper_input_features: [B, S, F] float
            device = audio_emb.device
            whisper_feats = whisper_input_features.to(device)
            if whisper_feats.dim() == 2:
                whisper_feats = whisper_feats.unsqueeze(0)

            # vq_adaptor expects [S, B, F]
            adaptor = getattr(self, "vq_adaptor", None)
            if adaptor is None:
                adaptor = getattr(self.model, "vq_adaptor", None)
            if adaptor is None:
                # Some vLLM model wrappers may not include the Kimi-Audio
                # adaptor module yet. In that case, we cannot apply whisper
                # conditioning.
                return super().forward(*args, **kwargs)

            whisper_emb = adaptor(whisper_feats.transpose(0, 1)).transpose(0, 1)

            mask = (
                is_continuous_mask.to(device)
                if is_continuous_mask is not None
                else None
            )
            if mask is not None:
                if mask.dtype != torch.bool:
                    mask = mask.to(torch.bool)
                whisper_emb = whisper_emb * mask[:, :, None]

                sqrt2 = torch.sqrt(
                    torch.tensor(2.0, dtype=whisper_emb.dtype, device=device)
                )
                encoder_add = (audio_emb + whisper_emb) * sqrt2
                audio_emb = (
                    audio_emb * (~mask[:, :, None]) + encoder_add * mask[:, :, None]
                )

            inputs_embeds = audio_emb
            if text_input_ids is not None and torch.any(text_input_ids != 0):
                inputs_embeds = inputs_embeds + self.model.embed_tokens(
                    text_input_ids.to(device)
                )

            kwargs["inputs_embeds"] = inputs_embeds

        out = super().forward(*args, **kwargs)
        if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            self._mask_audio_logits_(out.logits)
        return out

    # Weights loading: reuse Qwen2's loader, but skip unsupported audio-specific parts.
    hf_to_vllm_mapper = WeightsMapper()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Reuse the existing text-only skipping logic: ignore audio towers if present.
        # Kimi-Audio checkpoints include audio-specific modules that vLLM doesn't
        # instantiate in this text-only/transcription-only path.
        # Weight names can appear at the root level or under the Qwen2 `model.` prefix.
        skip_prefixes = {
            "mimo_layers.",
            "mimo_output.",
            "audio_tower.",
            "audio_encoder.",
            "speech_encoder.",
            "model.mimo_layers.",
            "model.mimo_output.",
            "model.audio_tower.",
            "model.audio_encoder.",
            "model.speech_encoder.",
        }

        # Also skip nested model prefixes if any.
        mapped = []
        for name, tensor in weights:
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            mapped.append((name, tensor))

        return super().load_weights(mapped)
