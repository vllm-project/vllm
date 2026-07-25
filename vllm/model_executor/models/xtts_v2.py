# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XTTS-v2: Coqui multilingual text-to-speech model.

Architecture overview
---------------------
XTTS-v2 is a two-stage TTS system:

1. **GPT stage** (``XttsV2GPT``): an autoregressive transformer that
   consumes text token embeddings + speaker conditioning and generates a
   sequence of discrete mel/audio tokens.  This stage runs inside the
   standard vLLM generation loop and benefits from continuous batching,
   KV-cache management, and tensor parallelism.

2. **HiFi-GAN decoder** (``XttsV2HifiGAN``): a convolutional vocoder that
   converts discrete audio tokens produced by the GPT stage into a raw
   24 kHz waveform.  This stage is invoked *after* each request's GPT
   generation completes, so it never blocks the scheduler.

Speaker conditioning is obtained by running a small DVAE encoder over a
reference audio clip (≥ 6 s recommended) to produce a 512-D embedding.
This embedding is prepended to the GPT input as a soft prompt, enabling
zero-shot voice cloning without any fine-tuning.

Reference
---------
* Coqui XTTS-v2 weights: ``coqui/XTTS-v2`` on Hugging Face Hub.
* Community vLLM integration:
  https://github.com/wuxuedaifu/xttsv2-vllm-streaming-server
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsSpeechSynthesis
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import ModelConfig, TextToSpeechConfig, TextToSpeechParams


# ---------------------------------------------------------------------------
# XTTS-v2 language table (ISO 639-1 → human-readable name).
# 17 languages supported by the official Coqui checkpoint.
# ---------------------------------------------------------------------------
XTTS_V2_LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh": "Chinese",
    "hu": "Hungarian",
    "ko": "Korean",
    "ja": "Japanese",
    "hi": "Hindi",
}


# ---------------------------------------------------------------------------
# HuggingFace-compatible config
# ---------------------------------------------------------------------------

class XttsV2Config(PretrainedConfig):
    """Configuration for XTTS-v2.

    Mirrors the hyperparameters used by the official Coqui checkpoint.
    """

    model_type = "xtts_v2"

    def __init__(
        self,
        # GPT stage
        gpt_layers: int = 30,
        gpt_n_heads: int = 16,
        gpt_n_model_channels: int = 1024,
        gpt_n_audio_tokens: int = 8192,
        gpt_start_audio_token: int = 8192,
        gpt_stop_audio_token: int = 8193,
        gpt_n_text_tokens: int = 6681,
        gpt_n_text_start_stop_tokens: int = 2,
        gpt_max_audio_tokens: int = 605,
        gpt_max_text_tokens: int = 402,
        gpt_batch_size: int = 1,
        gpt_code_stride_len: int = 1024,
        gpt_use_masking_gt_prompt_approach: bool = True,
        # Speaker conditioning / DVAE
        gpt_cond_input_dim: int = 512,
        gpt_cond_len: int = 6,
        gpt_cond_chunk_len: int = 4,
        # HiFi-GAN decoder
        decoder_input_dim: int = 1024,
        decoder_hidden_size: int = 1024,
        decoder_upsample_rates: list[int] | None = None,
        decoder_upsample_initial_channel: int = 512,
        decoder_resblock_kernel_sizes: list[int] | None = None,
        decoder_resblock_dilation_sizes: list[list[int]] | None = None,
        decoder_resblock_type: str = "1",
        decoder_output_channels: int = 1,
        # Audio properties
        audio_sample_rate: int = 24000,
        # Misc
        tokenizer_name_or_path: str = "coqui/XTTS-v2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # GPT stage
        self.gpt_layers = gpt_layers
        self.gpt_n_heads = gpt_n_heads
        self.gpt_n_model_channels = gpt_n_model_channels
        self.gpt_n_audio_tokens = gpt_n_audio_tokens
        self.gpt_start_audio_token = gpt_start_audio_token
        self.gpt_stop_audio_token = gpt_stop_audio_token
        self.gpt_n_text_tokens = gpt_n_text_tokens
        self.gpt_n_text_start_stop_tokens = gpt_n_text_start_stop_tokens
        self.gpt_max_audio_tokens = gpt_max_audio_tokens
        self.gpt_max_text_tokens = gpt_max_text_tokens
        self.gpt_batch_size = gpt_batch_size
        self.gpt_code_stride_len = gpt_code_stride_len
        self.gpt_use_masking_gt_prompt_approach = gpt_use_masking_gt_prompt_approach
        # Speaker conditioning
        self.gpt_cond_input_dim = gpt_cond_input_dim
        self.gpt_cond_len = gpt_cond_len
        self.gpt_cond_chunk_len = gpt_cond_chunk_len
        # HiFi-GAN
        self.decoder_input_dim = decoder_input_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_upsample_rates = decoder_upsample_rates or [
            8, 8, 2, 2
        ]
        self.decoder_upsample_initial_channel = decoder_upsample_initial_channel
        self.decoder_resblock_kernel_sizes = decoder_resblock_kernel_sizes or [
            3, 7, 11
        ]
        self.decoder_resblock_dilation_sizes = (
            decoder_resblock_dilation_sizes
            or [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        )
        self.decoder_resblock_type = decoder_resblock_type
        self.decoder_output_channels = decoder_output_channels
        self.audio_sample_rate = audio_sample_rate
        self.tokenizer_name_or_path = tokenizer_name_or_path

    @property
    def gpt_vocab_size(self) -> int:
        """Total vocabulary seen by the GPT: text + audio + BOS/EOS."""
        return (
            self.gpt_n_text_tokens
            + self.gpt_n_text_start_stop_tokens
            + self.gpt_n_audio_tokens
            + 2  # audio BOS / EOS
        )


# ---------------------------------------------------------------------------
# GPT stage
# ---------------------------------------------------------------------------

class XttsV2GPTAttention(nn.Module):
    """Multi-head causal self-attention for the XTTS GPT decoder."""

    def __init__(
        self,
        config: XttsV2Config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.gpt_n_model_channels
        tp = get_tensor_model_parallel_world_size()
        self.total_heads = config.gpt_n_heads
        assert self.total_heads % tp == 0, (
            f"gpt_n_heads ({self.total_heads}) must be divisible by TP world "
            f"size ({tp})."
        )
        self.num_heads = self.total_heads // tp
        self.head_dim = self.hidden_size // self.total_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_heads,
            total_num_kv_heads=self.total_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.gpt_max_audio_tokens + config.gpt_max_text_tokens,
            base=10_000,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scale,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim,
            ],
            dim=-1,
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)
        out, _ = self.out_proj(attn_out)
        return out


class XttsV2GPTMLP(nn.Module):
    def __init__(
        self,
        config: XttsV2Config,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        hidden = config.gpt_n_model_channels
        intermediate = hidden * 4
        self.fc1 = ColumnParallelLinear(
            hidden, intermediate, quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            intermediate, hidden, quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.act = get_act_fn("gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class XttsV2GPTDecoderLayer(nn.Module):
    def __init__(
        self,
        config: XttsV2Config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.gpt_n_model_channels)
        self.attn = XttsV2GPTAttention(
            config, cache_config, quant_config, layer_idx,
            prefix=f"{prefix}.attn",
        )
        self.ln2 = nn.LayerNorm(config.gpt_n_model_channels)
        self.mlp = XttsV2GPTMLP(config, quant_config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class XttsV2GPT(nn.Module):
    """Autoregressive transformer for mel-token generation.

    Receives: (text_embeds ‖ speaker_cond_embeds) prepended to the
    autoregressive audio-token context.  Produces logits over the audio
    codebook at each step, which the vLLM sampler then greedy- or
    temperature-decodes.
    """

    def __init__(
        self,
        config: XttsV2Config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        hidden = config.gpt_n_model_channels

        # Token embeddings for the unified vocabulary (text + audio tokens)
        self.text_embedding = VocabParallelEmbedding(
            config.gpt_n_text_tokens + config.gpt_n_text_start_stop_tokens,
            hidden,
            prefix=f"{prefix}.text_embedding",
        )
        self.mel_embedding = VocabParallelEmbedding(
            config.gpt_n_audio_tokens + 2,  # +2 for BOS/EOS audio tokens
            hidden,
            prefix=f"{prefix}.mel_embedding",
        )

        # Learned positional embeddings (GPT-2-style absolute positions)
        max_pos = config.gpt_max_text_tokens + config.gpt_max_audio_tokens + 10
        self.text_pos_embedding = nn.Embedding(max_pos, hidden)
        self.mel_pos_embedding = nn.Embedding(max_pos, hidden)

        # Speaker conditioning projection: DVAE 512-D → hidden
        self.speaker_embedding = nn.Linear(
            config.gpt_cond_input_dim, hidden, bias=True
        )

        self.layers = nn.ModuleList([
            XttsV2GPTDecoderLayer(
                config, cache_config, quant_config,
                layer_idx=i,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.gpt_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden)

        # Output head over the audio codebook only
        self.audio_head = ParallelLMHead(
            config.gpt_n_audio_tokens + 2,
            hidden,
            prefix=f"{prefix}.audio_head",
        )
        self.logits_processor = LogitsProcessor(config.gpt_n_audio_tokens + 2)
        self.sampler = get_sampler()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed text or audio token IDs.

        IDs < (n_text_tokens + n_text_start_stop) are text tokens;
        higher IDs are audio tokens shifted back to the mel vocabulary.
        """
        text_vocab_size = (
            self.config.gpt_n_text_tokens
            + self.config.gpt_n_text_start_stop_tokens
        )
        text_mask = input_ids < text_vocab_size
        out = torch.zeros(
            (*input_ids.shape, self.config.gpt_n_model_channels),
            device=input_ids.device,
            dtype=self.text_embedding.weight.dtype,
        )
        if text_mask.any():
            out[text_mask] = self.text_embedding(input_ids[text_mask])
        audio_mask = ~text_mask
        if audio_mask.any():
            audio_ids = input_ids[audio_mask] - text_vocab_size
            out[audio_mask] = self.mel_embedding(audio_ids)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata,
        intermediate_tensors: IntermediateTensors | None,
        speaker_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        if speaker_embeddings is not None:
            # Inject speaker conditioning as an additive bias on the first
            # token position to avoid changing the sequence length visible
            # to the scheduler.
            cond = self.speaker_embedding(speaker_embeddings)  # (B, hidden)
            hidden_states[0] = hidden_states[0] + cond

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                positions, hidden_states, kv_caches[layer_idx], attn_metadata
            )

        return self.final_norm(hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        return self.logits_processor(
            self.audio_head, hidden_states, sampling_metadata
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)


# ---------------------------------------------------------------------------
# HiFi-GAN decoder (vocoder)
# ---------------------------------------------------------------------------

class _ResBlock1(nn.Module):
    """HiFi-GAN residual block type 1."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels, channels, kernel_size, 1,
                    dilation=d, padding=(kernel_size - 1) * d // 2,
                )
            )
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels, channels, kernel_size, 1,
                    dilation=1, padding=(kernel_size - 1) // 2,
                )
            )
            for _ in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x


class XttsV2HifiGAN(nn.Module):
    """HiFi-GAN vocoder that maps discrete audio tokens → waveform.

    This module is called *outside* the vLLM generation loop (after each
    request's GPT decoding finishes) so its compute does not block the
    continuous-batching scheduler.
    """

    def __init__(self, config: XttsV2Config):
        super().__init__()
        self.config = config
        # Project audio token embeddings to the decoder channel space
        self.pre_conv = nn.utils.weight_norm(
            nn.Conv1d(
                config.decoder_input_dim,
                config.decoder_upsample_initial_channel,
                kernel_size=7, stride=1, padding=3,
            )
        )
        self.ups = nn.ModuleList()
        in_channels = config.decoder_upsample_initial_channel
        for rate in config.decoder_upsample_rates:
            out_channels = in_channels // 2
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        in_channels, out_channels,
                        kernel_size=rate * 2, stride=rate,
                        padding=rate // 2,
                    )
                )
            )
            in_channels = out_channels

        # Residual blocks
        self.resblocks = nn.ModuleList()
        channels = config.decoder_upsample_initial_channel
        for rate in config.decoder_upsample_rates:
            channels //= 2
            for k, d in zip(
                config.decoder_resblock_kernel_sizes,
                config.decoder_resblock_dilation_sizes,
            ):
                self.resblocks.append(_ResBlock1(channels, k, tuple(d)))

        self.post_conv = nn.utils.weight_norm(
            nn.Conv1d(channels, config.decoder_output_channels, 7, 1, padding=3)
        )

        # Codebook: map discrete audio token IDs to continuous feature vectors
        # before passing into the convolutional decoder.
        self.code_embedding = nn.Embedding(
            config.gpt_n_audio_tokens, config.decoder_input_dim
        )

        self._num_resblocks = len(config.decoder_resblock_kernel_sizes)

    def forward(self, audio_token_ids: torch.Tensor) -> torch.Tensor:
        """Decode a 1-D sequence of audio token IDs to a waveform.

        Args:
            audio_token_ids: Long tensor of shape ``(T,)`` containing
                indices into the XTTS-v2 audio codebook.

        Returns:
            Float32 tensor of shape ``(N,)`` representing the PCM waveform
            at ``config.audio_sample_rate`` Hz.
        """
        # (T,) → (T, D) → (1, D, T) for conv1d
        codes = self.code_embedding(audio_token_ids)
        x = codes.T.unsqueeze(0)  # (1, D, T)
        x = self.pre_conv(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs: torch.Tensor | None = None
            for rb_idx in range(self._num_resblocks):
                res = self.resblocks[i * self._num_resblocks + rb_idx](x)
                xs = res if xs is None else xs + res
            x = xs / self._num_resblocks  # type: ignore[operator]
        x = F.leaky_relu(x, 0.1)
        x = self.post_conv(x)
        x = torch.tanh(x)
        return x.squeeze(0).squeeze(0)  # (N,)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class XttsV2ForConditionalGeneration(nn.Module, SupportsSpeechSynthesis):
    """XTTS-v2: two-stage TTS model surfaced as a vLLM model.

    The GPT stage integrates with the vLLM scheduling engine for efficient
    batched inference.  The HiFi-GAN vocoder runs as a post-processing step
    via ``decode_audio``, which is called by the serving layer after the
    autoregressive decoding loop finishes for a request.
    """

    # SupportsMultiModal / SupportsSpeechSynthesis class vars
    supports_speech_synthesis: ClassVar[Literal[True]] = True
    supports_voice_cloning: ClassVar[bool] = True
    supported_languages: ClassVar[dict[str, str]] = XTTS_V2_LANGUAGES

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: XttsV2Config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.gpt = XttsV2GPT(
            config, cache_config, quant_config, prefix=f"{prefix}.gpt"
        )
        self.hifigan = XttsV2HifiGAN(config)

    # ------------------------------------------------------------------
    # SupportsSpeechSynthesis interface
    # ------------------------------------------------------------------

    @classmethod
    def get_tts_config(cls, model_config: "ModelConfig") -> "TextToSpeechConfig":
        from vllm.config import TextToSpeechConfig
        hf: XttsV2Config = model_config.hf_config
        return TextToSpeechConfig(
            sample_rate=hf.audio_sample_rate,
            max_text_tokens=hf.gpt_max_text_tokens,
            max_mel_tokens=hf.gpt_max_audio_tokens,
            speaker_embedding_dim=hf.gpt_cond_input_dim,
        )

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None or language in cls.supported_languages:
            return language
        raise ValueError(
            f"Unsupported language: {language!r}. "
            f"Supported: {sorted(cls.supported_languages)}."
        )

    def decode_audio(
        self,
        mel_token_ids: torch.Tensor,
        tts_params: "TextToSpeechParams",
    ) -> np.ndarray:
        """Decode GPT-generated audio tokens to a PCM waveform.

        Strips BOS/EOS audio tokens before decoding.  Called by the
        vLLM serving layer *after* the GPT generation loop completes,
        so it never occupies the continuous-batching scheduler.

        Args:
            mel_token_ids: 1-D long tensor of raw audio token IDs
                (including any BOS/EOS boundaries) as produced by the
                vLLM sampler.
            tts_params: Full TTS request parameters.

        Returns:
            Float32 numpy array of shape ``(num_samples,)`` at 24 kHz.
        """
        text_vocab = (
            self.config.gpt_n_text_tokens
            + self.config.gpt_n_text_start_stop_tokens
        )
        bos_id = text_vocab + self.config.gpt_start_audio_token
        eos_id = text_vocab + self.config.gpt_stop_audio_token

        # Filter out BOS/EOS and shift back to codebook range.
        audio_ids = mel_token_ids[
            (mel_token_ids != bos_id) & (mel_token_ids != eos_id)
        ] - text_vocab
        # Clamp to valid codebook range.
        audio_ids = audio_ids.clamp(0, self.config.gpt_n_audio_tokens - 1)
        with torch.inference_mode():
            waveform = self.hifigan(audio_ids)
        return waveform.float().cpu().numpy()

    # ------------------------------------------------------------------
    # vLLM model interface
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        speaker_embeddings: torch.Tensor | None = kwargs.get(
            "speaker_embeddings", None
        )
        return self.gpt(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            speaker_embeddings=speaker_embeddings,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        return self.gpt.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.gpt.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from the Coqui checkpoint.

        The HuggingFace checkpoint uses nested key names under
        ``gpt.*`` and ``hifigan.*`` (or ``decoder.*``).  Unmapped keys
        are logged and silently skipped.
        """
        stacked = {
            "gpt.layers": {},
        }
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Normalise Coqui checkpoint key conventions.
            name = name.replace("xtts.", "")
            if name not in params:
                continue
            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
