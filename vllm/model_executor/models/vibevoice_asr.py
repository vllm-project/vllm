# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VibeVoice ASR model.

Reference: https://github.com/microsoft/VibeVoice
Architecture: Acoustic+Semantic VAE encoder → SpeechConnector → Qwen2 decoder.

The acoustic/semantic tokenizer encoder code is ported from the VibeVoice
package (MIT licence) to avoid an external runtime dependency.
"""

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    ClassVar,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
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
from vllm.sequence import IntermediateTensors

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Config classes (replaces vibevoice.modular.configuration_vibevoice)
# ---------------------------------------------------------------------------


class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = "gaussian",
        mixer_layer: str = "depthwise_conv",
        conv_norm: str = "none",
        pad_mode: str = "constant",
        disable_last_norm: bool = True,
        layernorm: str = "RMSNorm",
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: list | None = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        decoder_n_filters: int = 32,
        decoder_ratios: list | None = None,
        decoder_depths: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        self.mixer_layer = mixer_layer
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_eps = layernorm_eps
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = (
            encoder_ratios if encoder_ratios is not None else [8, 5, 5, 4, 2, 2]
        )
        self.encoder_depths = encoder_depths
        self.decoder_n_filters = decoder_n_filters
        self.decoder_ratios = (
            decoder_ratios if decoder_ratios is not None else self.encoder_ratios
        )
        self.decoder_depths = decoder_depths


class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        std_dist_type: str = "none",
        mixer_layer: str = "depthwise_conv",
        conv_norm: str = "none",
        pad_mode: str = "constant",
        disable_last_norm: bool = True,
        layernorm: str = "RMSNorm",
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: list | None = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        self.mixer_layer = mixer_layer
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_eps = layernorm_eps
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = (
            encoder_ratios if encoder_ratios is not None else [8, 5, 5, 4, 2, 2]
        )
        self.encoder_depths = encoder_depths


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------


class _ConvLayerNorm(nn.LayerNorm):
    """LayerNorm applied along the channel dimension of (B, C, T) tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).to(x.dtype)
        return x.transpose(1, 2)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True) + self.eps
        out = x.float() * torch.rsqrt(var)
        out = out.to(x.dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


class _ConvRMSNorm(_RMSNorm):
    """RMSNorm applied along the channel dim of (B, C, T) tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = super().forward(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# Causal Conv1d utilities (non-streaming only)
# ---------------------------------------------------------------------------


def _extra_padding(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return max(0, ideal - length)


def _pad1d(
    x: torch.Tensor, paddings: tuple[int, int], mode: str = "zero", value: float = 0.0
) -> torch.Tensor:
    pad_l, pad_r = paddings
    if mode == "reflect":
        max_pad = max(pad_l, pad_r)
        extra = 0
        if x.shape[-1] <= max_pad:
            extra = max_pad - x.shape[-1] + 1
            x = F.pad(x, (0, extra))
        out = F.pad(x, paddings, "reflect")
        return out[..., : out.shape[-1] - extra]
    return F.pad(x, paddings, mode if mode != "zero" else "constant", value)


class _NormConv1d(nn.Module):
    def __init__(self, *args, norm: str = "none", **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        if norm == "weight_norm":
            self.conv = nn.utils.weight_norm(self.conv)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _SConv1d(nn.Module):
    """Causal strided Conv1d with automatic padding (non-streaming)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        norm: str = "none",
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.conv = _NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            norm=norm,
        )
        self.causal = causal
        self.pad_mode = pad_mode
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)

    def forward(self, x: torch.Tensor, **_ignored) -> torch.Tensor:
        extra = _extra_padding(x, self.kernel_size, self.stride, self.padding_total)
        if self.causal:
            x = _pad1d(x, (self.padding_total, extra), mode=self.pad_mode)
        else:
            pad_r = self.padding_total // 2
            pad_l = self.padding_total - pad_r
            x = _pad1d(x, (pad_l, pad_r + extra), mode=self.pad_mode)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder building blocks
# ---------------------------------------------------------------------------


class _FFN(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, bias: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.gelu = ACT2FN["gelu"]
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.gelu(self.linear1(x)))


class _Convlayer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        groups: int = 1,
        pad_mode: str = "reflect",
        norm: str = "none",
        causal: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = _SConv1d(
            in_ch,
            out_ch,
            kernel_size,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            pad_mode=pad_mode,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.conv(x, **kwargs)


class _Block1D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        drop_path: float = 0.0,
        mixer_layer: str = "conv",
        layer_scale_init_value: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        ln_type = kwargs.get("layernorm", "LN")
        eps = kwargs.get("eps", 1e-6)
        if ln_type == "LN":
            self.norm = _ConvLayerNorm(dim, eps=eps)
            self.ffn_norm = _ConvLayerNorm(dim, eps=eps)
        else:
            elem_affine = kwargs.get("elementwise_affine", True)
            self.norm = _ConvRMSNorm(dim, eps=eps, elementwise_affine=elem_affine)
            self.ffn_norm = _ConvRMSNorm(dim, eps=eps, elementwise_affine=elem_affine)

        groups = dim if mixer_layer == "depthwise_conv" else kwargs.get("groups", 1)
        self.mixer = _Convlayer(
            dim,
            dim,
            kernel_size,
            groups=groups,
            pad_mode=kwargs.get("pad_mode", "reflect"),
            norm=kwargs.get("norm", "none"),
            causal=kwargs.get("causal", True),
            bias=kwargs.get("bias", True),
        )
        self.ffn = _FFN(dim, 4 * dim, bias=False)
        self.drop_path = nn.Identity()

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)
        return x


# ---------------------------------------------------------------------------
# TokenizerEncoder (ported from vibevoice.modular.modular_vibevoice_tokenizer)
# ---------------------------------------------------------------------------


class _TokenizerEncoder(nn.Module):
    """VAE encoder: raw audio waveform → latent representation."""

    def __init__(self, config: Any):
        super().__init__()
        self.channels = config.channels
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = list(reversed(config.ratios))
        self.depths = config.depths
        self.causal = config.causal

        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        elem_affine = getattr(config, "layernorm_elementwise_affine", True)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        ls_init = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        norm_cls: type
        if layernorm == "LN":
            norm_cls = partial(_ConvLayerNorm, eps=layernorm_eps)
        else:
            norm_cls = partial(
                _ConvRMSNorm, eps=layernorm_eps, elementwise_affine=elem_affine
            )

        stem = nn.Sequential(
            _SConv1d(
                self.channels,
                self.n_filters,
                kernel_size,
                norm=norm,
                causal=self.causal,
                pad_mode=pad_mode,
                bias=bias,
            ),
        )
        self.downsample_layers = nn.ModuleList([stem])
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2**i)
            out_ch = self.n_filters * (2 ** (i + 1))
            self.downsample_layers.append(
                nn.Sequential(
                    _SConv1d(
                        in_ch,
                        out_ch,
                        kernel_size=self.ratios[i] * 2,
                        stride=self.ratios[i],
                        causal=self.causal,
                        pad_mode=pad_mode,
                        norm=norm,
                        bias=bias,
                    ),
                )
            )

        block_kwargs = dict(
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            elementwise_affine=elem_affine,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=ls_init,
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, 0.0, sum(self.depths))]
        cur = 0
        for i, depth in enumerate(self.depths):
            in_ch = self.n_filters * (2**i)
            stage = nn.Sequential(
                *[
                    _Block1D(dim=in_ch, drop_path=dp_rates[cur + j], **block_kwargs)
                    for j in range(depth)
                ]
            )
            self.stages.append(stage)
            cur += depth

        final_ch = self.n_filters * (2 ** (len(self.depths) - 1))
        self.norm = nn.Identity() if disable_last_norm else norm_cls(final_ch)
        self.head = _SConv1d(
            final_ch,
            self.dimension,
            kernel_size=last_kernel_size,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.depths)):
            for layer in self.downsample_layers[i]:
                x = layer(x)
            x = self.stages[i](x)
        x = self.norm(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Encoder output dataclass
# ---------------------------------------------------------------------------


@dataclass
class _TokenizerEncoderOutput:
    mean: torch.Tensor
    std: float | torch.Tensor | None = None

    @property
    def sample_mean(self) -> torch.Tensor:
        return self.mean


# ---------------------------------------------------------------------------
# VAE tokenizer model wrappers
# ---------------------------------------------------------------------------


class _AcousticTokenizerModel(PreTrainedModel):
    config_class = VibeVoiceAcousticTokenizerConfig
    base_model_prefix = "vibevoice_acoustic_tokenizer"

    def __init__(self, config: VibeVoiceAcousticTokenizerConfig):
        super().__init__(config)
        self.register_buffer("fix_std", torch.tensor(config.fix_std), persistent=False)
        self.std_dist_type = getattr(config, "std_dist_type", "fix")

        depths = (
            [int(d) for d in config.encoder_depths.split("-")]
            if isinstance(config.encoder_depths, str)
            else list(config.encoder_depths)
        )

        enc_cfg = copy.copy(config)
        enc_cfg.dimension = config.vae_dim
        enc_cfg.n_filters = config.encoder_n_filters
        enc_cfg.ratios = config.encoder_ratios
        enc_cfg.depths = depths
        enc_cfg.norm = config.conv_norm
        enc_cfg.bias = config.conv_bias
        self.encoder = _TokenizerEncoder(enc_cfg)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> _TokenizerEncoderOutput:
        latents = self.encoder(audio)  # (B, D, T)
        return _TokenizerEncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)


class _SemanticTokenizerModel(PreTrainedModel):
    config_class = VibeVoiceSemanticTokenizerConfig
    base_model_prefix = "vibevoice_semantic_tokenizer"

    def __init__(self, config: VibeVoiceSemanticTokenizerConfig):
        super().__init__(config)

        depths = (
            [int(d) for d in config.encoder_depths.split("-")]
            if isinstance(config.encoder_depths, str)
            else list(config.encoder_depths)
        )

        enc_cfg = copy.copy(config)
        enc_cfg.dimension = config.vae_dim
        enc_cfg.n_filters = config.encoder_n_filters
        enc_cfg.ratios = config.encoder_ratios
        enc_cfg.depths = depths
        enc_cfg.norm = config.conv_norm
        enc_cfg.bias = config.conv_bias
        self.encoder = _TokenizerEncoder(enc_cfg)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> _TokenizerEncoderOutput:
        latents = self.encoder(audio)
        return _TokenizerEncoderOutput(mean=latents.permute(0, 2, 1))


# ---------------------------------------------------------------------------
# SpeechConnector and VibeVoiceAudioEncoder
# ---------------------------------------------------------------------------


class _SpeechConnector(nn.Module):
    """Projects VAE latent → LM hidden dim: fc1 → RMSNorm → fc2."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = _RMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(self.fc1(x)))


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class VibeVoiceAudioEncoder(nn.Module):
    """Combines acoustic + semantic tokenizers and their projection connectors."""

    SAMPLE_RATE: int = 24_000

    def __init__(self, config: Any):
        super().__init__()
        acoustic_cfg_dict = _cfg_get(config, "acoustic_tokenizer_config", {})
        semantic_cfg_dict = _cfg_get(config, "semantic_tokenizer_config", {})
        decoder_cfg = _cfg_get(config, "decoder_config")

        self.acoustic_vae_dim = _cfg_get(config, "acoustic_vae_dim", 64)
        self.semantic_vae_dim = _cfg_get(config, "semantic_vae_dim", 128)

        hidden_size = (
            _cfg_get(decoder_cfg, "hidden_size", 3584)
            if decoder_cfg is not None
            else 3584
        )

        # Build tokenizer configs from the nested dicts in the HF config
        if isinstance(acoustic_cfg_dict, dict):
            acoustic_cfg = VibeVoiceAcousticTokenizerConfig(
                **{k: v for k, v in acoustic_cfg_dict.items() if k != "model_type"}
            )
        else:
            acoustic_cfg = acoustic_cfg_dict

        if isinstance(semantic_cfg_dict, dict):
            semantic_cfg = VibeVoiceSemanticTokenizerConfig(
                **{k: v for k, v in semantic_cfg_dict.items() if k != "model_type"}
            )
        else:
            semantic_cfg = semantic_cfg_dict

        self.acoustic_tokenizer = _AcousticTokenizerModel(acoustic_cfg)
        self.semantic_tokenizer = _SemanticTokenizerModel(semantic_cfg)

        self.acoustic_connector = _SpeechConnector(self.acoustic_vae_dim, hidden_size)
        self.semantic_connector = _SpeechConnector(self.semantic_vae_dim, hidden_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to LM-compatible embeddings.

        Args:
            audio: (B, T) or (T,) float32 tensor at 24 kHz.

        Returns:
            (B, N, hidden_size) embeddings.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Tokenizers expect (B, C, T)
        audio_input = audio.unsqueeze(1)

        with torch.no_grad():
            acoustic_out = self.acoustic_tokenizer.encode(audio_input)
            acoustic_embeds = self.acoustic_connector(acoustic_out.mean)

            semantic_out = self.semantic_tokenizer.encode(audio_input)
            semantic_embeds = self.semantic_connector(semantic_out.mean)

        return acoustic_embeds + semantic_embeds


# ---------------------------------------------------------------------------
# vLLM multimodal processing infrastructure
# ---------------------------------------------------------------------------

_AUDIO_TOKEN = "<|AUDIO|>"
_AUDIO_BOS_TOKEN = "<|audio_bos|>"
_AUDIO_EOS_TOKEN = "<|audio_eos|>"
_DEFAULT_COMPRESS_RATIO = 3200  # prod([8,5,5,4,2,2])
_DEFAULT_SAMPLE_RATE = 24_000


class VibeVoiceProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Any:
        return self.ctx.get_hf_config()

    def get_feature_extractor(self, **kwargs) -> WhisperFeatureExtractor:
        # Not used for inference; required by vLLM profiling infrastructure.
        return WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=_DEFAULT_SAMPLE_RATE,
            hop_length=240,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
        )

    def get_audio_token_info(self) -> dict:
        tokenizer = self.get_tokenizer()
        vocab = tokenizer.get_vocab()
        return {
            "audio_token": _AUDIO_TOKEN,
            "audio_token_id": vocab.get(_AUDIO_TOKEN),
            "audio_bos_token": _AUDIO_BOS_TOKEN,
            "audio_bos_id": vocab.get(_AUDIO_BOS_TOKEN),
            "audio_eos_token": _AUDIO_EOS_TOKEN,
            "audio_eos_id": vocab.get(_AUDIO_EOS_TOKEN),
        }

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        cfg = self.get_hf_config()
        compress_ratio = int(
            _cfg_get(cfg, "speech_tok_compress_ratio", _DEFAULT_COMPRESS_RATIO)
        )
        sr = int(_cfg_get(cfg, "target_sample_rate", _DEFAULT_SAMPLE_RATE))
        max_tokens = int(np.ceil(61 * 60 * sr / compress_ratio)) + 3
        return {"audio": min(max_tokens, seq_len)}


class VibeVoiceDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        n = mm_counts.get("audio", 0)
        return _AUDIO_TOKEN * n

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, object]:
        cfg = self.info.get_hf_config()
        compress_ratio = int(
            _cfg_get(cfg, "speech_tok_compress_ratio", _DEFAULT_COMPRESS_RATIO)
        )
        sr = int(_cfg_get(cfg, "target_sample_rate", _DEFAULT_SAMPLE_RATE))
        max_tokens = min(int(np.ceil(61 * 60 * sr / compress_ratio)) + 3, seq_len)
        max_samples = max_tokens * compress_ratio
        n = mm_counts.get("audio", 0)
        return {"audio": [np.zeros(max_samples, dtype=np.float32) for _ in range(n)]}


def _vibevoice_mm_fields(
    hf_inputs: BatchFeature,
) -> Mapping[str, MultiModalFieldConfig]:
    fields: dict[str, MultiModalFieldConfig] = {}
    if "raw_audio" in hf_inputs:
        fields["raw_audio"] = MultiModalFieldConfig.batched("audio")
    if "raw_audio_lengths" in hf_inputs:
        fields["raw_audio_lengths"] = MultiModalFieldConfig.batched("audio")
    return fields


class VibeVoiceMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=_DEFAULT_SAMPLE_RATE)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", None)
        if audios is not None and "audio" not in mm_data:
            mm_data["audio"] = audios

        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
        result = BatchFeature({"input_ids": [prompt_ids]}, tensor_type="pt")

        raw_audio_list: list = mm_data.get("audio", [])  # type: ignore
        if not raw_audio_list:
            return result

        if isinstance(raw_audio_list, np.ndarray):
            raw_audio_list = [raw_audio_list]
        elif not isinstance(raw_audio_list, list):
            raw_audio_list = list(raw_audio_list)

        max_len = max(len(a) for a in raw_audio_list)
        tensors, lengths = [], []
        for a in raw_audio_list:
            lengths.append(len(a))
            if len(a) < max_len:
                a = np.pad(a, (0, max_len - len(a)))
            tensors.append(torch.from_numpy(a.astype(np.float32)))

        result["raw_audio"] = torch.stack(tensors)
        result["raw_audio_lengths"] = torch.tensor(lengths, dtype=torch.long)
        return result

    def _hf_processor_applies_updates(self, *args, **kwargs) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _vibevoice_mm_fields(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        token_info = self.info.get_audio_token_info()
        audio_token_id = token_info["audio_token_id"]
        if audio_token_id is None:
            return []

        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        def _vid(name: str) -> int | None:
            return vocab.get(name)

        speech_start_id = _vid("<|object_ref_start|>") or _vid("<|speech_start|>")
        speech_end_id = _vid("<|object_ref_end|>") or _vid("<|speech_end|>")
        speech_pad_id = _vid("<|box_start|>") or _vid("<|speech_pad|>")

        cfg = self.info.get_hf_config()
        compress_ratio = int(
            _cfg_get(cfg, "speech_tok_compress_ratio", _DEFAULT_COMPRESS_RATIO)
        )

        out_data = out_mm_kwargs.get_data()
        raw_lengths = out_data.get("raw_audio_lengths", [])

        def _to_int(x: Any) -> int:
            if isinstance(x, torch.Tensor):
                return int(x.item()) if x.numel() == 1 else int(x.shape[0])
            return int(x)

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            if raw_lengths and item_idx < len(raw_lengths):
                audio_len = _to_int(raw_lengths[item_idx])
                n = max(1, int(np.ceil(audio_len / compress_ratio)))
            else:
                n = int(np.ceil(30 * _DEFAULT_SAMPLE_RATE / compress_ratio))

            newline_id = 198  # '\n'
            if speech_start_id and speech_pad_id and speech_end_id:
                ids = (
                    [speech_start_id]
                    + [speech_pad_id] * n
                    + [speech_end_id, newline_id]
                )
                embed_id = speech_pad_id
            else:
                ids = [audio_token_id] * n
                embed_id = audio_token_id

            return PromptUpdateDetails.select_token_id(ids, embed_token_id=embed_id)

        return [
            PromptReplacement(
                modality="audio",
                target=_AUDIO_TOKEN,
                replacement=get_replacement,
            )
        ]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceMultiModalProcessor,
    info=VibeVoiceProcessingInfo,
    dummy_inputs=VibeVoiceDummyInputsBuilder,
)
class VibeVoiceForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsTranscription
):
    """VibeVoice ASR model for vLLM.

    Acoustic + Semantic VAE tokenizers encode 24 kHz audio into embeddings
    that are injected into a Qwen2 causal LM for transcription.
    """

    supported_languages: ClassVar[Mapping[str, str]] = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
        "it": "Italian",
        "nl": "Dutch",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
        "pl": "Polish",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "vi": "Vietnamese",
        "th": "Thai",
    }
    supports_transcription_only: ClassVar[bool] = True

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: str,
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=_DEFAULT_SAMPLE_RATE,
            max_audio_clip_s=61 * 60,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_TOKEN
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.audio_encoder = VibeVoiceAudioEncoder(config)

        decoder_config = _cfg_get(config, "decoder_config", config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=decoder_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        raw_audio = kwargs.get("raw_audio")
        raw_audio_lengths = kwargs.get("raw_audio_lengths")
        if raw_audio is None:
            return []

        device = next(self.audio_encoder.parameters()).device

        # Normalise to list[Tensor]
        if isinstance(raw_audio, torch.Tensor):
            if raw_audio.dim() == 3:
                audio_list = [
                    raw_audio[i].squeeze(0) for i in range(raw_audio.shape[0])
                ]
            elif raw_audio.dim() == 2:
                audio_list = [raw_audio[i] for i in range(raw_audio.shape[0])]
            else:
                audio_list = [raw_audio]
        else:
            audio_list = list(raw_audio)

        if isinstance(raw_audio_lengths, torch.Tensor):
            lengths: list = raw_audio_lengths.tolist()
        else:
            lengths = list(raw_audio_lengths) if raw_audio_lengths else []

        embeddings: list[torch.Tensor] = []
        for i, audio in enumerate(audio_list):
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)
            audio = audio.to(device=device, dtype=torch.float32)
            if lengths and i < len(lengths):
                actual = int(lengths[i])
                if 0 < actual <= audio.shape[-1]:
                    audio = audio[..., :actual]
            if audio.numel() < 160:
                continue
            embeds = self.audio_encoder(audio)  # (1, N, H)
            embeddings.append(embeds.squeeze(0))

        return tuple(embeddings)

    def get_input_embeddings(self) -> nn.Module:
        lm = self.language_model
        if hasattr(lm, "language_model"):
            lm = lm.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
            return lm.model.embed_tokens
        raise AttributeError("Cannot find embed_tokens in language model")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor | list[torch.Tensor] | None = None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        inputs_embeds = self.get_input_embeddings()(input_ids)
        if multimodal_embeddings is not None and is_multimodal is not None:
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds, multimodal_embeddings, is_multimodal
            )
        return inputs_embeds

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.acoustic_tokenizer.": "audio_encoder.acoustic_tokenizer.",
                "model.semantic_tokenizer.": "audio_encoder.semantic_tokenizer.",
                "model.acoustic_connector.": "audio_encoder.acoustic_connector.",
                "model.semantic_connector.": "audio_encoder.semantic_connector.",
                "model.language_model.": "language_model.model.",
                "lm_head.": "language_model.lm_head.",
            }
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if intermediate_tensors is not None:
            inputs_embeds = None

        lm = self.language_model
        if hasattr(lm, "language_model"):
            lm = lm.language_model

        return lm.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


# Keep the training-time class name as a transparent alias so that
# `config.json → architectures: ["VibeVoiceForASRTraining"]` resolves correctly.
VibeVoiceForASRTraining = VibeVoiceForCausalLM
