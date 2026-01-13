# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import functools
import math
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import AttentionSpec

# From https://platform.openai.com/docs/guides/speech-to-text/supported-languages
ISO639_1_SUPPORTED_LANGS = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh",
}


def _pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """Tiny wrapper around F.pad, just to allow for
    reflect padding on small input.
    If this is the case, we insert extra 0 padding
    to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class WhisperCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self._stride = self.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (
            x.shape[-1] - self._effective_kernel_size + self._padding_total
        ) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (
            self._effective_kernel_size - self._padding_total
        )
        extra_padding = target_length - x.shape[-1]
        x = _pad1d(x, (self._padding_total, extra_padding), mode="constant")
        return super().forward(x)


@functools.lru_cache
def create_whisper_attention_backend_with_block_pooling(
    underlying_attn_backend: AttentionBackend, block_pool_size: int
) -> type[AttentionBackend]:
    prefix = "WhisperAttentionWithBlockPooling_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class WhisperAttentionWithBlockPoolingBuilder(underlying_builder):  # type: ignore
        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            assert kv_cache_spec.num_kv_heads % block_pool_size == 0
            kv_cache_spec = replace(
                kv_cache_spec,
                block_size=kv_cache_spec.block_size * block_pool_size,
                num_kv_heads=kv_cache_spec.num_kv_heads // block_pool_size,
            )
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_common_attn_metadata = copy.deepcopy(common_attn_metadata)
            new_common_attn_metadata.query_start_loc *= block_pool_size
            new_common_attn_metadata.query_start_loc_cpu *= block_pool_size
            new_common_attn_metadata.seq_lens *= block_pool_size
            new_common_attn_metadata._seq_lens_cpu *= block_pool_size
            new_common_attn_metadata._num_computed_tokens_cpu *= block_pool_size
            new_common_attn_metadata.num_actual_tokens *= block_pool_size
            new_common_attn_metadata.max_query_len *= block_pool_size
            new_common_attn_metadata.max_seq_len *= block_pool_size
            original_slot_mapping = common_attn_metadata.slot_mapping
            common_prefix_len *= block_pool_size
            new_common_attn_metadata.slot_mapping = (
                (
                    original_slot_mapping.unsqueeze(1) * block_pool_size
                    + torch.arange(block_pool_size, device=original_slot_mapping.device)
                )
                .flatten()
                .clamp(min=-1)
            )
            return super().build(
                common_prefix_len, new_common_attn_metadata, fast_build
            )

    if not issubclass(underlying_attn_backend, FlashAttentionBackend):
        raise NotImplementedError(
            f"{underlying_attn_backend} is not yet supported."
            "Contributions to support more backends are much "
            "appreciated."
        )

    attn_backend = subclass_attention_backend_with_overrides(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        overrides={
            "get_builder_cls": lambda: WhisperAttentionWithBlockPoolingBuilder,
            "get_kv_cache_shape": lambda num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str: (
                2,
                num_blocks,
                # we stretch each block by `block_pool_size`
                block_size * block_pool_size,
                num_kv_heads // block_pool_size,
                head_size,
            ),  # TODO: generalize to other backends
        },
    )

    return attn_backend


class WhisperAttentionWithBlockPooling(Attention):
    """Attention layer with block pooling."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        block_pool_size: int = 1,
        attn_backend: type[AttentionBackend] | None = None,
        **extra_impl_args,
    ) -> None:
        self.block_pool_size = block_pool_size
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        underlying_attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            attn_type=attn_type,
        )
        attn_backend = create_whisper_attention_backend_with_block_pooling(
            underlying_attn_backend, block_pool_size
        )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=logits_soft_cap,
            per_layer_sliding_window=per_layer_sliding_window,
            prefix=prefix,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=attn_backend,
            **extra_impl_args,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        kv_cache_spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(kv_cache_spec, AttentionSpec)
        kv_cache_spec = replace(
            kv_cache_spec,
            num_kv_heads=self.block_pool_size * kv_cache_spec.num_kv_heads,
        )
        return kv_cache_spec
