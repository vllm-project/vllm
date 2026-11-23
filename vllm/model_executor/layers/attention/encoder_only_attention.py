# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from copy import copy

import torch

from vllm.config import CacheConfig
from vllm.config.vllm import VllmConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheLayerConfig

logger = init_logger(__name__)


@functools.lru_cache
def create_encoder_only_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    prefix = "EncoderOnlyAttention_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class EncoderOnlyAttentionBuilder(underlying_builder):  # type: ignore
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_common_attn_metadata = copy(common_attn_metadata)
            new_common_attn_metadata.causal = False
            return super().build(
                common_prefix_len, new_common_attn_metadata, fast_build
            )

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=EncoderOnlyAttentionBuilder,
    )

    return attn_backend


def _warn_given_attn_backend(attn_backend: type[AttentionBackend]) -> None:
    vllm_config = get_current_vllm_config_or_none()
    requested = "auto"
    if vllm_config is not None:
        attention_config = vllm_config.attention_config
        if attention_config.backend is not None:
            requested = attention_config.backend.name
        if attention_config.backend_per_kind:
            requested += f" (per kind: {attention_config.backend_per_kind})"
    module, qualname = attn_backend.full_cls_name()
    logger.warning_once(
        "EncoderOnlyAttention uses the model's attention backend %s.%s instead "
        "of the selected one (--attention-backend: %s); the selector's "
        "validation is skipped for it.",
        module,
        qualname,
        requested,
    )


class EncoderOnlyAttention(Attention):
    """Encoder attention is a special case that doesn't need a KV Cache."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        cache_config: CacheConfig | None = None,
        attn_type: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        **kwargs,
    ):
        """`attn_backend` replaces the selected backend (e.g. a model's own
        subclass); it is wrapped for non-causal metadata like the default.
        A given backend skips the backend selector and its validation, and
        overrides `--attention-backend`: checking that it supports the head
        size, dtype, KV cache dtype and platform is the caller's
        responsibility. A warning names the backend that is used. Its
        `get_name()` must be the name of a registered `AttentionBackendEnum`
        member.

        Of the in-tree models, only the BERT classes (`BertModel`,
        `BertPoolingModel` and their layers) pass `attn_backend` through."""
        dtype = torch.get_default_dtype()

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"

        if attn_backend is None:
            attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                attn_type=AttentionType.ENCODER_ONLY,
            )
        else:
            _warn_given_attn_backend(attn_backend)

        if attn_type is not None:
            assert attn_type == AttentionType.ENCODER_ONLY, (
                "EncoderOnlyAttention only supports AttentionType.ENCODER_ONLY"
            )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=create_encoder_only_attention_backend(attn_backend),
            attn_type=AttentionType.ENCODER_ONLY,
            **kwargs,
        )

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig | None:
        # Does not need KV cache
        return None
